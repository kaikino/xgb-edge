"""Pack the trees into the byte stream walked by the generated C interpreter.

Only split nodes are stored.  Each tree's split nodes are laid out in
pre-order, so a child that is itself a split node needs no pointer when it is
the *next* node in the stream.  A node is

    header   1 byte  = type(2 bits) << 6 | slot(6 bits)          (<= 64 slots)
          or 2 bytes = type(2 bits) << 14 | slot(14 bits), little endian
    code     1 or 2 bytes: threshold rank; go left iff q[slot] < code
    operands by type:
      II  both children are split nodes : OFF  = byte length of the left subtree
                                           (left child follows; right child is OFF bytes later)
      LI  left child is a leaf          : LEAF (left);  right child follows
      IL  right child is a leaf         : LEAF (right); left child follows
      LL  both children are leaves      : LEAF (left), LEAF (right)

A LEAF operand is either the fixed-point leaf value itself ("inline") or an
index into a shared table of the distinct leaf values ("table"), whichever
makes the model smaller.  All multi-byte fields are little endian and
unaligned, so the stream is identical on every target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .model import Model, UnsupportedModelError, XgbEdgeError
from .quantize import LeafPlan, ThresholdPlan

TYPE_II, TYPE_LI, TYPE_IL, TYPE_LL = 0, 1, 2, 3
TYPE_NAMES = ("II", "LI", "IL", "LL")
NARROW_SLOTS = 64


@dataclass
class Flat:
    """All stored split nodes of all stored trees, in stream order."""

    stored_trees: List[int]     # model tree index of each stored tree, in stream order
    group_end: List[int]        # per group: number of stored trees in groups <= g
    tree_start: np.ndarray      # int64[n_stored + 1]: first node of each stored tree
    ntype: np.ndarray
    slot: np.ndarray
    code: np.ndarray
    leaf_a: np.ndarray          # the leaf child (LI, IL) or the left leaf (LL)
    leaf_b: np.ndarray          # the right leaf (LL)
    right_pos: np.ndarray       # II: stream index of the right child

    @property
    def n_nodes(self) -> int:
        return int(self.ntype.size)

    @property
    def leaf_refs(self) -> np.ndarray:
        """Every leaf operand in the stream (order irrelevant)."""
        a = self.leaf_a[self.ntype != TYPE_II]
        b = self.leaf_b[self.ntype == TYPE_LL]
        return np.concatenate([a, b])


def flatten(model: Model, tplan: ThresholdPlan, lplan: LeafPlan) -> Flat:
    stored, group_end = [], []
    parts = {k: [] for k in ("ntype", "slot", "code", "leaf_a", "leaf_b", "right_pos")}
    tree_start = [0]
    base = 0
    for g in range(model.n_groups):
        for ti, tree in enumerate(model.trees):
            if tree.group != g or lplan.constant_tree[ti]:
                continue
            internal = tree.internal_nodes  # pre-order
            m = internal.size
            pos = np.full(tree.left.size, -1, dtype=np.int64)
            pos[internal] = base + np.arange(m)
            l, r = tree.left[internal], tree.right[internal]
            l_leaf, r_leaf = tree.left[l] < 0, tree.left[r] < 0
            ntype = np.where(l_leaf & r_leaf, TYPE_LL,
                             np.where(l_leaf, TYPE_LI, np.where(r_leaf, TYPE_IL, TYPE_II)))
            q = lplan.q[ti]
            own = base + np.arange(m)
            # the layout relies on these pre-order facts
            nxt = np.where(l_leaf, pos[r], pos[l])
            follows = ntype != TYPE_LL
            assert np.array_equal(nxt[follows], own[follows] + 1), "pre-order invariant violated"
            parts["ntype"].append(ntype)
            parts["slot"].append(tplan.node_slot[ti][internal])
            parts["code"].append(tplan.node_code[ti][internal])
            parts["leaf_a"].append(np.where(l_leaf, q[l], np.where(r_leaf, q[r], 0)))
            parts["leaf_b"].append(np.where(l_leaf & r_leaf, q[r], 0))
            parts["right_pos"].append(np.where(ntype == TYPE_II, pos[r], -1))
            stored.append(ti)
            base += m
            tree_start.append(base)
        group_end.append(len(stored))
    cat = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)).astype(np.int64)
           for k, v in parts.items()}
    return Flat(stored_trees=stored, group_end=group_end,
                tree_start=np.asarray(tree_start, dtype=np.int64), **cat)


def _signed_bytes(lo: int, hi: int) -> int:
    for w in (1, 2, 3, 4):
        if -(1 << (8 * w - 1)) <= lo and hi <= (1 << (8 * w - 1)) - 1:
            return w
    raise UnsupportedModelError(f"leaf value range [{lo}, {hi}] does not fit in 32 bits")


def _unsigned_bytes(v: int) -> int:
    for w in (1, 2, 3, 4):
        if v < (1 << (8 * w)):
            return w
    raise UnsupportedModelError(f"value {v} does not fit in 32 bits")


def _array_width(v: int) -> int:
    """Element width (1, 2 or 4 bytes) of an unsigned C array holding values up to v."""
    w = _unsigned_bytes(v)
    return 4 if w == 3 else w


@dataclass
class Packed:
    flat: Flat
    hdr_bytes: int
    code_bytes: int
    off_bytes: int
    leaf_mode: str                    # "inline" or "table"
    leaf_bytes: int                   # width of a LEAF operand in the stream
    leaf_table: Optional[np.ndarray]  # distinct leaf values (table mode)
    leaf_table_bytes: int             # element width of the table (1, 2 or 4)
    nodes: np.ndarray                 # uint8 stream
    tree_sizes: np.ndarray            # bytes of each stored tree
    type_counts: List[int]

    @property
    def tree_size_bytes(self) -> int:
        return _array_width(int(self.tree_sizes.max())) if self.tree_sizes.size else 1

    @property
    def group_end_bytes(self) -> int:
        return _array_width(len(self.flat.stored_trees))

    @property
    def table_total_bytes(self) -> int:
        return 0 if self.leaf_table is None else int(self.leaf_table.size) * self.leaf_table_bytes


def _layout(flat: Flat, hdr: int, code: int, leaf: int, min_off: int):
    """Node sizes/positions for given widths, with the smallest consistent offset width."""
    is_ii = flat.ntype == TYPE_II
    is_ll = flat.ntype == TYPE_LL
    for off in (1, 2, 3, 4):
        if off < min_off:
            continue
        size = np.full(flat.n_nodes, hdr + code, dtype=np.int64)
        size += np.where(is_ii, off, np.where(is_ll, 2 * leaf, leaf))
        pos = np.concatenate([[0], np.cumsum(size)])
        offsets = np.zeros(flat.n_nodes, dtype=np.int64)
        if is_ii.any():
            offsets[is_ii] = pos[flat.right_pos[is_ii]] - (pos[:-1][is_ii] + size[is_ii])
            assert (offsets[is_ii] > 0).all()
        if int(offsets.max(initial=0)) < (1 << (8 * off)):
            return off, size, pos, offsets
    raise UnsupportedModelError("a subtree is larger than 4 GiB")  # pragma: no cover


def _put(out: np.ndarray, at: np.ndarray, values: np.ndarray, width: int) -> None:
    v = values.astype(np.int64) & ((1 << (8 * width)) - 1)
    for k in range(width):
        out[at + k] = (v >> (8 * k)) & 0xFF


def pack(model: Model, tplan: ThresholdPlan, lplan: LeafPlan, *, leaf_store: str = "auto",
         min_hdr_bytes: int = 1, min_off_bytes: int = 1, min_leaf_bytes: int = 1) -> Packed:
    """Choose field widths and emit the node stream."""
    if leaf_store not in ("auto", "inline", "table"):
        raise XgbEdgeError(f"leaf_store must be auto, inline or table, got {leaf_store!r}")
    flat = flatten(model, tplan, lplan)
    hdr = max(min_hdr_bytes, 1 if tplan.n_slots <= NARROW_SLOTS else 2)
    code = tplan.code_bytes
    refs = flat.leaf_refs

    candidates = []
    if refs.size:
        if leaf_store in ("auto", "inline"):
            w = max(min_leaf_bytes, _signed_bytes(int(refs.min()), int(refs.max())))
            candidates.append(("inline", w, None, 0))
        if leaf_store in ("auto", "table"):
            table = np.unique(refs)
            w = max(min_leaf_bytes, _unsigned_bytes(int(table.size) - 1))
            tw = _signed_bytes(int(table.min()), int(table.max()))
            candidates.append(("table", w, table, 4 if tw == 3 else tw))
    else:
        candidates.append(("inline", max(min_leaf_bytes, 1), None, 0))

    best = None
    for mode, leaf_w, table, table_w in candidates:
        off, size, pos, offsets = _layout(flat, hdr, code, leaf_w, min_off_bytes)
        total = int(pos[-1]) + (0 if table is None else int(table.size) * table_w)
        if best is None or total < best[0]:  # ties keep the first candidate (inline)
            best = (total, mode, leaf_w, table, table_w, off, size, pos, offsets)
    _, mode, leaf_w, table, table_w, off, size, pos, offsets = best

    if int(pos[-1]) >= 2 ** 32:
        raise UnsupportedModelError("the packed model is larger than 4 GiB")
    out = np.zeros(int(pos[-1]), dtype=np.uint8)
    at = pos[:-1]
    if hdr == 1:
        assert tplan.n_slots <= NARROW_SLOTS
        _put(out, at, (flat.ntype << 6) | flat.slot, 1)
    else:
        _put(out, at, (flat.ntype << 14) | flat.slot, 2)
    _put(out, at + hdr, flat.code, code)
    op = at + hdr + code
    is_ii = flat.ntype == TYPE_II
    is_ll = flat.ntype == TYPE_LL
    has_leaf = ~is_ii
    _put(out, op[is_ii], offsets[is_ii], off)
    if mode == "inline":
        a_val, b_val = flat.leaf_a, flat.leaf_b
    else:
        a_val, b_val = np.searchsorted(table, flat.leaf_a), np.searchsorted(table, flat.leaf_b)
    _put(out, op[has_leaf], a_val[has_leaf], leaf_w)
    _put(out, op[is_ll] + leaf_w, b_val[is_ll], leaf_w)

    tree_sizes = np.diff(pos[flat.tree_start])
    counts = [int((flat.ntype == k).sum()) for k in range(4)]
    return Packed(flat=flat, hdr_bytes=hdr, code_bytes=code, off_bytes=off, leaf_mode=mode,
                  leaf_bytes=leaf_w, leaf_table=table, leaf_table_bytes=table_w, nodes=out,
                  tree_sizes=tree_sizes, type_counts=counts)
