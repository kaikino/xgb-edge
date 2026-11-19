"""Command line interface: ``xgb-edge export model.json -o out/``."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import __version__
from .export import ExportOptions, export_model
from .model import XgbEdgeError
from .report import DEFAULT_CFLAGS, add_flash_totals, find_cross_compiler, measure, render


def load_matrix(path: str) -> np.ndarray:
    """Load a feature matrix from .npy, .npz (first array) or .csv (empty cell = missing)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            return np.load(path, allow_pickle=False)
        if ext == ".npz":
            with np.load(path, allow_pickle=False) as z:
                return z[z.files[0]]
        if ext in (".csv", ".txt"):
            data = np.genfromtxt(path, delimiter=",", dtype=np.float64)
            return data.reshape(1, -1) if data.ndim == 1 else data
    except (OSError, ValueError) as e:
        raise XgbEdgeError(f"cannot read {path}: {e}") from None
    raise XgbEdgeError(f"unsupported data file {path!r}: use .npy, .npz or .csv")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xgb-edge", description=__doc__)
    p.add_argument("--version", action="version", version=f"xgb-edge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, help_ in (("export", "generate C and print the size/cost report"),
                       ("report", "print the size/cost report without writing C")):
        s = sub.add_parser(cmd, help=help_)
        s.add_argument("model", help="XGBoost model saved with save_model('model.json')")
        if cmd == "export":
            s.add_argument("-o", "--out", default=".", help="output directory (default: current)")
        s.add_argument("--name", default="xgb_model", help="C identifier prefix and file name (default: xgb_model)")
        s.add_argument("--validation", metavar="FILE",
                       help="feature matrix (.npy/.npz/.csv, NaN or empty = missing) used to check tie risk")
        s.add_argument("--leaf-bits", default="auto",
                       help="fixed-point leaf width: auto (default), or 2..32")
        s.add_argument("--input", default="f32", choices=["f32", "i8", "u8", "i16", "u16", "i32"],
                       help="type of the feature array passed to predict (default: f32)")
        s.add_argument("--leaf-store", default="auto", choices=["auto", "inline", "table"],
                       help="store leaf values in the nodes or in a shared table (default: smaller of the two)")
        s.add_argument("--cc", help="cross compiler used to measure code size and stack "
                                    "(default: arm-none-eabi-gcc if found)")
        s.add_argument("--cflags", default=DEFAULT_CFLAGS, help=f"flags for --cc (default: {DEFAULT_CFLAGS})")
        s.add_argument("--no-measure", action="store_true", help="do not run a compiler; estimate code size")
        s.add_argument("--report-json", metavar="FILE", help="also write the report as JSON")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        opt = ExportOptions(name=args.name, input=args.input, leaf_bits=args.leaf_bits,
                            leaf_store=args.leaf_store)
        if opt.leaf_bits != "auto":
            try:
                opt.leaf_bits = int(opt.leaf_bits)
            except ValueError:
                raise XgbEdgeError(f"--leaf-bits must be 'auto' or an integer, got {args.leaf_bits!r}") from None
        X = load_matrix(args.validation) if args.validation else None
        exp = export_model(args.model, opt, X)
        measured = None
        if not args.no_measure:
            cc = args.cc or find_cross_compiler()
            if cc:
                try:
                    measured = measure(exp.generated.header, exp.generated.source, exp.generated.name,
                                       cc, args.cflags)
                except (OSError, RuntimeError) as e:
                    print(f"warning: could not measure with {cc}: {e}", file=sys.stderr)
        add_flash_totals(exp.report, measured)
        if args.cmd == "export":
            paths = exp.write(args.out)
        sys.stdout.write(render(exp.report))
        if args.cmd == "export":
            print("wrote " + ", ".join(paths))
        if args.report_json:
            with open(args.report_json, "w") as f:
                json.dump(exp.report, f, indent=2)
    except XgbEdgeError as e:
        print(f"xgb-edge: error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
