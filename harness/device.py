"""On-device harness for the Seeed XIAO nRF52840: build, flash, run every test row.

    python -m harness.device list
    python -m harness.device run MODEL [--variant N] [--kind xgbe|float] [--board A|B]
    python -m harness.device suite [--board A]
    python -m harness.device example [--board A]     (the packaged Arduino example)

For a zoo model it exports C (or generates the float baseline), builds the
test firmware in firmware/xgbe_bench around it, uploads it to the chosen board
(found by USB serial number, never by a fixed port name), sends every test row
over USB serial and compares what the board returns with xgboost and with the
tool's own exact simulation.  Inference time is measured on the board.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import zoo  # noqa: E402
from harness.parity import rows_for_input, variant_label  # noqa: E402
from xgb_edge import ExportOptions, export_model, load  # noqa: E402
from xgb_edge.report import find_cross_compiler, measure  # noqa: E402

FQBN = "Seeeduino:nrf52:xiaonRF52840SensePlus"
BOARDS = {"A": "C8B9EFDEFE7136C7", "B": "9857559A47B38970"}
FLASH_MAX_CORE = 811008     # what the Arduino core's boards.txt advertises
FLASH_MAX = 798720          # what the bootloader really accepts over DFU (measured: see harness/flash_limit.py)
SKETCH = os.path.join(zoo.ROOT, "firmware", "xgbe_bench")
DEVDIR = os.path.join(zoo.ROOT, "build", "device")
CORTEX_FLAGS = "-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16"
MODEL_NAME = "xgbm"  # C prefix of the model under test (must not collide with core symbols)


class DeviceError(RuntimeError):
    pass


def _env() -> Dict[str, str]:
    """Environment for arduino-cli: the Seeed core's post-build step runs plain `python`."""
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    return env


def board_serial(board: str) -> str:
    return BOARDS.get(board.upper(), board)


def list_ports() -> Dict[str, str]:
    """USB serial number -> current port, for every attached XIAO."""
    r = subprocess.run(["arduino-cli", "board", "list", "--format", "json"], capture_output=True, text=True)
    out: Dict[str, str] = {}
    try:
        doc = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return out
    for p in doc.get("detected_ports", []):
        port = p.get("port", {})
        sn = port.get("properties", {}).get("serialNumber") or port.get("hardware_id")
        if sn and port.get("address", "").startswith("/dev/"):
            out[sn] = port["address"]
    return out


def find_port(serial: str, wait: float = 0.0) -> Optional[str]:
    end = time.time() + wait
    while True:
        port = list_ports().get(serial)
        if port or time.time() >= end:
            return port
        time.sleep(0.5)


@contextlib.contextmanager
def locked(name: str):
    """Cross-process lock so two runs never use one board (or upload) at the same time."""
    os.makedirs(DEVDIR, exist_ok=True)
    f = open(os.path.join(DEVDIR, f".{name}.lock"), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


# --------------------------------------------------------------------------- build

MUT_H = """#ifndef MUT_H
#define MUT_H
#define MUT_NUM_FEATURES {nf}
#define MUT_NUM_MARGINS {nm}
#define MUT_FEATURE_BYTES {fb}
#define MUT_MARGIN_IS_FLOAT {isf}
#define MUT_BUILD_ID 0x{bid:08X}UL
int mut_predict(const void *features, void *margins);
#endif
"""


def make_sketch(tag: str, files: Dict[str, str], nf: int, nm: int, feat_bytes: int, margin_float: bool,
                mut_body: str, pad_bytes: int = 0, opt: str = "-Os") -> Tuple[str, int]:
    """Copy the firmware sketch to build/device/<tag>/ and add the model under test."""
    d = os.path.join(DEVDIR, tag, "xgbe_bench")
    os.makedirs(d, exist_ok=True)
    # the id identifies the binary: sources *and* optimisation level, so that a board still running
    # the same sources built with another flag is not mistaken for up to date
    blob = "".join(files[k] for k in sorted(files)) + mut_body + open(os.path.join(SKETCH, "xgbe_bench.ino")).read() \
        + str(pad_bytes) + opt
    build_id = zlib.crc32(blob.encode()) & 0xFFFFFFFF
    wanted = dict(files)
    wanted["xgbe_bench.ino"] = open(os.path.join(SKETCH, "xgbe_bench.ino")).read()
    wanted["mut.h"] = MUT_H.format(nf=nf, nm=nm, fb=feat_bytes, isf=int(margin_float), bid=build_id)
    if pad_bytes:
        wanted["mut.h"] = wanted["mut.h"].replace("#endif", f"#define MUT_PAD_BYTES {pad_bytes}\n#endif")
    wanted["mut.c"] = mut_body
    for stale in os.listdir(d):
        if stale not in wanted:
            os.remove(os.path.join(d, stale))
    for fn, text in wanted.items():
        path = os.path.join(d, fn)
        if not os.path.exists(path) or open(path).read() != text:
            with open(path, "w") as f:
                f.write(text)
    return d, build_id


def compile_sketch(sketch_dir: str, opt: str = "-Os") -> Dict:
    bp = os.path.join(os.path.dirname(sketch_dir), "build" + opt.replace("-", "_"))
    cmd = ["arduino-cli", "compile", "--fqbn", FQBN, "--build-path", bp, "--warnings", "all",
           "--build-property", f"compiler.optimization_flag={opt}", sketch_dir]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env())
    text = r.stdout + "\n" + r.stderr
    if r.returncode != 0:
        raise DeviceError("arduino-cli compile failed:\n" + text[-4000:])
    m = re.search(r"Sketch uses (\d+) bytes .*?Maximum is (\d+) bytes", text, re.S)
    g = re.search(r"Global variables use (\d+) bytes", text)
    ours = [ln for ln in text.splitlines()
            if "warning" in ln and re.search(r"(xgbe_bench\.ino|mut\.c|/xgbe_bench/[A-Za-z0-9_]+\.[ch])", ln)]
    elf = os.path.join(bp, "xgbe_bench.ino.elf")
    return {"build_path": bp, "elf": elf, "program_bytes": int(m.group(1)) if m else None,
            "max_bytes": int(m.group(2)) if m else None, "ram_bytes": int(g.group(1)) if g else None,
            "warnings": ours, "compile_seconds": round(time.time() - t0, 1), "opt": opt}


def elf_symbols(elf: str, prefix: str) -> Dict[str, int]:
    """Sizes of the linked symbols whose name starts with ``prefix``."""
    cc = find_cross_compiler()
    nm = re.sub(r"gcc$", "nm", cc)
    r = subprocess.run([nm, "-S", elf], capture_output=True, text=True, check=True)
    out: Dict[str, int] = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[3].startswith(prefix):
            out[parts[3]] = int(parts[1], 16)
    return out


def upload(serial: str, build_path: str, retries: int = 3) -> None:
    last = ""
    for attempt in range(retries):
        port = find_port(serial, wait=20)
        if not port:
            raise DeviceError(f"board {serial} is not attached (ports: {list_ports()})")
        with locked("upload"):
            r = subprocess.run(["arduino-cli", "upload", "-p", port, "--fqbn", FQBN, "--input-dir", build_path],
                               capture_output=True, text=True, env=_env())
        last = r.stdout + r.stderr
        if r.returncode == 0 and "Failed to upgrade target" not in last:
            time.sleep(1.5)
            if find_port(serial, wait=30):
                return
            last += "\nboard did not come back after upload"
        time.sleep(3)
    raise DeviceError(f"upload to {serial} failed after {retries} attempts:\n{last[-3000:]}")


# --------------------------------------------------------------------------- protocol

def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


try:  # table-driven CRC for speed
    import binascii

    def crc16(data: bytes, crc: int = 0xFFFF) -> int:  # noqa: F811
        return binascii.crc_hqx(data, crc)
except ImportError:  # pragma: no cover
    pass


class Link:
    def __init__(self, serial_number: str):
        import serial as pyserial
        self.sn = serial_number
        port = find_port(serial_number, wait=30)
        if not port:
            raise DeviceError(f"board {serial_number} not found")
        self.ser = pyserial.Serial(port, 115200, timeout=0.5, write_timeout=5)
        self.seq = 0
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def close(self):
        with contextlib.suppress(Exception):
            self.ser.close()

    def _read(self, n: int, deadline: float) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            buf += chunk
            if not chunk and time.time() > deadline:
                raise TimeoutError(f"timeout waiting for {n} bytes (got {len(buf)})")
        return buf

    def request(self, typ: bytes, payload: bytes = b"", timeout: float = 5.0, retries: int = 3):
        last: Exception = DeviceError("no attempt")
        for _ in range(retries):
            self.seq = (self.seq + 1) & 0xFF
            body = typ + bytes([self.seq]) + struct.pack("<H", len(payload)) + payload
            frame = b"\xA5\x5A" + body + struct.pack("<H", crc16(body))
            try:
                self.ser.reset_input_buffer()
                self.ser.write(frame)
                self.ser.flush()
                deadline = time.time() + timeout
                # hunt for the sync bytes
                state = 0
                while state < 2:
                    b = self._read(1, deadline)
                    state = state + 1 if b[0] == (0xA5, 0x5A)[state] else (1 if b[0] == 0xA5 else 0)
                head = self._read(4, deadline)
                rtyp, rseq, rlen = head[0:1], head[1], struct.unpack("<H", head[2:4])[0]
                data = self._read(rlen, deadline)
                rcrc = struct.unpack("<H", self._read(2, deadline))[0]
                if rcrc != crc16(head + data):
                    raise DeviceError("bad CRC from board")
                if rseq != self.seq:
                    raise DeviceError(f"sequence mismatch ({rseq} != {self.seq})")
                if rtyp == b"e":
                    raise DeviceError(f"board reported error code {data[0] if data else '?'}")
                return rtyp, data
            except (TimeoutError, DeviceError, OSError) as e:
                last = e
                time.sleep(0.2)
        raise DeviceError(f"request {typ!r} failed: {last}")

    def info(self) -> Dict:
        typ, d = self.request(b"I")
        if typ != b"i" or d[:4] != b"XGBE":
            raise DeviceError("unexpected info reply")
        nf, nm = struct.unpack("<HH", d[8:12])
        bid, = struct.unpack("<I", d[12:16])
        mp, = struct.unpack("<H", d[16:18])
        t_over, c_over = struct.unpack("<II", d[18:26])
        return {"proto": d[4], "feature_bytes": d[5], "margin_float": bool(d[6]), "num_features": nf,
                "num_margins": nm, "build_id": bid, "max_payload": mp, "tick_overhead": t_over,
                "cycle_overhead": c_over}

    def stack(self, row: bytes) -> int:
        typ, d = self.request(b"S", row, timeout=20)
        return struct.unpack("<I", d[:4])[0]

    def predict(self, rows: np.ndarray, n_margins: int, reps: int, max_payload: int, per_row_s: float = 0.05,
                progress: bool = False):
        n = rows.shape[0]
        row_bytes = rows.shape[1] * rows.dtype.itemsize
        res_bytes = 12 + 4 * n_margins
        batch = max(1, min(255, (max_payload - 2) // max(row_bytes, 1), max_payload // res_bytes))
        if row_bytes + 2 > max_payload:
            raise DeviceError("a feature row does not fit in one frame")
        cls = np.zeros(n, dtype=np.int64)
        ticks = np.zeros(n, dtype=np.int64)
        cycles = np.zeros(n, dtype=np.int64)
        margins = np.zeros((n, n_margins), dtype=np.uint32)
        t0 = time.time()
        for s in range(0, n, batch):
            part = rows[s:s + batch]
            payload = bytes([reps, part.shape[0]]) + part.tobytes()
            typ, d = self.request(b"P", payload, timeout=5 + part.shape[0] * reps * per_row_s * 4)
            if typ != b"p" or len(d) != part.shape[0] * res_bytes:
                raise DeviceError(f"unexpected predict reply ({typ!r}, {len(d)} bytes)")
            a = np.frombuffer(d, dtype=np.uint8).reshape(part.shape[0], res_bytes)
            head = a[:, :12].copy().view("<u4")
            cls[s:s + part.shape[0]] = head[:, 0].astype(np.int32)
            ticks[s:s + part.shape[0]] = head[:, 1]
            cycles[s:s + part.shape[0]] = head[:, 2]
            margins[s:s + part.shape[0]] = a[:, 12:].copy().view("<u4")
            if progress and (s // batch) % 20 == 0:
                print(f"    {s + part.shape[0]}/{n} rows", flush=True)
        return cls, margins, ticks, cycles, time.time() - t0


# --------------------------------------------------------------------------- runs

def timing_stats(ticks: np.ndarray, cycles: np.ndarray, info: Dict) -> Dict:
    us_t = (ticks - info["tick_overhead"]) / 16.0
    us_c = (cycles - info["cycle_overhead"]) / 64.0
    return {"us_mean": float(us_t.mean()), "us_median": float(np.median(us_t)), "us_min": float(us_t.min()),
            "us_max": float(us_t.max()), "us_p99": float(np.percentile(us_t, 99)),
            "cycles_mean": float(cycles.mean() - info["cycle_overhead"]),
            "us_mean_from_cycle_counter": float(us_c.mean())}


def flash_and_open(serial: str, sketch_dir: str, build: Dict, build_id: int, force: bool = False) -> Tuple[Link, Dict]:
    """Make sure the board runs exactly this build; upload only if it does not already."""
    if not force:
        link = None
        try:
            link = Link(serial)
            typ, d = link.request(b"I", timeout=2.0, retries=1)
            if typ == b"i" and d[:4] == b"XGBE" and struct.unpack("<I", d[12:16])[0] == build_id:
                return link, link.info()
        except (DeviceError, OSError):
            pass
        if link is not None:
            link.close()
    upload(serial, build["build_path"])
    last: Exception = DeviceError("no attempt")
    for _ in range(10):
        try:
            link = Link(serial)
            info = link.info()
            if info["build_id"] != build_id:
                link.close()
                raise DeviceError(f"board runs build {info['build_id']:08X}, expected {build_id:08X}")
            return link, info
        except (DeviceError, OSError) as e:
            last = e
            time.sleep(1.5)
    raise DeviceError(f"board {serial} does not answer after upload: {last}")


def run_model(model_name: str, variant: int = 0, kind: str = "xgbe", board: str = "A", opt: str = "-Os",
              reps: int = 3, max_rows: Optional[int] = None, force_upload: bool = False,
              quiet: bool = False, label: str = "") -> Dict:
    serial = board_serial(board)
    spec = zoo.by_name(model_name)
    tr = zoo.train(spec)
    model = load(tr.model_path)
    opts = spec.exports[variant] if kind == "xgbe" else {}
    options = ExportOptions(name=MODEL_NAME, **opts)
    # The float baseline gets exactly the rows the xgb-edge export of this model gets (for an
    # integer-input export: the integer rows), so that the two are compared on the same data.
    row_input = ExportOptions(**spec.exports[variant]).input
    sel = rows_for_input(tr.X, row_input)
    if max_rows:
        sel = sel[:max_rows]
    X = tr.X[sel]
    xgb_class, xgb_margin = tr.xgb_class[sel], tr.xgb_margin[sel]
    res: Dict = {"model": model_name, "kind": kind, "variant": variant_label(opts) if kind == "xgbe" else "float32 baseline",
                 "board": serial, "opt": opt, "rows": int(sel.size), "rows_total": int(tr.X.shape[0]),
                 "input": options.input if kind == "xgbe" else "f32", "row_set": row_input, "failures": []}
    fails: List[str] = res["failures"]
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))

    if kind == "xgbe":
        exp = export_model(model, options, validation=X)
        files = {MODEL_NAME + ".h": exp.generated.header, MODEL_NAME + ".c": exp.generated.source}
        dev_in = exp.device_inputs(X)
        body = (f'#include "mut.h"\n#include "{MODEL_NAME}.h"\n'
                "int mut_predict(const void *features, void *margins)\n"
                f"{{\n    return {MODEL_NAME}_predict((const {MODEL_NAME}_feature_t *)features, (int32_t *)margins);\n}}\n")
        margin_float = False
        tool = measure(exp.generated.header, exp.generated.source, MODEL_NAME, find_cross_compiler(),
                       f"{CORTEX_FLAGS} {opt}")
        res["tool_report"] = {"data_bytes": exp.report["flash"]["data_bytes"], "code_bytes": tool["code_bytes"],
                              "rodata_bytes": tool["rodata_bytes"],
                              "total_bytes": tool["rodata_bytes"] + tool["code_bytes"],
                              "stack_bytes": tool["stack_bytes"],
                              "comparisons_worst_case": exp.report["comparisons_worst_case"]["total"],
                              "leaf_bits": exp.lplan.bits, "bytes_per_split": exp.report["layout"]["bytes_per_split"]}
    elif kind == "float":
        from bench import float_baseline
        header, source = float_baseline.generate(tr.model_path, MODEL_NAME)
        files = {MODEL_NAME + ".h": header, MODEL_NAME + ".c": source}
        dev_in = np.ascontiguousarray(X.astype(np.float32))
        body = (f'#include "mut.h"\n#include "{MODEL_NAME}.h"\n'
                "int mut_predict(const void *features, void *margins)\n"
                f"{{\n    return {MODEL_NAME}_predict((const float *)features, (float *)margins);\n}}\n")
        margin_float = True
    else:
        raise ValueError(kind)

    nf, nm = model.n_features, model.n_groups
    tag = f"{model_name}__{kind}{variant if kind == 'xgbe' else ''}"
    sketch, build_id = make_sketch(tag, files, nf, nm, dev_in.dtype.itemsize, margin_float, body, opt=opt)
    log(f"[{model_name} {res['variant']} on {board}] compiling ({opt}) ...")
    build = compile_sketch(sketch, opt)
    res["build"] = {k: build[k] for k in ("program_bytes", "max_bytes", "ram_bytes", "compile_seconds", "warnings")}
    res["build"]["bootloader_limit"] = FLASH_MAX
    if build["program_bytes"] and build["program_bytes"] > FLASH_MAX:
        raise DeviceError(f"firmware is {build['program_bytes']} bytes; the bootloader accepts at most {FLASH_MAX} "
                          f"(the Arduino core's limit of {FLASH_MAX_CORE} is not reachable over DFU)")
    if build["warnings"]:
        fails.append("compiler warnings in generated/harness files: " + "; ".join(build["warnings"][:3]))
    syms = elf_symbols(build["elf"], MODEL_NAME + "_")
    res["linked_model_bytes"] = int(sum(syms.values()))
    res["linked_symbols"] = syms
    if kind == "xgbe":
        tr_ = res["tool_report"]
        res["size_matches_tool"] = res["linked_model_bytes"] == tr_["total_bytes"]
        if not res["size_matches_tool"]:
            fails.append(f"tool reported {tr_['total_bytes']} bytes, linked image holds {res['linked_model_bytes']}")

    with locked("board_" + serial):
        log(f"  program {build['program_bytes']} / {build['max_bytes']} bytes; uploading to {serial} ...")
        link, info = flash_and_open(serial, sketch, build, build_id, force_upload)
        try:
            if info["num_features"] != nf or info["num_margins"] != nm:
                raise DeviceError("board firmware reports a different model shape")
            # deepest stack use over a handful of rows (different rows walk different paths)
            probe = dev_in[np.linspace(0, dev_in.shape[0] - 1, num=min(8, dev_in.shape[0]), dtype=int)]
            res["stack_measured"] = max(link.stack(row.tobytes()) for row in probe)
            est = 0.0005 + 2e-6 * model.n_internal ** 0.5 * len(model.trees) ** 0.5
            cls, raw, ticks, cycles, secs = link.predict(dev_in, nm, reps, info["max_payload"], per_row_s=max(est, 0.002),
                                                         progress=not quiet)
        finally:
            link.close()
    res["transfer_seconds"] = round(secs, 1)
    res["timing"] = timing_stats(ticks, cycles, info)

    differ = cls != xgb_class
    res["differ_from_xgboost"] = int(differ.sum())
    res["code"] = zoo.code_stamp("device")
    res["build_id"] = f"{build_id:08X}"
    if kind == "xgbe":
        ev = exp.evaluation
        margins = raw.view(np.int32).astype(np.int64)
        if not np.array_equal(margins, ev.integer):
            fails.append(f"board margins differ from the exact integer simulation on "
                         f"{int((margins != ev.integer).any(axis=1).sum())} rows")
        if not np.array_equal(cls, ev.int_class):
            fails.append("board class differs from the simulation")
        # the reported stack figure is predict()'s frame: the board must not use more than that
        res["stack_reported"] = res["tool_report"]["stack_bytes"]
        if res["stack_reported"] is not None and res["stack_measured"] > res["stack_reported"]:
            fails.append(f"stack measured on the board ({res['stack_measured']} B) exceeds the reported "
                         f"{res['stack_reported']} B")
        if tr.xgb_class_inplace is not None:
            differ |= cls != tr.xgb_class_inplace[sel]
            res["differ_from_xgboost"] = int(differ.sum())
        unflagged = differ & ~ev.at_risk
        res["at_risk"] = int(ev.at_risk.sum())
        res["differ_unflagged"] = int(unflagged.sum())
        if unflagged.any():
            fails.append(f"{int(unflagged.sum())} rows differ from xgboost without having been flagged")
    else:
        margins = raw.view(np.float32)
        res["margins_bit_identical_to_xgboost"] = bool(np.array_equal(margins.view(np.uint32),
                                                                      xgb_margin.astype(np.float32).view(np.uint32)))
        res["max_margin_diff"] = float(np.abs(margins.astype(np.float64) - xgb_margin).max())
    res["ok"] = not fails
    os.makedirs(os.path.join(DEVDIR, "results"), exist_ok=True)
    out = os.path.join(DEVDIR, "results", f"{tag}{opt.replace('-', '_')}{('.' + label) if label else ''}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    t = res["timing"]
    log(f"  {res['rows']} rows checked on the board: {'OK' if res['ok'] else 'FAIL'}; "
        f"differ from xgboost: {res['differ_from_xgboost']}"
        + (f" (flagged in advance: {res['at_risk']}, unflagged: {res['differ_unflagged']})" if kind == "xgbe" else "")
        + f"; inference {t['us_mean']:.1f} us mean, {t['us_max']:.1f} us max; "
        f"model {res['linked_model_bytes']} bytes linked; stack {res['stack_measured']} B")
    for msg in fails:
        log("  FAIL: " + msg)
    return res


def run_packaged_example(board: str = "A", model_name: str = "mc5_d4_r50_hist_missing", opt: str = "-Os") -> Dict:
    """Export the Arduino package, build its example, flash it and read its self-check line from the board."""
    from xgb_edge.package import write_package
    serial = board_serial(board)
    tr = zoo.train(zoo.by_name(model_name))
    exp = export_model(tr.model_path, ExportOptions(name="demo_model", leaf_bits=16), validation=tr.X)
    pkg = os.path.join(DEVDIR, "arduino_example", "demo_model")
    shutil.rmtree(pkg, ignore_errors=True)
    write_package(exp, "arduino", pkg, sample=tr.X)
    sketch = os.path.join(pkg, "examples", "DemoModelPredict")
    bp = os.path.join(DEVDIR, "arduino_example", "build")
    r = subprocess.run(["arduino-cli", "compile", "--fqbn", FQBN, "--library", pkg, "--build-path", bp,
                        "--build-property", f"compiler.optimization_flag={opt}", sketch],
                       capture_output=True, text=True, env=_env())
    if r.returncode != 0:
        raise DeviceError("building the packaged example failed:\n" + (r.stdout + r.stderr)[-3000:])
    want = f"class {int(exp.evaluation.int_class[0])}, margins [" + ", ".join(
        str(int(m)) for m in exp.evaluation.integer[0]) + "]"
    with locked("board_" + serial):
        upload(serial, bp)
        import serial as pyserial
        lines: List[str] = []
        end = time.time() + 20
        port = find_port(serial, wait=30)
        ser = pyserial.Serial(port, 115200, timeout=1)
        try:
            while time.time() < end and len(lines) < 2:
                ln = ser.readline().decode(errors="replace").strip()
                if ln:
                    lines.append(ln)
        finally:
            ser.close()
    ok = bool(lines) and all(want in ln and ln.endswith("self-check OK") for ln in lines)
    res = {"board": serial, "model": model_name, "lines": lines, "expected": want, "ok": ok,
           "code": zoo.code_stamp("device")}
    os.makedirs(os.path.join(DEVDIR, "results"), exist_ok=True)
    with open(os.path.join(DEVDIR, "results", "arduino_example.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    e = sub.add_parser("example", help="run the packaged Arduino example on a board and check its self-check line")
    e.add_argument("--board", default="A")
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--variant", type=int, default=0)
    r.add_argument("--kind", default="xgbe", choices=["xgbe", "float"])
    r.add_argument("--board", default="A")
    r.add_argument("--opt", default="-Os")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--rows", type=int, default=None)
    r.add_argument("--force-upload", action="store_true")
    r.add_argument("--label", default="", help="suffix for the result file (keeps a run from replacing another)")
    s = sub.add_parser("suite")
    s.add_argument("models", nargs="*", help="zoo models (default: the on-device subset)")
    s.add_argument("--board", default="A")
    s.add_argument("--opt", default="-Os")
    s.add_argument("--kind", default="xgbe", choices=["xgbe", "float"])
    args = ap.parse_args(argv)
    if args.cmd == "list":
        ports = list_ports()
        for name, sn in BOARDS.items():
            print(f"board {name}: {sn} -> {ports.get(sn, 'NOT ATTACHED')}")
        return 0
    if args.cmd == "example":
        res = run_packaged_example(args.board)
        for ln in res["lines"]:
            print("board says: " + ln)
        print("packaged Arduino example on the board: " + ("OK" if res["ok"] else f"FAIL (expected '{res['expected']}')"))
        return 0 if res["ok"] else 1
    if args.cmd == "run":
        res = run_model(args.model, args.variant, args.kind, args.board, args.opt, args.reps, args.rows,
                        args.force_upload, label=args.label)
        return 0 if res["ok"] else 1
    bad = 0
    names = args.models or [spec.name for spec in zoo.SPECS if spec.board]
    for name in names:
        try:
            res = run_model(name, 0, args.kind, args.board, args.opt)
            bad += not res["ok"]
        except Exception as e:  # keep going: one model failing must not hide the others
            print(f"[{name}] ERROR: {e}", flush=True)
            bad += 1
    print(f"suite finished: {len(names) - bad} ok, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
