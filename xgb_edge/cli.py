"""Command line interface: ``xgb-edge export model.json -o out/``."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile

import numpy as np

from . import __version__
from .export import ExportOptions, export_model
from .model import XgbEdgeError
from .report import DEFAULT_CFLAGS, TARGETS, add_flash_totals, find_compiler, measure, render


_MISSING_CELLS = {"", "nan", "na", "n/a", "null", "none", "?"}


def _load_csv(path: str) -> np.ndarray:
    """Read a CSV of numbers strictly: an empty or NaN cell is missing, anything else must be a number.

    A first line whose cells are not numbers is taken as a header and skipped,
    and so is a first line that reads exactly 0,1,2,... (the header pandas
    writes for unnamed columns); text anywhere else is an error rather than a
    silently missing value.
    """
    import csv
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        lineno = 0
        while True:
            try:
                cells = next(reader)
            except StopIteration:
                break
            except csv.Error as e:
                raise XgbEdgeError(f"{path}, line {lineno + 1}: not readable as CSV ({e})") from None
            lineno += 1
            if len(cells) == 0 or (len(cells) == 1 and cells[0].strip() == ""):
                continue                                  # blank line
            vals, bad = [], None
            for col, cell in enumerate(cells, start=1):
                c = cell.strip()
                if c.lower() in _MISSING_CELLS:
                    vals.append(np.nan)
                    continue
                try:
                    vals.append(float(c))
                except ValueError:
                    bad = (col, c)
                    break
            if bad is not None:
                if not rows and lineno == 1:
                    continue                              # header line
                raise XgbEdgeError(f"{path}, line {lineno}, column {bad[0]}: {bad[1]!r} is not a number")
            rows.append(vals)
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    # pandas' to_csv writes the default column labels 0,1,2,... as a header line
    if len(rows) > 1 and len(rows[0]) > 1 and rows[0] == [float(i) for i in range(len(rows[0]))]:
        rows = rows[1:]
    width = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != width:
            raise XgbEdgeError(f"{path}: data row {i + 1} has {len(r)} columns, the first data row has {width}")
    return np.asarray(rows, dtype=np.float64)


def load_matrix(path: str) -> np.ndarray:
    """Load a feature matrix from .npy, .npz (first array) or .csv (empty cell = missing)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".npy", ".npz", ".csv", ".txt"):
        raise XgbEdgeError(f"unsupported data file {path!r}: use .npy, .npz or .csv")
    try:
        if ext == ".npy":
            data = np.load(path, allow_pickle=False)
        elif ext == ".npz":
            with np.load(path, allow_pickle=False) as z:
                if not z.files:
                    raise XgbEdgeError(f"{path} holds no arrays")
                data = z[z.files[0]]
        else:
            data = _load_csv(path)
    except XgbEdgeError:
        raise
    except (OSError, ValueError, IndexError, EOFError, zipfile.BadZipFile) as e:
        raise XgbEdgeError(f"cannot read {path}: {e}") from None
    if not (np.issubdtype(data.dtype, np.floating) or np.issubdtype(data.dtype, np.integer)
            or data.dtype == np.bool_):
        raise XgbEdgeError(f"{path} does not hold numbers (array type {data.dtype})")
    return data


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
            s.add_argument("--package", choices=["arduino", "zephyr", "esp-idf"],
                           help="write an Arduino library, Zephyr module or ESP-IDF component with an example "
                                "into the output directory, instead of the bare .h/.c")
        s.add_argument("--name", default="xgb_model", help="C identifier prefix and file name (default: xgb_model)")
        s.add_argument("--validation", metavar="FILE",
                       help="feature matrix (.npy/.npz/.csv, NaN or empty = missing) used to check tie risk")
        s.add_argument("--leaf-bits", default="auto",
                       help="fixed-point leaf width: auto (default), or 2..32")
        s.add_argument("--input", default="f32", choices=["f32", "i8", "u8", "i16", "u16", "i32"],
                       help="type of the feature array passed to predict (default: f32)")
        s.add_argument("--leaf-store", default="auto", choices=["auto", "inline", "table"],
                       help="store leaf values in the nodes or in a shared table (default: smaller of the two)")
        s.add_argument("--thresholds", default="auto", choices=["auto", "rank", "direct"],
                       help="threshold encoding: ranks into per-feature cut tables, or keys stored in the "
                            "nodes (default: smaller of the two)")
        s.add_argument("--rounds", default="auto",
                       help="boosting rounds to export: all, best (stop at best_iteration, as "
                            "XGBClassifier.predict does) or a number; required for early-stopped models")
        s.add_argument("--target", default="cortex-m4", choices=sorted(TARGETS),
                       help="target whose compiler measures code size and stack for the report, if that "
                            "compiler is installed (default: cortex-m4)")
        s.add_argument("--cc", help="measure with this cross compiler instead of the --target one")
        s.add_argument("--cflags", metavar="FLAGS",
                       help="flags for the measuring compiler, written as --cflags=\"-mcpu=cortex-m0 -Os\" "
                            "(with the equals sign, because the value starts with a dash); default: the "
                            "--target flags")
        s.add_argument("--no-measure", action="store_true", help="do not run a compiler; estimate code size")
        s.add_argument("--report-json", metavar="FILE", help="also write the report as JSON")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        opt = ExportOptions(name=args.name, input=args.input, leaf_bits=args.leaf_bits,
                            leaf_store=args.leaf_store, thresholds=args.thresholds, rounds=args.rounds)
        if opt.leaf_bits != "auto":
            try:
                opt.leaf_bits = int(opt.leaf_bits)
            except ValueError:
                raise XgbEdgeError(f"--leaf-bits must be 'auto' or a whole number, got {args.leaf_bits!r}") from None
        X = load_matrix(args.validation) if args.validation else None
        # refuse unusable output locations before doing (and writing) anything
        if args.report_json:
            parent = os.path.dirname(os.path.abspath(args.report_json))
            if not os.path.isdir(parent):
                raise XgbEdgeError(f"--report-json: the directory {parent} does not exist")
            if os.path.isdir(args.report_json):
                raise XgbEdgeError(f"--report-json {args.report_json} is a directory, not a file name")
        out_dir = getattr(args, "out", None)
        if out_dir is not None and os.path.exists(out_dir) and not os.path.isdir(out_dir):
            raise XgbEdgeError(f"--out {out_dir} exists and is not a directory")
        exp = export_model(args.model, opt, X)
        measured = None
        if not args.no_measure:
            exe, flags = TARGETS[args.target]
            cc = args.cc or find_compiler(exe)
            if cc:
                try:
                    measured = measure(exp.generated.header, exp.generated.source, exp.generated.name,
                                       cc, args.cflags if args.cflags is not None else flags)
                    measured["target"] = args.target
                except (OSError, RuntimeError, subprocess.SubprocessError) as e:
                    print(f"warning: could not measure code size with {cc}; the report uses an estimate.\n"
                          f"  {str(e).strip()[:600]}", file=sys.stderr)
        add_flash_totals(exp.report, measured)
        package = getattr(args, "package", None)
        if package:
            from .package import write_package
            paths = write_package(exp, package, args.out, sample=X)   # the example runs on the first row of X
        elif args.cmd == "export":
            paths = exp.write(args.out)
        sys.stdout.write(render(exp.report))
        if package:
            print(f"wrote {package} package to {args.out}:\n  "
                  + "\n  ".join(os.path.relpath(p, args.out) for p in paths))
        elif args.cmd == "export":
            print("wrote " + ", ".join(paths))
        if args.report_json:
            with open(args.report_json, "w") as f:
                json.dump(exp.report, f, indent=2)
    except XgbEdgeError as e:
        print(f"xgb-edge: error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        # an output path that is a file, a missing or read-only directory, ...
        where = f" ({e.filename})" if getattr(e, "filename", None) else ""
        print(f"xgb-edge: error: cannot write the output{where}: {e.strerror or e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
