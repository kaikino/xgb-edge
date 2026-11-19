"""Find the largest firmware image the XIAO nRF52840 bootloader accepts over serial DFU.

    python -m harness.flash_limit [--board A] SIZE [SIZE ...]

Builds the test firmware with a tiny model plus ballast so that the program is
exactly SIZE bytes, uploads it and reports whether the board then runs it.
The Arduino core advertises 811008 bytes; the bootloader may accept less.
"""
import argparse
import sys

from harness import device as D
from harness import zoo
from xgb_edge import ExportOptions, export_model


def build_padded(target: int):
    tr = zoo.train(zoo.by_name("bin_d2_r5_hist"))
    exp = export_model(tr.model_path, ExportOptions(name=D.MODEL_NAME, leaf_bits=16))
    files = {D.MODEL_NAME + ".h": exp.generated.header, D.MODEL_NAME + ".c": exp.generated.source}
    body = (f'#include "mut.h"\n#include "{D.MODEL_NAME}.h"\nint mut_predict(const void *f, void *m)\n'
            f"{{\n    return {D.MODEL_NAME}_predict((const {D.MODEL_NAME}_feature_t *)f, (int32_t *)m);\n}}\n")
    pad = 4
    for _ in range(4):
        sketch, bid = D.make_sketch("flash_limit", files, exp.model.n_features, 1, 4, False, body, pad_bytes=pad)
        build = D.compile_sketch(sketch)
        if build["program_bytes"] == target:
            break
        pad += target - build["program_bytes"]
    return sketch, build, bid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sizes", nargs="+", type=int)
    ap.add_argument("--board", default="A")
    args = ap.parse_args(argv)
    serial = D.board_serial(args.board)
    results = []
    for size in args.sizes:
        sketch, build, bid = build_padded(size)
        try:
            with D.locked("board_" + serial):
                link, _ = D.flash_and_open(serial, sketch, build, bid, force=True)
                link.close()
            accepted = True
            print(f"{build['program_bytes']} bytes: accepted, board runs it", flush=True)
        except D.DeviceError as e:
            accepted = False
            print(f"{build['program_bytes']} bytes: REJECTED ({str(e).splitlines()[0][:120]})", flush=True)
        results.append({"program_bytes": build["program_bytes"], "accepted": accepted})
    # keep what was measured, so that the limit quoted in the README has an artifact behind it
    import json
    import os
    path = os.path.join(D.DEVDIR, "results", "flash_limit.json")
    old = []
    if os.path.exists(path):
        old = [r for r in json.load(open(path))["probes"]
               if r["program_bytes"] not in {x["program_bytes"] for x in results}]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    probes = sorted(old + results, key=lambda r: r["program_bytes"])
    with open(path, "w") as f:
        json.dump({"board": serial, "limit_used_by_the_harness": D.FLASH_MAX,
                   "core_advertised_maximum": D.FLASH_MAX_CORE, "probes": probes}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
