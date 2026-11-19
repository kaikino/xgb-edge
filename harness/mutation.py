"""Mutation check: do the threshold tests notice when the rounding is deliberately broken?

    python -m harness.mutation

Copies the repository to a temporary directory, applies one small, wrong
change at a time to the quantization, the generated C or the error bounds (for
example `<` instead of `<=` in the rank search, or a float-noise bound of
zero) and runs tests/test_thresholds.py and tests/test_guarantee.py against
the changed copy.  Every mutant must make the tests fail; the unchanged copy
must pass.  Exit status is non-zero if a mutant survives.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (what is broken, file, original text, replacement)
MUTANTS: List[Tuple[str, str, str, str]] = [
    ("rank search uses < instead of <=", "xgb_edge/codegen.py",
     "[mid] <= key) lo = mid + 1u", "[mid] < key) lo = mid + 1u"),
    ("threshold code off by one", "xgb_edge/quantize.py",
     "codes[idx] = np.searchsorted(cuts, keys) + 1", "codes[idx] = np.searchsorted(cuts, keys)"),
    ("integer cut uses > instead of >=", "xgb_edge/quantize.py",
     "return bool(np.float32(n) >= t32)", "return bool(np.float32(n) > t32)"),
    ("-0.0 not folded onto +0.0 (rank keys)", "xgb_edge/codegen.py",
     "const int32_t key = (bits >> 31) ? -(int32_t)mag : (int32_t)mag;",
     "const int32_t key = (bits >> 31) ? -(int32_t)mag - 1 : (int32_t)mag;"),
    ("missing value in a default-left slot gets code 1", "xgb_edge/codegen.py",
     'vl = f"(({qtype})0u)" if missing', 'vl = f"(({qtype})1u)" if missing'),
    ("node compare uses <= instead of <", "xgb_edge/codegen.py",
     "left = (uint32_t)q[{dec_slot}] < (uint32_t){dec_code};",
     "left = (uint32_t)q[{dec_slot}] <= (uint32_t){dec_code};"),
    ("integer always-left code dropped (rank)", "xgb_edge/quantize.py",
     "code_of[k] = cuts.size + 1    # every input is below it: always left", "code_of[k] = cuts.size"),
    ("python key of negative floats off by one", "xgb_edge/quantize.py",
     "return np.where((bits >> 31) != 0, -mag, mag)", "return np.where((bits >> 31) != 0, -mag - 1, mag)"),
    ("direct: key of negative floats off by one", "xgb_edge/codegen.py",
     "const uint32_t c = (bits >> 31) ? 0x80000000UL - mag : 0x80000000UL + mag;",
     "const uint32_t c = (bits >> 31) ? 0x80000000UL - mag - 1u : 0x80000000UL + mag;"),
    ("direct: signed integer input not offset to unsigned", "xgb_edge/codegen.py",
     'conv = f"({qtype})(({qtype})features[{feat_expr}] ^ {bias_hex})"',
     'conv = f"({qtype})(({qtype})features[{feat_expr}])"'),
    ("direct: integer threshold off by one", "xgb_edge/quantize.py",
     "codes[idx] = np.where(cut_of > kind.hi, 1, cut_of - kind.lo)[inv]",
     "codes[idx] = np.where(cut_of > kind.hi, 1, np.minimum(cut_of - kind.lo + 1, kind.hi - kind.lo))[inv]"),
    ("direct: always-left split gets code 0", "xgb_edge/quantize.py",
     "codes[idx] = np.where(cut_of > kind.hi, 1, cut_of - kind.lo)[inv]",
     "codes[idx] = np.where(cut_of > kind.hi, 0, cut_of - kind.lo)[inv]"),
    ("direct: float threshold key off by one", "xgb_edge/quantize.py",
     "codes[idx] = f32_key(thrs[idx]) + (1 << 31)", "codes[idx] = f32_key(thrs[idx]) + (1 << 31) + 1"),
    ("direct: constant slot holds 1 instead of 0", "xgb_edge/codegen.py",
     'const_init = f"    q[{tp.const_slot}] = 0;', 'const_init = f"    q[{tp.const_slot}] = 1;'),
    ("missing value in a default-right slot gets code 0", "xgb_edge/codegen.py",
     'vr = f"(({qtype}){tp.missing_right_code}UL)" if missing', 'vr = f"(({qtype})0UL)" if missing'),
    # --- the terms of the guarantee (tests/test_guarantee.py)
    ("no slack for XGBoost's float32 softmax/sigmoid ties", "xgb_edge/analysis.py",
     "DECISION_SLACK = 2.0 ** -21", "DECISION_SLACK = 0.0"),
    ("SAFE_GAP ignores XGBoost's float32 noise", "xgb_edge/analysis.py",
     "need = float(plan.err_hi.max() - plan.err_lo.min()) + 2.0 * float(noise.max()) + DECISION_SLACK",
     "need = float(plan.err_hi.max() - plan.err_lo.min()) + DECISION_SLACK"),
    ("per-row float32 noise bound dropped", "xgb_edge/analysis.py",
     "nz = 1.01 * U32 * (acc_noise + np.abs(total))", "nz = 0.0 * U32 * (acc_noise + np.abs(total))"),
    ("float32 noise bound inflated a thousandfold", "xgb_edge/analysis.py",
     "U32 = 2.0 ** -24 ", "U32 = 2.0 ** -14 "),
    ("leaf-error bound halved", "xgb_edge/quantize.py",
     "hi_terms[t.group].append(float(e.max()))", "hi_terms[t.group].append(float(e.max()) * 0.5)"),
    ("leaf-error bound ignores the bias rounding and doubles the rest", "xgb_edge/quantize.py",
     "lo_terms[t.group].append(float(e.min()))", "lo_terms[t.group].append(float(e.min()) * 2.0)"),
    ("leaves rounded down instead of to nearest", "xgb_edge/quantize.py",
     "qt[leaves] = np.rint(p[leaves] * scale).astype(np.int64)", "qt[leaves] = np.floor(p[leaves] * scale).astype(np.int64)"),
    ("header reports the wrong margin scale", "xgb_edge/codegen.py",
     "#define {P}_MARGIN_SHIFT {lp.shift}", "#define {P}_MARGIN_SHIFT {lp.shift + 1}"),
    ("SAFE_GAP always 1", "xgb_edge/analysis.py",
     "return int(min(g, 2 ** 31 - 1))", "return 1"),
]


def _run_tests(copy: str) -> Tuple[bool, str]:
    env = dict(os.environ, PYTHONPATH=copy, XGBE_NO_SHRINK="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_guarantee.py", "tests/test_thresholds.py",
                        "-x", "-q", "-p", "no:cacheprovider"], cwd=copy, env=env, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED") or " passed" in ln or " failed" in ln]
    return r.returncode == 0, (tail[0] if tail else r.stdout[-200:]).strip()[:110]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        copy = os.path.join(tmp, "repo")
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(
            ".venv", "build", ".sdk", ".git", "__pycache__", ".hypothesis", ".pytest_cache", "*.egg-info"))
        t0 = time.time()
        ok, what = _run_tests(copy)
        print(f"unchanged copy: {'passes' if ok else 'FAILS'}  ({what}; {time.time() - t0:.0f} s)", flush=True)
        if not ok:
            return 2
        survivors = 0
        for label, rel, old, new in MUTANTS:
            path = os.path.join(copy, rel)
            text = open(path).read()
            if text.count(old) != 1:
                print(f"mutant '{label}': the original text was not found exactly once in {rel}")
                return 2
            with open(path, "w") as f:
                f.write(text.replace(old, new))
            t0 = time.time()
            passed, what = _run_tests(copy)
            with open(path, "w") as f:
                f.write(text)
            survivors += passed
            print(f"{'SURVIVED' if passed else 'caught  '}  {label}  ({what}; {time.time() - t0:.0f} s)", flush=True)
        print(f"\n{len(MUTANTS) - survivors} of {len(MUTANTS)} mutants caught by tests/test_thresholds.py "
              "and tests/test_guarantee.py")
        return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
