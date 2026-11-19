"""Compile generated C for the host and push test rows through it (used by tests and bench.altgen).

Every driver reads rows of float32 features from stdin and writes one fixed-size
record per row to stdout, so a whole test set is evaluated by one process.
"""
from __future__ import annotations

import os
import subprocess
from typing import List, Optional, Sequence

import numpy as np

#: Driver for anything with the xgb-edge style API ``int NAME_predict(const float *, float *margins)``.
#: Per row it writes NUM_MARGINS + 2 floats: class, class from predict(x, NULL), then the margins.
PREDICT_DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include "@NAME@.h"
#define NF @UNAME@_NUM_FEATURES
#define NM @UNAME@_NUM_MARGINS
int main(int argc, char **argv)
{
    long rows = argc > 1 ? atol(argv[1]) : 0, r;
    static float x[NF + 1];
    static float out[NM + 2];
    for (r = 0; r < rows; r++) {
        if (NF > 0 && fread(x, sizeof x[0], NF, stdin) != (size_t)NF) return 2;
        out[1] = (float)@NAME@_predict(x, NULL);
        out[0] = (float)@NAME@_predict(x, out + 2);
        if (fwrite(out, sizeof out[0], NM + 2, stdout) != (size_t)(NM + 2)) return 3;
    }
    return 0;
}
"""


def compile_host(sources: Sequence[str], exe: str, cflags: Sequence[str], include_dirs: Sequence[str] = (),
                 cc: str = "cc", timeout: Optional[float] = None) -> str:
    """Compile and link ``sources`` into ``exe``.  Returns the compiler's stderr (warnings).

    Raises RuntimeError with the compiler output if the build fails.
    """
    cmd = [cc] + list(cflags)
    for d in include_dirs:
        cmd += ["-I", d]
    cmd += list(sources) + ["-o", exe, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("host compile failed:\n" + " ".join(cmd) + "\n" + r.stderr[-4000:])
    return r.stderr


def run_rows(exe: str, X: np.ndarray, out_dtype, out_cols: int, timeout: Optional[float] = None) -> np.ndarray:
    """Feed float32 rows to a driver executable; returns its output as an (n, out_cols) array."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    r = subprocess.run([exe, str(n)], input=X.tobytes(), capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{exe} failed (exit {r.returncode}): {r.stderr.decode(errors='replace')[-2000:]}")
    return np.frombuffer(r.stdout, dtype=out_dtype).reshape(n, out_cols)


def build_predict(workdir: str, name: str, header: str, source: str, cflags: Sequence[str], cc: str = "cc"):
    """Write ``name.h``/``name.c`` plus the driver into ``workdir`` and build them.

    Returns (exe path, compiler stderr).
    """
    os.makedirs(workdir, exist_ok=True)
    paths: List[str] = []
    for fn, text in ((name + ".h", header), (name + ".c", source),
                     ("driver.c", PREDICT_DRIVER.replace("@NAME@", name).replace("@UNAME@", name.upper()))):
        p = os.path.join(workdir, fn)
        with open(p, "w") as f:
            f.write(text)
        paths.append(p)
    exe = os.path.join(workdir, "run")
    err = compile_host(paths[1:], exe, cflags, include_dirs=[workdir], cc=cc)
    return exe, err


def run_predict(exe: str, X: np.ndarray, n_margins: int):
    """Returns (class int64[n], class_with_null_margins int64[n], margins float32[n, n_margins])."""
    out = run_rows(exe, X, np.float32, n_margins + 2)
    return out[:, 0].astype(np.int64), out[:, 1].astype(np.int64), np.ascontiguousarray(out[:, 2:])
