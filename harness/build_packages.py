"""Build every platform package for real and record what happened.

Run with ``python -m harness.build_packages``.  It exports one small zoo model
as an Arduino library, a Zephyr module and an ESP-IDF component (see
``xgb_edge.package``) and builds the example that ships in each package with
the platform's own build system:

  arduino   arduino-cli compile, for every entry of ARDUINO_BUILDS
  esp-idf   idf.py build of examples/predict for target esp32
  zephyr    west build of samples/predict for every board in ZEPHYR_BOARDS

Everything is compile-only: nothing is uploaded and no serial port is opened.

ESP-IDF and Zephyr are installed on first use under ``.sdk/`` (shallow clones,
only the toolchains, modules and submodules these builds need; about 5 GB on
disk, slow the first time).  Pass ``--no-install`` to only use what is already
there.  A platform whose SDK cannot be installed, or whose build fails, is
recorded as such with the failing command and the end of its output; nothing
is made up.

Each build is made with -Wall -Wextra in effect for the generated files, and
its output is searched for warnings that mention them.  ``--inject-warning``
checks that search: it plants four warnings in the packaged files and expects
every build to report them.

The result is ``build/packages/results.json``; full command output is kept in
``build/packages/logs/``.  Exit status is non-zero unless every requested
platform built without a warning in the generated files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import zoo  # noqa: E402

ROOT = zoo.ROOT
SDK = os.path.join(ROOT, ".sdk")
OUT = os.path.join(ROOT, "build", "packages")
LOGS = os.path.join(OUT, "logs")

ZOO_MODEL = "mc5_d4_r50_hist_missing"
MODEL_NAME = "demo_model"

# --- ESP-IDF
ESP_IDF_TAG = "v6.1.1"
ESP_IDF_URL = "https://github.com/espressif/esp-idf.git"
ESP_IDF_DIR = os.path.join(SDK, "esp-idf")
ESP_IDF_TOOLS = os.path.join(SDK, "espressif")
ESP_IDF_TARGET = "esp32"
# To keep the download small the clone is shallow, blobless and sparse (no docs/, no examples/),
# and only the submodules a minimal esp32 application links are fetched.  ESP-IDF would
# otherwise clone every missing submodule in full during configuration, so the build sets
# IDF_SKIP_CHECK_SUBMODULES=1 (see esp-idf/tools/cmake/git_submodules.cmake).
# A project that does not set MINIMAL_BUILD needs more submodules (lwip, wifi, bt, ...) than these.
ESP_IDF_SPARSE = ["components", "tools"]
ESP_IDF_SUBMODULES = [
    "components/heap/tlsf",
    "components/mbedtls/mbedtls",
    "components/bootloader/subproject/components/micro-ecc/micro-ecc",
]
# ESP-IDF compiles everything with -Wall -Wextra -Werror, but switches these two -Wextra warnings
# off.  Its EXTRA_CFLAGS environment variable appends options after its own; ESP-IDF's own sources
# are not clean under the two warnings, so they must not become errors.
ESP_IDF_EXTRA_CFLAGS = "-Wunused-parameter -Wsign-compare -Wno-error=unused-parameter -Wno-error=sign-compare"

# --- Zephyr
ZEPHYR_TAG = "v4.5.0"
ZEPHYR_URL = "https://github.com/zephyrproject-rtos/zephyr"
ZEPHYR_WS = os.path.join(SDK, "zephyrproject")
ZEPHYR_VENV = os.path.join(SDK, "zephyr-venv")
ZEPHYR_MODULES = ["cmsis", "cmsis_6", "hal_nordic"]     # all an nRF52840 build needs
ZEPHYR_BOARDS = ["xiao_ble/nrf52840/sense", "nrf52840dk/nrf52840"]
ZEPHYR_SDK_VERSION = "1.0.1"                            # zephyr/SDK_VERSION of ZEPHYR_TAG
ZEPHYR_SDK_DIR = os.path.join(SDK, f"zephyr-sdk-{ZEPHYR_SDK_VERSION}")
ZEPHYR_SDK_BASE = f"https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v{ZEPHYR_SDK_VERSION}"
ZEPHYR_SDK_TOOLCHAIN = "arm-zephyr-eabi"
# sha256 of the release archives, from the release's sha256.sum
ZEPHYR_SDK_SHA256 = {
    "zephyr-sdk-1.0.1_macos-aarch64_minimal.tar.xz":
        "867063901f39528a6175a80ebc20367bd6cb440593e7e2650eda30392f1f6b65",
    "toolchain_gnu_macos-aarch64_arm-zephyr-eabi.tar.xz":
        "4008edb5d4840cd994aedd7f1309bfb63e7243729d57839ebf1cc83c1f17c886",
}


class StepFailed(Exception):
    """A setup or build command failed; carries what is needed to report it."""

    def __init__(self, what: str, cmd: str, rc, tail: str):
        super().__init__(f"{what}: exit {rc}")
        self.what, self.cmd, self.rc, self.tail = what, cmd, rc, tail

    def as_dict(self) -> Dict:
        return {"step": self.what, "command": self.cmd, "exit_status": self.rc, "output_tail": self.tail}


def _say(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: List[str], log: str, what: str, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None, check: bool = True, attempts: int = 1) -> Tuple[int, str]:
    """Run ``cmd``, append its combined output to the log file ``log`` and return (status, output).

    ``attempts`` > 1 retries a failing command (for downloads over a flaky connection).
    """
    for attempt in range(1, attempts + 1):
        try:
            return _run_once(cmd, log, what, cwd, env, timeout, check)
        except StepFailed:
            if attempt == attempts:
                raise
            _say(f"      failed (attempt {attempt} of {attempts}), retrying in 10 s")
            time.sleep(10)
    raise AssertionError("unreachable")


def _run_once(cmd: List[str], log: str, what: str, cwd: Optional[str], env: Optional[Dict[str, str]],
              timeout: Optional[float], check: bool) -> Tuple[int, str]:
    os.makedirs(os.path.dirname(log), exist_ok=True)
    shown = " ".join(shlex.quote(c) for c in cmd)
    _say(f"    $ {shown}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=timeout)
        rc, out = r.returncode, r.stdout.decode(errors="replace")
    except FileNotFoundError as e:
        rc, out = 127, str(e)
    except subprocess.TimeoutExpired as e:
        rc, out = "timeout", (e.stdout or b"").decode(errors="replace") + f"\n[timed out after {timeout:.0f}s]"
    with open(log, "a") as f:
        f.write(f"\n$ {shown}\n[cwd: {cwd or os.getcwd()}]\n{out}\n[exit {rc}, {time.time() - t0:.1f}s]\n")
    if check and rc != 0:
        raise StepFailed(what, shown, rc, out[-3000:])
    return rc, out


def net_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """``env`` (default: this process's) with git told to give up on a stalled transfer, so a retry can happen."""
    env = dict(os.environ if env is None else env)
    env.setdefault("GIT_HTTP_LOW_SPEED_LIMIT", "1000")   # bytes/s ...
    env.setdefault("GIT_HTTP_LOW_SPEED_TIME", "60")      # ... for this many seconds
    return env


def _host() -> str:
    """Host name as used in Zephyr SDK archive names."""
    system = {"Darwin": "macos", "Linux": "linux"}.get(platform.system())
    machine = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "AMD64": "x86_64"}.get(platform.machine())
    if not system or not machine:
        raise StepFailed("zephyr sdk", "(host detection)", 1,
                         f"no Zephyr SDK archive known for {platform.system()} {platform.machine()}")
    return f"{system}-{machine}"


def _download(url: str, dest: str, log: str, what: str) -> None:
    """Download ``url`` to ``dest`` (resumable), checking the sha256 when it is pinned above."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    want = ZEPHYR_SDK_SHA256.get(os.path.basename(dest))

    def digest() -> str:
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.exists(dest) and want and digest() == want:
        return
    run(["curl", "-fL", "--retry", "3", "-C", "-", "-sS", "-o", dest, url], log, what, timeout=3600, attempts=3)
    if want and digest() != want:
        os.remove(dest)
        raise StepFailed(what, f"sha256 check of {os.path.basename(dest)}", 1,
                         f"downloaded file does not have the expected sha256 {want}")


# --------------------------------------------------------------------------- ESP-IDF

def esp_idf_env() -> Dict[str, str]:
    """Environment for install.sh / export.sh: tools under .sdk/, and one fixed Python.

    ESP-IDF names its Python environment after the interpreter that created it,
    so the same ``python3`` has to come first on PATH for install and for build.
    """
    env = dict(os.environ)
    env["IDF_TOOLS_PATH"] = ESP_IDF_TOOLS
    env["PATH"] = os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + env.get("PATH", "")
    env["IDF_SKIP_CHECK_SUBMODULES"] = "1"      # only ESP_IDF_SUBMODULES are present, on purpose
    for k in ("IDF_PATH", "IDF_PYTHON_ENV_PATH", "ESP_IDF_VERSION", "IDF_TARGET"):
        env.pop(k, None)
    return env


def setup_esp_idf(log: str) -> None:
    """Shallow-clone ESP-IDF at ESP_IDF_TAG and install the tools for ESP_IDF_TARGET, once."""
    cloned = os.path.join(ESP_IDF_DIR, f".xgb_edge_cloned_{ESP_IDF_TAG}")
    if not os.path.exists(cloned):
        if not os.path.isdir(os.path.join(ESP_IDF_DIR, ".git")):
            run(["git", "clone", "--depth", "1", "--branch", ESP_IDF_TAG, "--filter=blob:none", "--sparse",
                 ESP_IDF_URL, ESP_IDF_DIR], log, "clone esp-idf", env=net_env(), timeout=3600, attempts=4)
        run(["git", "-C", ESP_IDF_DIR, "sparse-checkout", "set"] + ESP_IDF_SPARSE,
            log, "check out esp-idf components and tools", env=net_env(), timeout=3600, attempts=4)
        open(cloned, "w").close()
    subs = os.path.join(ESP_IDF_DIR, ".xgb_edge_submodules_" + hashlib.sha1(
        "\n".join(ESP_IDF_SUBMODULES).encode()).hexdigest()[:12])
    if not os.path.exists(subs):
        run(["git", "-C", ESP_IDF_DIR, "submodule", "update", "--init", "--recursive", "--depth", "1", "--"]
            + ESP_IDF_SUBMODULES, log, "fetch esp-idf submodules", env=net_env(), timeout=3600, attempts=4)
        open(subs, "w").close()
    installed = os.path.join(ESP_IDF_TOOLS, f".xgb_edge_installed_{ESP_IDF_TAG}_{ESP_IDF_TARGET}")
    if not os.path.exists(installed):
        os.makedirs(ESP_IDF_TOOLS, exist_ok=True)
        run([os.path.join(ESP_IDF_DIR, "install.sh"), ESP_IDF_TARGET], log, "esp-idf install.sh",
            cwd=ESP_IDF_DIR, env=net_env(esp_idf_env()), timeout=3 * 3600, attempts=3)
        open(installed, "w").close()


# --------------------------------------------------------------------------- Zephyr

def zephyr_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = os.path.join(ZEPHYR_VENV, "bin") + os.pathsep + env.get("PATH", "")
    env["ZEPHYR_BASE"] = os.path.join(ZEPHYR_WS, "zephyr")
    env["ZEPHYR_SDK_INSTALL_DIR"] = ZEPHYR_SDK_DIR
    env["ZEPHYR_TOOLCHAIN_VARIANT"] = "zephyr"
    env.pop("VIRTUAL_ENV", None)
    return env


def _pip_install(python: str, args: List[str], log: str, what: str) -> None:
    if shutil.which("uv"):
        run(["uv", "pip", "install", "--python", python] + args, log, what, timeout=1800, attempts=3)
    else:
        run([python, "-m", "pip", "install"] + args, log, what, timeout=1800, attempts=3)


def setup_zephyr(log: str) -> None:
    """A west workspace with Zephyr and only ZEPHYR_MODULES, plus the ARM toolchain of the Zephyr SDK."""
    vpy = os.path.join(ZEPHYR_VENV, "bin", "python")
    west = os.path.join(ZEPHYR_VENV, "bin", "west")
    if not os.path.exists(vpy):
        if shutil.which("uv"):
            run(["uv", "venv", "--python", sys.executable, ZEPHYR_VENV], log, "create zephyr venv")
        else:
            run([sys.executable, "-m", "venv", ZEPHYR_VENV], log, "create zephyr venv")
    if not os.path.exists(west):
        _pip_install(vpy, ["west"], log, "install west")
    env = net_env(zephyr_env())
    if not os.path.exists(os.path.join(ZEPHYR_WS, ".west", "config")):
        # west refuses to initialise next to a leftover .west directory, or when ZEPHYR_BASE points into it
        init_env = {k: v for k, v in env.items() if k != "ZEPHYR_BASE"}
        for attempt in range(1, 5):
            shutil.rmtree(os.path.join(ZEPHYR_WS, ".west"), ignore_errors=True)
            try:
                run([west, "init", "-m", ZEPHYR_URL, "--mr", ZEPHYR_TAG, "-o=--depth=1", ZEPHYR_WS],
                    log, "west init", env=init_env, timeout=3600)
                break
            except StepFailed:
                if attempt == 4:
                    raise
                _say(f"      failed (attempt {attempt} of 4), retrying in 10 s")
                time.sleep(10)
    updated = os.path.join(ZEPHYR_WS, f".xgb_edge_updated_{ZEPHYR_TAG}_" + "_".join(ZEPHYR_MODULES))
    if not os.path.exists(updated):
        run([west, "update", "--narrow", "-o=--depth=1"] + ZEPHYR_MODULES, log, "west update",
            cwd=ZEPHYR_WS, env=env, timeout=3600, attempts=4)
        open(updated, "w").close()
    reqs = os.path.join(ZEPHYR_VENV, f".xgb_edge_requirements_{ZEPHYR_TAG}")
    if not os.path.exists(reqs):
        _pip_install(vpy, ["-r", os.path.join(ZEPHYR_WS, "zephyr", "scripts", "requirements-base.txt")],
                     log, "install zephyr python requirements")
        open(reqs, "w").close()
    host = _host()
    if not os.path.exists(os.path.join(ZEPHYR_SDK_DIR, "sdk_version")):
        name = f"zephyr-sdk-{ZEPHYR_SDK_VERSION}_{host}_minimal.tar.xz"
        arc = os.path.join(SDK, "downloads", name)
        _download(f"{ZEPHYR_SDK_BASE}/{name}", arc, log, "download zephyr sdk (minimal bundle)")
        run(["tar", "-xJf", arc, "-C", SDK], log, "extract zephyr sdk")
    tc_dir = os.path.join(ZEPHYR_SDK_DIR, "gnu", ZEPHYR_SDK_TOOLCHAIN)
    if not os.path.exists(os.path.join(tc_dir, "bin", f"{ZEPHYR_SDK_TOOLCHAIN}-gcc")):
        # what the SDK's "setup.sh -t arm-zephyr-eabi" does, minus its wget and CMake registry steps
        name = f"toolchain_gnu_{host}_{ZEPHYR_SDK_TOOLCHAIN}.tar.xz"
        arc = os.path.join(SDK, "downloads", name)
        _download(f"{ZEPHYR_SDK_BASE}/{name}", arc, log, "download zephyr sdk arm toolchain")
        os.makedirs(os.path.dirname(tc_dir), exist_ok=True)
        run(["tar", "-xJf", arc, "-C", os.path.dirname(tc_dir)], log, "extract zephyr sdk arm toolchain")


SETUP = {"esp-idf": setup_esp_idf, "zephyr": setup_zephyr}


def sdk_missing(kind: str) -> Optional[str]:
    """Why the SDK of ``kind`` cannot be used as it is on disk (None if it looks complete)."""
    if kind == "esp-idf":
        need = [os.path.join(ESP_IDF_DIR, "export.sh"),
                os.path.join(ESP_IDF_TOOLS, f".xgb_edge_installed_{ESP_IDF_TAG}_{ESP_IDF_TARGET}")]
    elif kind == "zephyr":
        need = [os.path.join(ZEPHYR_VENV, "bin", "west"), os.path.join(ZEPHYR_WS, "zephyr", "VERSION"),
                os.path.join(ZEPHYR_SDK_DIR, "gnu", ZEPHYR_SDK_TOOLCHAIN, "bin", f"{ZEPHYR_SDK_TOOLCHAIN}-gcc")]
    else:
        return None
    gone = [os.path.relpath(n, ROOT) for n in need if not os.path.exists(n)]
    return ("missing " + ", ".join(gone)) if gone else None


# --------------------------------------------------------------------------- shared helpers

def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def first_line(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """First output line of a ``--version`` style command, or None if it cannot be run."""
    try:
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [ln.strip() for ln in r.stdout.decode(errors="replace").splitlines() if ln.strip()]
    return lines[0] if r.returncode == 0 and lines else None


def fresh_dir(path: str) -> str:
    """Empty ``path``, which must be inside build/packages (nothing else is ever deleted)."""
    path = os.path.abspath(path)
    if os.path.commonpath([path, OUT]) != OUT or path == OUT:
        raise ValueError(f"refusing to clear {path}: not inside {OUT}")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


WARNING = re.compile(r"\bwarning\b", re.IGNORECASE)


def scan_warnings(output: str, marks: List[str]) -> Dict:
    """Count compiler/CMake/Kconfig warning lines, and collect those that mention the generated files.

    ``marks`` are strings that identify the package: its directory, the model
    name, the example's name, the Kconfig symbol.  Compile command lines echoed
    by verbose builds are not diagnostics and are skipped.
    """
    total, ours = 0, []
    for line in output.splitlines():
        if not WARNING.search(line) or " -o " in line or " -c " in line:
            continue
        total += 1
        low = line.lower()
        if any(m.lower() in low for m in marks):
            ours.append(line.strip()[:400])
    return {"warnings_total": total, "warnings_in_generated_files": ours}


def compile_flags(build_dir: str, source_suffix: str) -> Optional[Dict]:
    """Compiler, -O and -W flags actually used for the file ending in ``source_suffix`` (from compile_commands.json)."""
    path = os.path.join(build_dir, "compile_commands.json")
    try:
        entries = json.load(open(path))
    except (OSError, ValueError):
        return None
    for e in entries:
        if not e.get("file", "").endswith(source_suffix):
            continue
        args = e.get("arguments") or shlex.split(e.get("command", ""))
        # ESP toolchains keep most flags in @response files
        flat: List[str] = []
        for a in args:
            if a.startswith("@") and os.path.isfile(a[1:]):
                flat.extend(shlex.split(open(a[1:]).read()))
            else:
                flat.append(a)
        cc = args[0]
        return {
            "compiler": cc,
            "compiler_version": first_line([cc, "--version"]),
            "optimization": [a for a in flat if re.fullmatch(r"-O\w*", a)],
            "warning_flags": [a for a in flat if a.startswith("-W") and not a.startswith("-Wl,")
                              and not a.startswith("-Wa,") and not a.startswith("-Wp,")],
        }
    return None


def object_sizes(size_tool: str, obj: str) -> Optional[Dict]:
    """Code and read-only data bytes of one object file or archive (``size -A``)."""
    try:
        r = subprocess.run([size_tool, "-A", obj], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    text = rodata = ram = 0
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            if parts[0].startswith(".text") or parts[0].startswith(".literal"):
                text += int(parts[1])
            elif parts[0].startswith((".rodata", ".srodata")):
                rodata += int(parts[1])
            elif parts[0].startswith((".data", ".bss", ".sdata", ".sbss")):
                ram += int(parts[1])
    return {"code_bytes": text, "rodata_bytes": rodata, "static_ram_bytes": ram}


def size_tool_for(compiler: str) -> str:
    return re.sub(r"(gcc|g\+\+|cc|c\+\+)(\.exe)?$", "size", compiler)


INJECT_MODEL = """
/* injected by harness.build_packages --inject-warning: three warnings that only -Wextra gives
 * (-Wtype-limits, -Wsign-compare, -Wunused-parameter) */
int xgb_edge_injected_wextra(unsigned v, int xgb_edge_injected_unused);
int xgb_edge_injected_wextra(unsigned v, int xgb_edge_injected_unused)
{
    int i = -1;
    return (v < 0) + (i < v);
}
"""
INJECT_EXAMPLE = "int xgb_edge_injected_wall; /* injected: -Wall (-Wunused-variable) reports this */ "


def inject_warnings(pkg: str, name: str, kind: str) -> None:
    """Self-test of the warning check: put one -Wall and one -Wextra warning into the package."""
    from xgb_edge.package import camel_case
    source = {"arduino": f"src/{name}.c", "zephyr": f"src/{name}.c", "esp-idf": f"{name}.c"}[kind]
    example = {"arduino": f"examples/{camel_case(name)}Predict/{camel_case(name)}Predict.ino",
               "zephyr": "samples/predict/src/main.c", "esp-idf": "examples/predict/main/predict_main.c"}[kind]
    with open(os.path.join(pkg, source), "a") as f:
        f.write(INJECT_MODEL)
    path = os.path.join(pkg, example)
    text = open(path).read()
    anchor = f"int32_t margins[{name.upper()}_NUM_MARGINS];"
    assert text.count(anchor) == 1
    open(path, "w").write(text.replace(anchor, INJECT_EXAMPLE + anchor))


def injected_seen(warnings: List[str]) -> Dict[str, bool]:
    """Which of the injected warnings a build reported (all four must be, for the check to be trusted)."""
    return {"example -Wunused-variable": any("xgb_edge_injected_wall" in w for w in warnings),
            "model -Wtype-limits": any("=type-limits]" in w or "-Wtype-limits]" in w for w in warnings),
            "model -Wsign-compare": any("-Wsign-compare]" in w for w in warnings),
            "model -Wunused-parameter": any("-Wunused-parameter]" in w for w in warnings)}


# --------------------------------------------------------------------------- Arduino

# (FQBN, extra build properties, label).  The Seeed core optimises with -Ofast by default,
# so that board is also built with -Os to show the size-optimised result.
ARDUINO_BUILDS: List[Tuple[str, Dict[str, str], str]] = [
    ("Seeeduino:nrf52:xiaonRF52840SensePlus", {}, "core default"),
    ("Seeeduino:nrf52:xiaonRF52840SensePlus", {"compiler.optimization_flag": "-Os"}, "-Os"),
    ("esp32:esp32:esp32", {}, "core default"),
    ("esp32:esp32:esp32c3", {}, "core default"),
]
# "--warnings all" means "-Wall -Wextra" in both cores, but each then switches some -Wextra
# warnings off again (Seeed: unused-parameter, missing-field-initializers, pointer-arith; ESP32:
# unused-parameter, sign-compare and others, in its SDK flag file).  A specific -W option beats
# -Wextra wherever it stands, so the ones that matter are named explicitly, and they are put in
# compiler.warning_flags.all because that is expanded after the cores' own flags in both recipes.
ARDUINO_WARNINGS = "-Wall -Wextra -Wunused-parameter -Wsign-compare -Wmissing-field-initializers -Wpointer-arith"


def arduino_env() -> Dict[str, str]:
    """The Seeed nRF52 core runs plain ``python`` after linking: give it the interpreter running this."""
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + env.get("PATH", "")
    return env


def build_arduino(pkg: str, name: str, work: str, marks: List[str], install: bool, inject: bool = False) -> Dict:
    from xgb_edge.package import camel_case
    cli = shutil.which("arduino-cli")
    res: Dict = {"package": rel(pkg), "builds": []}
    if not cli:
        res.update(status="not built", reason="arduino-cli was not found on PATH")
        return res
    env = arduino_env()
    tools: Dict = {"arduino-cli": None, "cores": {}}
    try:
        tools["arduino-cli"] = json.loads(subprocess.run(
            [cli, "version", "--json"], capture_output=True, text=True, env=env).stdout)["VersionString"]
        plats = json.loads(subprocess.run([cli, "core", "list", "--json"], capture_output=True, text=True,
                                          env=env).stdout).get("platforms", [])
        tools["cores"] = {p["id"]: p.get("installed_version") for p in plats}
    except (OSError, ValueError, KeyError) as e:
        tools["error"] = f"could not query arduino-cli: {e}"
    res["tools"] = tools
    sketch_name = camel_case(name) + "Predict"
    sketch = os.path.join(pkg, "examples", sketch_name)
    for fqbn, props, label in ARDUINO_BUILDS:
        tag = fqbn.replace(":", "_") + ("" if not props else "_" + re.sub(r"\W+", "", "".join(props.values())))
        bdir = fresh_dir(os.path.join(work, "build", tag))
        log = os.path.join(LOGS, f"arduino-{tag}.log")
        open(log, "w").close()
        cmd = [cli, "compile", "--fqbn", fqbn, "--library", pkg, "--warnings", "all",
               "--build-property", f"compiler.warning_flags.all={ARDUINO_WARNINGS}",
               "--build-property", "compiler.c.extra_flags=-Wall -Wextra",
               "--build-property", "compiler.cpp.extra_flags=-Wall -Wextra"]
        for k, v in props.items():
            cmd += ["--build-property", f"{k}={v}"]
        cmd += ["--build-path", bdir, sketch]
        b: Dict = {"fqbn": fqbn, "variant": label, "verification": "compiled, not run",
                   "command": "env PATH=" + shlex.quote(os.path.dirname(os.path.abspath(sys.executable)))
                              + ':"$PATH" ' + " ".join(shlex.quote(c) for c in cmd),
                   "log": rel(log)}
        core = ":".join(fqbn.split(":")[:2])
        if core not in tools["cores"] and "error" not in tools:
            b.update(status="not built", reason=f"Arduino core {core} is not installed")
            res["builds"].append(b)
            continue
        _say(f"  arduino: {fqbn} ({label})")
        t0 = time.time()
        rc, out = run(cmd + ["--json", "--verbose"], log, f"arduino-cli compile {fqbn}", env=env, timeout=3600,
                      check=False)
        b["seconds"] = round(time.time() - t0, 1)
        try:
            j = json.loads(out[out.index("{"):])
        except ValueError:
            j = {}
        text = (j.get("compiler_out") or "") + "\n" + (j.get("compiler_err") or "")
        with open(log, "a") as f:
            f.write("\n---- compiler output ----\n" + text + "\n")
        if rc != 0 or not j.get("success"):
            b.update(status="failed", exit_status=rc, output_tail=(text if text.strip() else out)[-3000:])
            res["builds"].append(b)
            continue
        b["status"] = "built"
        for sec in (j.get("builder_result") or {}).get("executable_sections_size") or []:
            if sec.get("name") == "text":
                b["program_bytes"], b["program_max_bytes"] = sec.get("size"), sec.get("max_size")
            elif sec.get("name") == "data":
                b["global_ram_bytes"], b["global_ram_max_bytes"] = sec.get("size"), sec.get("max_size")
        flags = compile_flags(bdir, f"{name}.c")
        if flags:
            b["model_compile"] = flags
            sizes = object_sizes(size_tool_for(flags["compiler"]),
                                 os.path.join(bdir, "libraries", os.path.basename(pkg), f"{name}.c.o"))
            if sizes:
                b["model_object"] = sizes
        sflags = compile_flags(bdir, f"{sketch_name}.ino.cpp")
        if sflags:
            b["sketch_compile"] = {k: sflags[k] for k in ("optimization", "warning_flags")}
        b.update(scan_warnings(text, marks))
        res["builds"].append(b)
    states = {b["status"] for b in res["builds"]}
    res["status"] = "built" if states == {"built"} else ("failed" if "failed" in states else "not built")
    return res


# --------------------------------------------------------------------------- ESP-IDF build

def build_esp_idf(pkg: str, name: str, work: str, marks: List[str], install: bool, inject: bool = False) -> Dict:
    res: Dict = {"package": rel(pkg), "target": ESP_IDF_TARGET, "verification": "compiled, not run",
                 "sdk": {"esp-idf": ESP_IDF_TAG, "path": rel(ESP_IDF_DIR), "tools_path": rel(ESP_IDF_TOOLS),
                         "checkout": "shallow, blobless, sparse: " + " ".join(ESP_IDF_SPARSE),
                         "submodules": ESP_IDF_SUBMODULES}}
    setup_log = os.path.join(LOGS, "esp-idf-setup.log")
    if install:
        try:
            setup_esp_idf(setup_log)
        except StepFailed as e:
            res.update(status="not built", reason="ESP-IDF could not be installed", error=e.as_dict(),
                       log=rel(setup_log))
            return res
    missing = sdk_missing("esp-idf")
    if missing:
        res.update(status="not built", reason=f"ESP-IDF is not installed under .sdk/ ({missing})")
        return res
    env = esp_idf_env()
    extra_cflags = ESP_IDF_EXTRA_CFLAGS
    if inject:
        # Under ESP-IDF's -Werror the injected -Wtype-limits warning is an error and stops the build
        # (which is a detection too); downgrade it so that the self-test sees all four diagnostics.
        extra_cflags += " -Wno-error=type-limits"
        res["self_test_note"] = ("without -Wno-error=type-limits, added for this self-test only, the injected "
                                 "-Wtype-limits warning is an error under ESP-IDF's -Werror and the build fails")
    env["EXTRA_CFLAGS"] = extra_cflags
    project = os.path.join(pkg, "examples", "predict")
    bdir = fresh_dir(os.path.join(work, "build"))
    log = os.path.join(LOGS, "esp-idf-build.log")
    open(log, "w").close()
    idf = (f'idf.py -C {shlex.quote(project)} -B {shlex.quote(bdir)} '
           f'-D SDKCONFIG={shlex.quote(os.path.join(bdir, "sdkconfig"))} -D IDF_TARGET={ESP_IDF_TARGET}')
    script = f'. {shlex.quote(os.path.join(ESP_IDF_DIR, "export.sh"))} && {idf} build'
    res["command"] = (f'env IDF_TOOLS_PATH={shlex.quote(ESP_IDF_TOOLS)} IDF_SKIP_CHECK_SUBMODULES=1 '
                      f'EXTRA_CFLAGS={shlex.quote(extra_cflags)} '
                      f'PATH={shlex.quote(os.path.dirname(os.path.abspath(sys.executable)))}:"$PATH" '
                      f'bash -c {shlex.quote(script)}')
    res["log"] = rel(log)
    _say(f"  esp-idf: idf.py build for {ESP_IDF_TARGET}")
    t0 = time.time()
    rc, out = run(["bash", "-c", script], log, "idf.py build", env=env, timeout=3600, check=False)
    res["seconds"] = round(time.time() - t0, 1)
    res["tools"] = {
        "idf.py": first_line(["bash", "-c", f'. {shlex.quote(os.path.join(ESP_IDF_DIR, "export.sh"))} '
                              '>/dev/null 2>&1 && idf.py --version'], env=env),
        "python": first_line([sys.executable, "--version"]),
    }
    if rc != 0:
        res.update(status="failed", exit_status=rc, output_tail=out[-3000:])
        return res
    res["status"] = "built"
    app = os.path.join(bdir, f"{name}_predict.bin")
    if os.path.exists(app):
        res["app_bin_bytes"] = os.path.getsize(app)
    boot = os.path.join(bdir, "bootloader", "bootloader.bin")
    if os.path.exists(boot):
        res["bootloader_bin_bytes"] = os.path.getsize(boot)
    flags = compile_flags(bdir, f"{name}.c")
    if flags:
        res["model_compile"] = flags
        res["tools"]["compiler"] = flags["compiler_version"]
        elf = os.path.join(bdir, f"{name}_predict.elf")
        tool = size_tool_for(flags["compiler"])
        try:
            r = subprocess.run([tool, elf], capture_output=True, text=True, timeout=120)
            parts = r.stdout.splitlines()[1].split()
            res["elf"] = {"text": int(parts[0]), "data": int(parts[1]), "bss": int(parts[2])}
        except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
            pass
        for d, _, fs in os.walk(os.path.join(bdir, "esp-idf")):
            for f in fs:
                if f == f"{name}.c.obj":
                    sizes = object_sizes(tool, os.path.join(d, f))
                    if sizes:
                        res["model_object"] = sizes
    m = re.search(r"cmake version (\S+)", out) or re.search(r"CMake version (\S+)", out)
    if m:
        res["tools"]["cmake"] = m.group(1)
    # "idf.py size" summary, kept verbatim
    rc2, size_out = run(["bash", "-c", f'. {shlex.quote(os.path.join(ESP_IDF_DIR, "export.sh"))} '
                         f'>/dev/null 2>&1 && {idf} size'], log, "idf.py size", env=env, timeout=600, check=False)
    if rc2 == 0:
        m = re.search(r"Total image size:\s*(\d+)", size_out)
        if m:
            res["total_image_bytes"] = int(m.group(1))
    res.update(scan_warnings(out, marks))
    return res


# --------------------------------------------------------------------------- Zephyr build

MEMORY_REGION = re.compile(r"^\s*(\w+):\s+(\d+)\s*(B|KB|MB|GB)\s+(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)\s+([\d.]+)%", re.M)
_UNIT = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}


def build_zephyr(pkg: str, name: str, work: str, marks: List[str], install: bool, inject: bool = False) -> Dict:
    res: Dict = {"package": rel(pkg), "builds": [],
                 "sdk": {"zephyr": ZEPHYR_TAG, "workspace": rel(ZEPHYR_WS), "modules": ZEPHYR_MODULES,
                         "zephyr-sdk": ZEPHYR_SDK_VERSION, "toolchain": ZEPHYR_SDK_TOOLCHAIN,
                         "sdk_path": rel(ZEPHYR_SDK_DIR)}}
    setup_log = os.path.join(LOGS, "zephyr-setup.log")
    if install:
        try:
            setup_zephyr(setup_log)
        except StepFailed as e:
            res.update(status="not built", reason="Zephyr could not be installed", error=e.as_dict(),
                       log=rel(setup_log))
            return res
    missing = sdk_missing("zephyr")
    if missing:
        res.update(status="not built", reason=f"Zephyr is not installed under .sdk/ ({missing})")
        return res
    env = zephyr_env()
    west = os.path.join(ZEPHYR_VENV, "bin", "west")
    gcc = os.path.join(ZEPHYR_SDK_DIR, "gnu", ZEPHYR_SDK_TOOLCHAIN, "bin", f"{ZEPHYR_SDK_TOOLCHAIN}-gcc")
    version = {}
    for line in open(os.path.join(ZEPHYR_WS, "zephyr", "VERSION")):
        k, _, v = line.partition("=")
        version[k.strip()] = v.strip()
    res["tools"] = {
        "zephyr": "{VERSION_MAJOR}.{VERSION_MINOR}.{PATCHLEVEL}".format(**version),
        "west": first_line([west, "--version"], env=env),
        "zephyr-sdk": open(os.path.join(ZEPHYR_SDK_DIR, "sdk_version")).read().strip(),
        "compiler": first_line([gcc, "--version"]),
        "cmake": first_line(["cmake", "--version"], env=env),
        "ninja": first_line(["ninja", "--version"], env=env),
        "python": first_line([os.path.join(ZEPHYR_VENV, "bin", "python"), "--version"]),
    }
    app = os.path.join(pkg, "samples", "predict")
    for board in ZEPHYR_BOARDS:
        tag = re.sub(r"\W+", "_", board)
        bdir = os.path.join(work, "build", tag)
        fresh_dir(bdir)
        log = os.path.join(LOGS, f"zephyr-{tag}.log")
        open(log, "w").close()
        cmd = [west, "build", "-p", "always", "-b", board, "-d", bdir, app, "--", "-DEXTRA_CFLAGS=-Wextra"]
        b: Dict = {"board": board, "verification": "compiled, not run", "log": rel(log),
                   "command": (f"env ZEPHYR_BASE={shlex.quote(env['ZEPHYR_BASE'])} "
                               f"ZEPHYR_SDK_INSTALL_DIR={shlex.quote(ZEPHYR_SDK_DIR)} "
                               + " ".join(shlex.quote(c) for c in cmd))}
        _say(f"  zephyr: west build for {board}")
        t0 = time.time()
        rc, out = run(cmd, log, f"west build {board}", env=env, timeout=3600, check=False)
        b["seconds"] = round(time.time() - t0, 1)
        if rc != 0:
            b.update(status="failed", exit_status=rc, output_tail=out[-3000:])
            res["builds"].append(b)
            continue
        b["status"] = "built"
        for m in MEMORY_REGION.finditer(out):
            region = m.group(1).lower()
            if region in ("flash", "ram"):
                b[f"{region}_used_bytes"] = int(m.group(2)) * _UNIT[m.group(3)]
                b[f"{region}_region_bytes"] = int(float(m.group(4)) * _UNIT[m.group(5)])
        zbin = os.path.join(bdir, "zephyr", "zephyr.bin")
        if os.path.exists(zbin):
            b["zephyr_bin_bytes"] = os.path.getsize(zbin)
        flags = compile_flags(bdir, f"{name}.c")
        if flags:
            b["model_compile"] = flags
            for d, _, fs in os.walk(os.path.join(bdir, "modules")):
                for f in fs:
                    if f == f"{name}.c.obj":
                        sizes = object_sizes(size_tool_for(flags["compiler"]), os.path.join(d, f))
                        if sizes:
                            b["model_object"] = sizes
        b.update(scan_warnings(out, marks))
        res["builds"].append(b)
    states = {b["status"] for b in res["builds"]}
    res["status"] = "built" if states == {"built"} else "failed"
    return res


BUILDERS = {"arduino": build_arduino, "esp-idf": build_esp_idf, "zephyr": build_zephyr}


# --------------------------------------------------------------------------- driver

def make_packages(kinds: List[str], work_root: str, inject: bool) -> Tuple[Dict, Dict[str, str], List[str]]:
    """Export the zoo model and write one package per kind.  Returns (model info, package dirs, marks)."""
    from xgb_edge import ExportOptions, export_model
    from xgb_edge.package import camel_case, kconfig_symbol, make_sample, write_package
    tr = zoo.train(zoo.by_name(ZOO_MODEL))
    exp = export_model(tr.model_path, ExportOptions(name=MODEL_NAME), validation=tr.X)
    smp = make_sample(exp, tr.X)
    info = {
        "zoo_model": ZOO_MODEL, "name": MODEL_NAME, "objective": exp.model.objective,
        "classes": exp.model.n_classes, "features": exp.model.n_features, "trees": len(exp.model.trees),
        "input": exp.options.input, "leaf_bits": exp.lplan.bits, "margin_shift": exp.lplan.shift,
        "model_data_bytes": exp.generated.data_bytes,
        "example_sample": smp.source,
        "example_expected_class": smp.expected_class, "example_expected_margins": smp.expected_margins,
    }
    dirs: Dict[str, str] = {}
    for kind in kinds:
        pkg = fresh_dir(os.path.join(work_root, kind, MODEL_NAME))
        write_package(exp, kind, pkg, sample=tr.X)
        if inject:
            inject_warnings(pkg, MODEL_NAME, kind)
        dirs[kind] = pkg
    marks = [MODEL_NAME, camel_case(MODEL_NAME) + "Predict", kconfig_symbol(MODEL_NAME), "predict_main", "xgb_edge"]
    return info, dirs, marks


def platform_ok(p: Dict) -> bool:
    builds = p.get("builds") or [p]
    return p.get("status") == "built" and all(
        b.get("status") == "built" and not b.get("warnings_in_generated_files") for b in builds)


def summarize(results: Dict) -> None:
    _say("")
    for kind, p in results["platforms"].items():
        for b in p.get("builds") or [p]:
            what = b.get("fqbn") or b.get("board") or p.get("target") or ""
            if b.get("variant") and b["variant"] != "core default":
                what += f" [{b['variant']}]"
            size = ""
            if "program_bytes" in b:
                size = f"program {b['program_bytes']:,} B, globals {b.get('global_ram_bytes', 0):,} B"
            elif "flash_used_bytes" in b:
                size = f"FLASH {b['flash_used_bytes']:,} B, RAM {b.get('ram_used_bytes', 0):,} B"
            elif "app_bin_bytes" in b:
                size = f"app binary {b['app_bin_bytes']:,} B"
            mo = b.get("model_object")
            if mo:
                size += f"; model object {mo['code_bytes']:,} B code + {mo['rodata_bytes']:,} B data"
            status = b.get("status", p.get("status"))
            extra = b.get("reason") or p.get("reason") or ""
            if status == "failed" and not extra:
                extra = f"exit status {b.get('exit_status')}, see {b.get('log') or p.get('log')}"
            if status == "built":
                ours = len(b.get("warnings_in_generated_files") or [])
                extra = (f"{ours or 'no'} warnings in generated files, "
                         f"{b.get('warnings_total', 0) - ours} in platform code; compiled, not run")
            _say(f"{kind:<8} {what:<40} {status:<10} {size}{'; ' if size and extra else ''}{extra}")
    _say(f"\nresults: {rel(results['results_file'])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=["arduino", "zephyr", "esp-idf"],
                    help="build only this platform (may be repeated; default: all three)")
    ap.add_argument("--no-install", action="store_true",
                    help="never download or install an SDK; report a platform whose SDK is missing as not built")
    ap.add_argument("--setup-only", action="store_true", help="install the SDKs and stop")
    ap.add_argument("--inject-warning", action="store_true",
                    help="self-test of the warning check: add one -Wall and one -Wextra warning to the packaged "
                         "files and build under build/packages/inject/ (results-inject.json); the builds must "
                         "then report warnings in the generated files, so the exit status is 0 only if they do")
    args = ap.parse_args(argv)
    kinds = args.only or ["arduino", "esp-idf", "zephyr"]
    os.makedirs(LOGS, exist_ok=True)
    if args.setup_only:
        rc = 0
        for k in kinds:
            if k in SETUP:
                _say(f"== {k}: setup")
                try:
                    SETUP[k](os.path.join(LOGS, f"{k}-setup.log"))
                    _say(f"   {k}: ready")
                except StepFailed as e:
                    rc = 1
                    _say(f"   {k}: setup FAILED at '{e.what}' (exit {e.rc})\n{e.tail}")
        return rc

    work_root = os.path.join(OUT, "inject") if args.inject_warning else OUT
    os.makedirs(work_root, exist_ok=True)
    results_file = os.path.join(OUT, "results-inject.json" if args.inject_warning else "results.json")
    info, dirs, marks = make_packages(kinds, work_root, args.inject_warning)
    from harness.zoo import code_stamp
    results: Dict = {
        "code": code_stamp("packages"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                 "python": platform.python_version()},
        "note": "every build is compile-only: nothing was uploaded to or run on a board",
        "model": info,
        "platforms": {},
        "results_file": results_file,
    }
    if args.inject_warning:
        results["note"] += "; SELF-TEST with warnings injected into the packaged files"
    for kind in kinds:
        _say(f"== {kind}")
        try:
            p = BUILDERS[kind](dirs[kind], MODEL_NAME, os.path.join(work_root, kind), marks, not args.no_install,
                               inject=args.inject_warning)
        except Exception as e:      # a bug here must not hide the other platforms' results
            import traceback
            p = {"status": "failed", "reason": f"harness error: {e!r}", "traceback": traceback.format_exc()}
        if args.inject_warning:
            for b in p.get("builds") or [p]:
                if b.get("status") == "built":
                    b["injected_warnings_reported"] = injected_seen(b.get("warnings_in_generated_files") or [])
        results["platforms"][kind] = p
    if args.inject_warning:
        seen = [b.get("injected_warnings_reported") for p in results["platforms"].values()
                for b in (p.get("builds") or [p])]
        results["ok"] = bool(seen) and all(s and all(s.values()) for s in seen)
    else:
        results["ok"] = all(platform_ok(p) for p in results["platforms"].values())
    results["results_file"] = rel(results_file)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    summarize(dict(results, results_file=results_file))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
