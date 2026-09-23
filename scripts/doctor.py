"""Environment check (``make doctor``, README R11, PHASES T1.3).

Prints one line per check with a status and exactly what is missing:

* OK   — present and usable
* WARN — optional or not needed yet (GPU, C++ backend, and data/weights until their phase)
* FAIL — required and missing; the script exits 1

``--require-data`` turns missing SemanticKITTI files into FAIL (Phase 2 onwards);
``--require-weights`` does the same for the model checkpoint (Phase 8 onwards).
Standard library only at module level, so it runs before ``pip install``.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "OK", "WARN", "FAIL"
SYMBOLS = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}

REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "yaml",
    "pydantic",
    "torch",
    "matplotlib",
    "sklearn",
    "cv2",
    "imageio",
    "fastapi",
    "uvicorn",
    "socketio",
    "pytest",
    "hypothesis",
    "foveamap",
]
MIN_FREE_DISK_GB = 10.0  # README 4.1: ~10 GB for sequences 04, 07, 08


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _version_tuple(text: str) -> tuple[int, ...]:
    """Version in a ``--version`` line; () if none.

    Prefers the number after the word "version" ("Apple clang version 15.0.0 (clang-1500...)" -> 15.0.0),
    else the last dotted number ("g++ (Rev6, MSYS2) 13.1.0" -> 13.1.0).
    """
    after_word = re.search(r"version\s+v?(\d+(?:\.\d+)+)", text)
    dotted = [after_word.group(1)] if after_word else re.findall(r"\d+(?:\.\d+)+", text)
    return tuple(int(p) for p in dotted[-1].split(".")[:3]) if dotted else ()


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout or out.stderr).strip() or None


def check_python() -> Check:
    v = sys.version_info
    status = OK if v >= (3, 10) else FAIL
    return Check("python", status, f"{v.major}.{v.minor}.{v.micro} ({sys.executable}); need >= 3.10")


def check_packages() -> list[Check]:
    checks = []
    for name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            hint = "pip install -e ." if name == "foveamap" else "pip install -r requirements.txt"
            checks.append(Check(f"pkg:{name}", FAIL, f"not importable ({type(exc).__name__}); run `{hint}`"))
        else:
            checks.append(Check(f"pkg:{name}", OK, getattr(mod, "__version__", "installed")))
    return checks


def check_gpu() -> Check:
    try:
        import torch
    except ImportError:
        return Check("gpu", WARN, "torch not importable; see pkg:torch")
    if torch.cuda.is_available():
        return Check(
            "gpu",
            OK,
            f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}, torch {torch.__version__})",
        )
    return Check(
        "gpu",
        WARN,
        f"CUDA not available (torch {torch.__version__}); live inference and the e2e latency benchmark "
        "need a GPU machine; everything else runs from the prediction cache (R13)",
    )


def check_tool(name: str, cmd: list[str], minimum: tuple[int, ...] | None, why: str) -> Check:
    if shutil.which(cmd[0]) is None:
        return Check(name, WARN, f"not found on PATH; needed for {why}")
    out = _run(cmd) or ""
    first = out.splitlines()[0] if out else "unknown version"
    if minimum and _version_tuple(first) and _version_tuple(first) < minimum:
        need = ".".join(map(str, minimum))
        return Check(name, WARN, f"{first}; need >= {need} for {why}")
    return Check(name, OK, first)


def check_node() -> Check:
    if shutil.which("node") is None:
        return Check("node", FAIL, "not found; Node.js >= 18 is required for the dashboard (README 4.1)")
    out = _run(["node", "--version"]) or ""
    if _version_tuple(out) < (18,):
        return Check("node", FAIL, f"{out}; need >= 18")
    return Check("node", OK, out)


def check_compilers() -> list[Check]:
    checks = [check_tool("cmake", ["cmake", "--version"], (3, 20), "the optional C++ backend (tier P2)")]
    compiler = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    if compiler is None:
        checks.append(
            Check("c++ compiler", WARN, "no g++/clang++ found; needed only for the optional C++ backend")
        )
    else:
        checks.append(check_tool("c++ compiler", [compiler, "--version"], (9,), "the optional C++ backend"))
    return checks


def check_ffmpeg() -> Check:
    found = shutil.which("ffmpeg")
    if found:
        return Check("ffmpeg", OK, found)
    try:
        import imageio_ffmpeg

        return Check("ffmpeg", OK, f"bundled via imageio-ffmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
    except Exception:  # noqa: BLE001
        return Check(
            "ffmpeg", FAIL, "not found (neither on PATH nor via imageio-ffmpeg); needed for `make video`"
        )


def check_disk() -> Check:
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    status = OK if free_gb >= MIN_FREE_DISK_GB else WARN
    return Check(
        "disk",
        status,
        f"{free_gb:.1f} GB free at {ROOT}; ~{MIN_FREE_DISK_GB:.0f} GB needed for sequences 04/07/08",
    )


def check_data(data_root: Path, sequences: list[str], required: bool) -> list[Check]:
    missing_status = FAIL if required else WARN
    checks = []
    for seq in sequences:
        seq_dir = data_root / "sequences" / seq
        problems = []
        n_scans = len(list((seq_dir / "velodyne").glob("*.bin"))) if (seq_dir / "velodyne").is_dir() else 0
        n_labels = len(list((seq_dir / "labels").glob("*.label"))) if (seq_dir / "labels").is_dir() else 0
        if n_scans == 0:
            problems.append("no velodyne/*.bin")
        if n_labels == 0:
            problems.append("no labels/*.label")
        elif n_scans and n_labels != n_scans:
            problems.append(f"{n_labels} labels vs {n_scans} scans")
        if not (seq_dir / "calib.txt").is_file():
            problems.append("no calib.txt")
        if not ((seq_dir / "poses.txt").is_file() or (data_root / "poses" / f"{seq}.txt").is_file()):
            problems.append("no poses")
        name = f"data:{seq}"
        if problems:
            hint = "download per README 4.3, then run scripts/verify_data.py"
            checks.append(Check(name, missing_status, f"{seq_dir}: {', '.join(problems)}; {hint}"))
        else:
            checks.append(Check(name, OK, f"{n_scans} scans, {n_labels} labels"))
    return checks


def check_weights(required: bool) -> Check:
    missing_status = FAIL if required else WARN
    model_yaml = ROOT / "configs" / "model.yaml"
    try:
        import yaml

        cfg = yaml.safe_load(model_yaml.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return Check("weights", missing_status, f"cannot read {model_yaml}: {exc}")
    name, ckpt = cfg.get("name"), cfg.get("checkpoint")
    if not name or not ckpt:
        return Check(
            "weights",
            missing_status,
            "no model chosen yet (configs/model.yaml name/checkpoint are null; Phase 8)",
        )
    path = ROOT / ckpt
    if not path.is_file():
        return Check("weights", missing_status, f"{ckpt} not found; download per README 4.3 step 4")
    return Check("weights", OK, f"{name}: {ckpt} ({path.stat().st_size / 1e6:.1f} MB)")


def check_cpp() -> Check:
    try:
        importlib.import_module("foveamap._fovea_cpp")
    except ImportError:
        return Check("cpp backend", WARN, "foveamap._fovea_cpp not built; the NumPy backend is used (L8)")
    return Check("cpp backend", OK, "foveamap._fovea_cpp importable")


def run_checks(args: argparse.Namespace) -> list[Check]:
    checks = [
        check_python(),
        *check_packages(),
        check_gpu(),
        *check_compilers(),
        check_node(),
        check_ffmpeg(),
    ]
    checks.append(check_disk())
    checks.extend(check_data(Path(args.data_root), args.sequences, args.require_data))
    checks.append(check_weights(args.require_weights))
    checks.append(check_cpp())
    return checks


def _symbols_supported() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        "".join(SYMBOLS.values()).encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(ROOT / "data" / "dataset"))
    parser.add_argument("--sequences", nargs="+", default=["04", "07", "08"])
    parser.add_argument("--require-data", action="store_true", help="missing data is FAIL (Phase 2+)")
    parser.add_argument("--require-weights", action="store_true", help="missing weights are FAIL (Phase 8+)")
    args = parser.parse_args(argv)

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    use_symbols = _symbols_supported()

    checks = run_checks(args)
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = SYMBOLS[c.status] if use_symbols else f"[{c.status}]"
        print(f"{mark} {c.name:<{width}}  {c.detail}")

    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    print(f"\n{len(checks)} checks: {len(checks) - n_fail - n_warn} ok, {n_warn} warnings, {n_fail} failures")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
