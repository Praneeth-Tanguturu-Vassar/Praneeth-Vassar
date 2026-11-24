"""
preflight.py — Pre-flight validation for the CocoPalm pipeline.

All requirements are verified ONCE before any phase starts.
If anything fails the pipeline exits cleanly — it will NEVER die
unexpectedly halfway through a long GPU run.

Checks per mode
---------------
  train    : imports (torch, lightning …), image dir + CSV exist and are readable,
             CSV has the required columns, checkpoint dir is writable,
             GPU total memory ≥ TRAIN_GPU_REQUIRED_MIB
  predict  : imports (torch, rasterio …), checkpoint file exists,
             AOI folder has rasters, output is writable,
             GPU total memory ≥ PREDICT_GPU_REQUIRED_MIB
  both     : train + predict checks
"""
import importlib
import os
import shutil
import sys
from pathlib import Path

from gpu_utils import get_gpu_memory_info, print_gpu_stats


# ---------------------------------------------------------------------------
# Low-level check helpers  (each returns None on pass, error string on fail)
# ---------------------------------------------------------------------------

def _chk_import(pkg: str) -> str | None:
    try:
        importlib.import_module(pkg)
        return None
    except ImportError as e:
        return f"Cannot import '{pkg}': {e}"


def _chk_path(var: str, path: str, kind: str = "path") -> str | None:
    if not path or "/path/to/" in path:
        return f"{var} is still a placeholder — edit the CONFIGURATION section."
    if not Path(path).exists():
        return f"{var} = '{path}' does not exist ({kind} not found)."
    return None


def _chk_dir_writable(var: str, path: str) -> str | None:
    p = Path(path)
    target = p if p.exists() else p.parent
    if not target.exists():
        return f"{var}: parent directory '{target}' does not exist."
    if not os.access(str(target), os.W_OK):
        return f"{var} = '{path}' is not writable."
    return None


def _chk_has_files(var: str, folder: str, patterns: list[str]) -> str | None:
    p = Path(folder)
    for pat in patterns:
        if any(p.rglob(pat)):
            return None
    return f"{var} = '{folder}' has no files matching {patterns}."


def _chk_disk(path: str, required_gb: float) -> str | None:
    target = Path(path)
    target = target if target.exists() else target.parent
    if not target.exists():
        return None  # will be caught by dir check
    free_gb = shutil.disk_usage(str(target)).free / (1024 ** 3)
    if free_gb < required_gb:
        return (
            f"Disk at '{path}': only {free_gb:.1f} GB free, "
            f"need at least {required_gb:.0f} GB."
        )
    return None


def _chk_csv_cols(var: str, csv_path: str, required: list[str]) -> str | None:
    try:
        with open(csv_path) as f:
            header = [c.strip() for c in f.readline().split(",")]
        missing = [c for c in required if c not in header]
        if missing:
            return f"{var}: CSV missing columns {missing}  (found: {header})"
    except Exception as e:
        return f"{var}: cannot read CSV '{csv_path}': {e}"
    return None


def _chk_gpu_total(required_mib: int, device_idx: int) -> str | None:
    info = get_gpu_memory_info(device_idx)
    if info is None:
        return None  # nvidia-smi unavailable — warn, don't block
    if info["total_mib"] < required_mib:
        return (
            f"GPU {device_idx} total memory {info['total_mib']:,} MiB < "
            f"required {required_mib:,} MiB.  This GPU may be too small."
        )
    return None


# ---------------------------------------------------------------------------
# Per-phase check suites
# ---------------------------------------------------------------------------

_PHASE_PKGS = ["torch", "torchvision", "lightning", "torchgeo"]
_TRAIN_CSV_COLS = [
    "image_name_1", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "label"
]


def _checks_train(cfg: dict) -> list[str]:
    errs = []
    for pkg in _PHASE_PKGS:
        e = _chk_import(pkg); errs += [e] if e else []

    e = _chk_path("TRAIN_IMAGE_DIR", cfg["TRAIN_IMAGE_DIR"], "directory")
    if e:
        errs.append(e)
    else:
        e = _chk_has_files(
            "TRAIN_IMAGE_DIR", cfg["TRAIN_IMAGE_DIR"], ["*.tif", "*.jpg", "*.png"]
        )
        if e: errs.append(e)

    e = _chk_path("TRAIN_CSV_FILE", cfg["TRAIN_CSV_FILE"], "file")
    if e:
        errs.append(e)
    else:
        e = _chk_csv_cols("TRAIN_CSV_FILE", cfg["TRAIN_CSV_FILE"], _TRAIN_CSV_COLS)
        if e: errs.append(e)

    e = _chk_dir_writable("CHECKPOINT_DIR", cfg["CHECKPOINT_DIR"])
    if e: errs.append(e)

    device_idx = (cfg.get("DEVICES") or [0])[0]
    e = _chk_gpu_total(cfg.get("TRAIN_GPU_REQUIRED_MIB", 4000), device_idx)
    if e: errs.append(e)
    return errs


def _checks_predict(cfg: dict) -> list[str]:
    errs = []
    for pkg in _PHASE_PKGS:
        e = _chk_import(pkg); errs += [e] if e else []

    e = _chk_path("CKPT_PATH", cfg["CKPT_PATH"], "file")
    if e: errs.append(e)

    e = _chk_path("AOI_FOLDER", cfg["AOI_FOLDER"], "directory")
    if e:
        errs.append(e)
    else:
        e = _chk_has_files(
            "AOI_FOLDER", cfg["AOI_FOLDER"], ["*.jp2", "*.tif", "*.tiff"]
        )
        if e: errs.append(e)

    e = _chk_dir_writable("OUTPUT_DIR", cfg["OUTPUT_DIR"])
    if e: errs.append(e)

    device_idx = (cfg.get("DEVICES") or [0])[0]
    e = _chk_gpu_total(cfg.get("PREDICT_GPU_REQUIRED_MIB", 2000), device_idx)
    if e: errs.append(e)
    return errs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_PHASE_CHECKS = {
    "train":   [("Phase 1 — Training",   _checks_train)],
    "predict": [("Phase 2 — Prediction", _checks_predict)],
    "both":    [("Phase 1 — Training",   _checks_train),
                ("Phase 2 — Prediction", _checks_predict)],
}


def run_preflight(mode: str, cfg: dict) -> bool:
    """
    Run all pre-flight checks for `mode`.

    Parameters
    ----------
    mode : str
        One of 'prepare', 'train', 'predict', 'both', 'all'.
    cfg  : dict
        All pipeline config variables as a flat dict.

    Returns
    -------
    bool
        True  — all checks passed, pipeline may proceed.
        False — one or more checks failed, pipeline should exit.
    """
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECKS")
    print("=" * 60)
    print(f"  Pipeline mode : {mode}")

    # Show GPU state before anything starts
    device_idx = (cfg.get("DEVICES") or [0])[0]
    print_gpu_stats(device_idx, label=f"GPU {device_idx} status before pipeline")

    all_errors: list[str] = []

    for phase_label, check_fn in _PHASE_CHECKS.get(mode, []):
        errors = check_fn(cfg)
        status = "PASS" if not errors else f"FAIL  ({len(errors)} error(s))"
        print(f"\n  {phase_label:30s} … {status}")
        for err in errors:
            print(f"    ✗  {err}")
            all_errors.append(err)
        if not errors:
            print(f"    ✓  All checks passed.")

    print()
    if all_errors:
        print(f"  ╔══ PRE-FLIGHT FAILED — {len(all_errors)} error(s) ══╗")
        print( "  ║  Fix the issues above, then re-run.             ║")
        print( "  ║  The pipeline will NOT start.                   ║")
        print( "  ╚═══════════════════════════════════════════════  ╝")
        print("=" * 60)
        return False

    print( "  ╔══ PRE-FLIGHT PASSED — all checks OK ══╗")
    print( "  ║  Pipeline is starting now.             ║")
    print( "  ╚═══════════════════════════════════════╝")
    print("=" * 60)
    return True
