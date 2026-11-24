"""
gpu_utils.py — GPU memory monitoring and gating for the CocoPalm pipeline.

Used by run_pipeline.py and prepare_dinov3_dataset.py to:
  • Display real-time GPU memory stats at every phase boundary
  • Block (without killing) until enough GPU memory is free
  • Set the CUDA device visible to PyTorch

All queries go through nvidia-smi so no extra Python package is needed.
"""
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def get_gpu_memory_info(device_idx: int = 0) -> dict | None:
    """
    Query nvidia-smi for one GPU's memory stats.

    Returns:
        dict(total_mib, used_mib, free_mib, util_pct)  or  None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={device_idx}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "total_mib": int(parts[0]),
            "used_mib":  int(parts[1]),
            "free_mib":  int(parts[2]),
            "util_pct":  int(parts[3]),
        }
    except Exception:
        return None


def _mem_bar(used_mib: int, total_mib: int, width: int = 30) -> str:
    """ASCII progress bar for GPU memory usage."""
    frac   = used_mib / max(total_mib, 1)
    filled = int(width * frac)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_gpu_stats(device_idx: int = 0, label: str = "GPU stats") -> None:
    """Pretty-print current GPU memory and utilisation."""
    info = get_gpu_memory_info(device_idx)
    print(f"\n  ┌─ {label}")
    if info is None:
        print("  │  nvidia-smi not available — cannot report GPU stats.")
        print("  └─")
        return
    bar  = _mem_bar(info["used_mib"], info["total_mib"])
    pct  = info["used_mib"] / info["total_mib"] * 100
    print(
        f"  │  Memory  {bar}  "
        f"{info['used_mib']:,} / {info['total_mib']:,} MiB  ({pct:.0f}% used)"
    )
    print(
        f"  │  Free    {info['free_mib']:,} MiB   │   GPU util: {info['util_pct']}%"
    )
    print("  └─")


# ---------------------------------------------------------------------------
# Memory gate — wait until enough free GPU memory is available
# ---------------------------------------------------------------------------

def wait_for_gpu_memory(
    required_mib: int,
    device_idx: int = 0,
    interval: int = 20,
    label: str = "",
) -> None:
    """
    Block until `required_mib` MiB of GPU memory is free on `device_idx`.

    • Checks every `interval` seconds (default 20 s).
    • Prints a status line each time it has to wait.
    • NEVER kills any process — just pauses this one.
    • Returns immediately if nvidia-smi is unavailable.
    """
    tag = f"[GPU-wait{' ' + label if label else ''}]"

    first_check = True
    while True:
        info = get_gpu_memory_info(device_idx)

        if info is None:
            print(f"  {tag} nvidia-smi not available — skipping memory gate.")
            return

        bar = _mem_bar(info["used_mib"], info["total_mib"])

        if info["free_mib"] >= required_mib:
            if not first_check:
                # Only print the "ready" message if we actually had to wait
                print(
                    f"\n  {tag} GPU memory now OK — "
                    f"{info['free_mib']:,} MiB free "
                    f"(needed {required_mib:,} MiB).  {bar}  Proceeding.",
                    flush=True,
                )
            else:
                print(
                    f"  {tag} GPU memory OK — "
                    f"{info['free_mib']:,} MiB free "
                    f"(need {required_mib:,} MiB).  {bar}",
                    flush=True,
                )
            return

        print(
            f"  {tag} Low GPU memory: {info['free_mib']:,} MiB free, "
            f"need {required_mib:,} MiB.  {bar}  "
            f"Waiting {interval} s …",
            flush=True,
        )
        first_check = False
        time.sleep(interval)
