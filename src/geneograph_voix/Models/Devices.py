import os
import sys
import re
import warnings
import torch

from typing import Optional, List, Tuple

# =========================================
# Devices, threading, compute types
# =========================================

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _detect_physical_cores() -> Optional[int]:
    try:
        import importlib
        psutil = importlib.import_module("psutil")
        n = psutil.cpu_count(logical=False) or 0
        if n > 0:
            return int(n)
    except Exception:
        pass
    try:
        if sys.platform.startswith("win"):
            import subprocess
            out = subprocess.check_output(["wmic", "cpu", "get", "NumberOfCores"], text=True)
            nums = [int(x) for x in re.findall(r"\d+", out)]
            if nums:
                return sum(nums)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            import subprocess
            out = subprocess.check_output(["lscpu", "-p=Core"], text=True)
            ids = set()
            for line in out.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                core_id = line.split(",")[0].strip()
                if core_id:
                    ids.add(core_id)
            if ids:
                return len(ids)
    except Exception:
        pass
    try:
        n = os.cpu_count()
        return int(n) if n else None
    except Exception:
        return None

CPU_THREADS: Optional[int] = None
NUM_WORKERS: Optional[int] = None

_physical = _detect_physical_cores()
if _physical:
    CPU_THREADS = max(2, min(_physical, 8))
    NUM_WORKERS = 1
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def _resolve_compute_type(device: str) -> Tuple[List[str], str]:
    env_ct = os.getenv("WHISPER_COMPUTE_TYPE") or os.getenv("FAST_WHISPER_COMPUTE_TYPE")
    if env_ct:
        return [env_ct], env_ct
    if device == "cuda":
        chain = ["float16", "int8_float16", "float32"]
        return chain, chain[0]
    else:
        chain = ["int8", "float32"]
        return chain, chain[0]
