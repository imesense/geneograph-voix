import os
import sys

# =========================================
# Writable caches (helps PyInstaller one-folder)
# =========================================
def _prepare_frozen_caches():
    try:
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))

        cache_dir = os.path.join(base, "torch_cache")
        os.makedirs(cache_dir, exist_ok=True)

        os.environ.setdefault("TORCH_HOME", cache_dir)
        os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))

        # Avoid symlink/hardlink operations that cause WinError 1314
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WINDOWS", "1")

        # Runtime threading hints (only if user hasn't set them)
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("KMP_BLOCKTIME", "0")
    except Exception:
        pass
