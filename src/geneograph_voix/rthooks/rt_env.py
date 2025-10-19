# rthooks/rt_env.py
import os, sys

def _base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)  # dist\geneograph_voix
    return os.getcwd()

base = _base_dir()
os.environ.setdefault("TORCH_HOME", os.path.join(base, "torch_cache"))
os.environ.setdefault("HF_HOME",    os.path.join(base, "torch_cache", "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# If you ever meet MKL/OMP collisions:
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
