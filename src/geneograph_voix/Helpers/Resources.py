import os
import sys

def _resource_path(name: str) -> str:
    """Return an absolute path to a resource, working in dev and PyInstaller one-file."""
    try:
        base = getattr(sys, "_MEIPASS", None)  # PyInstaller temp dir
        if base:
            return os.path.join(base, name)
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
