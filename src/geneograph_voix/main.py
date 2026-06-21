import os
import sys

import tkinter as tk

from geneograph_voix.Helpers.ScreenHelpers import (
    _fit_to_screen,
    _set_windows_dpi_awareness
)
from geneograph_voix.Models.Cache import _prepare_frozen_caches
from geneograph_voix.Models.Config import (
    DATA_DIR
)
from geneograph_voix.Models.Settings import (
    MODEL_KEY,
    _apply_settings_to_globals,
    _read_settings_from_file
)
from geneograph_voix.Models.SileroVadSettings import (
    _load_silero_vad
)
from geneograph_voix.Models.WhisperModelWrapper import (
    _load_whisper_model,
    _pick_default_model_key,
    _warmup_model
)
from geneograph_voix.SpeechSheetApp import SpeechSheetApp
from geneograph_voix.Views.Styles import (
    _set_app_icon,
    enable_crisp_dark_mode
)

_prepare_frozen_caches()

os.makedirs(DATA_DIR, exist_ok=True)

# ===========================
# App entry
# ===========================

def _boost_process_priority_windows():
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        HIGH_PRIORITY_CLASS = 0x00000080
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)
    except Exception:
        pass

if __name__ == "__main__":
    # Load saved settings and decide defaults
    s = _read_settings_from_file()
    _apply_settings_to_globals(s)
    if not MODEL_KEY:
        MODEL_KEY = _pick_default_model_key()

    # Preload Silero + Whisper + warmup BEFORE UI shows
    _load_silero_vad()
    _load_whisper_model(MODEL_KEY)
    _warmup_model()

    _boost_process_priority_windows()
    _set_windows_dpi_awareness()
    root = tk.Tk()

    try:
        _set_app_icon(root)
        enable_crisp_dark_mode(root, dark=True, delay_ms=350)
    except Exception:
        pass
    try: _fit_to_screen(root, margin=60)
    except Exception: 
        pass

    app = SpeechSheetApp(root)
    root.mainloop()
