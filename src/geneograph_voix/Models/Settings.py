import json
import os

from geneograph_voix.Models.Config import (
    _MODELKEY_TO_LABEL,
    AUTOSAVE_EVERY_MS,
    SETTINGS_FILE,
    SPEED_MODE,
    UI_SCALE
)
from geneograph_voix.Models.ModelCoefficients import (
    GLOSSARY_STRICTNESS,
    VAD_STRICTNESS
)

# =========================================
# Settings helpers
# =========================================
LANGUAGE = "ru"
MODEL_KEY = None

def _read_settings_from_file() -> dict:
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            if isinstance(s, dict):
                return s
    except Exception as e:
        print("settings load error:", e)
    return {}

def _apply_settings_to_globals(s: dict):
    global LANGUAGE, AUTOSAVE_EVERY_MS, UI_SCALE, MODEL_KEY, SPEED_MODE, VAD_STRICTNESS, GLOSSARY_STRICTNESS
    try:
        if "language" in s and isinstance(s["language"], str):
            LANGUAGE = s["language"].strip() or "auto"
        if "autosave_minutes" in s:
            m = max(1, int(s["autosave_minutes"]))
            AUTOSAVE_EVERY_MS = m * 60 * 1000
        if "ui_scale" in s:
            UI_SCALE = float(s["ui_scale"])
        if "model_key" in s and s["model_key"] in _MODELKEY_TO_LABEL:
            MODEL_KEY = s["model_key"]
        if "speed_mode" in s:
            SPEED_MODE = bool(s["speed_mode"])
        if "vad_strictness" in s:
            try:
                lvl = int(s["vad_strictness"])
                globals()["VAD_STRICTNESS"] = max(1, min(5, lvl))
            except Exception:
                pass
        if "glossary_strictness" in s:
            try:
                lvl = int(s["glossary_strictness"])
                globals()["GLOSSARY_STRICTNESS"] = max(1, min(5, lvl))
            except Exception:
                pass
        if "remove_punct" in s:
            globals()["REMOVE_PUNCT"] = bool(s["remove_punct"])
    except Exception as e:
        print("settings apply error:", e)

def _save_settings_file(s: dict):
    try:
        # Merge with existing settings so geometry & other runtime keys survive
        old = _read_settings_from_file()
        if not isinstance(old, dict):
            old = {}
        old.update(s)

        tmp = os.path.join(os.path.dirname(SETTINGS_FILE) or ".", "~settings.tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=2)
        os.replace(tmp, SETTINGS_FILE)
    except Exception as e:
        print("settings save error:", e)
