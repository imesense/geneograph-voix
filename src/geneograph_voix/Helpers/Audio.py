import queue
import re
import time

import numpy as np

from typing import Optional

from geneograph_voix.Helpers.Language import looks_like_outro
from geneograph_voix.Models.Config import BAN_PHRASES
from geneograph_voix.Models.ModelCoefficients import _energy_gate_db

# =========================================
# Glossary + text utilities (adapted)
# =========================================

def _preview_is_banned(text: str) -> bool:
    try:
        return looks_like_outro(text)
    except Exception:
        t = (text or "").lower().replace("ё", "е")
        t = re.sub(r"[^a-zа-я0-9\s]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if "продолжение следует" in t:
            return True
        return any(p in t for p in BAN_PHRASES)

def clean_person_field(text: str) -> str:
    t = text or ""
    t = re.sub(r"[^A-Za-zА-Яа-яЁё\-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    return " ".join(w[:1].upper() + w[1:].lower() if w else "" for w in t.split())

AUDIO_QUEUE = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    global LAST_ACTIVITY_TS
    try:
        x = np.asarray(indata, dtype=np.float32)
        if x.size > 0:
            if rms_db(x.flatten()) > (_energy_gate_db() + 3.0):
                LAST_ACTIVITY_TS = time.monotonic()
    except Exception:
        LAST_ACTIVITY_TS = time.monotonic()
    AUDIO_QUEUE.put(indata.copy())

def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    x = x.astype(np.float64)
    rms = np.sqrt(np.mean(x**2))
    return 20*np.log10(rms + 1e-12)

def is_silence(buf: np.ndarray, threshold_db: Optional[float] = None) -> bool:
    if buf is None or getattr(buf, "size", 0) == 0:
        return True
    if threshold_db is None:
        threshold_db = _energy_gate_db()
    return rms_db(np.asarray(buf).flatten()) < float(threshold_db)
