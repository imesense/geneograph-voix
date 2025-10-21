import threading
import queue
import time
import os
import json
import sys
import re
import warnings
import unicodedata
import torch
import inspect

import tkinter as tk
import numpy as np
import pandas as pd
import sounddevice as sd

from collections import namedtuple, deque
from contextlib import contextmanager
from datetime import date
from functools import lru_cache
from typing import Optional, List, Tuple, Callable
from tkinter import filedialog, messagebox, simpledialog
from tksheet import Sheet
from faster_whisper import WhisperModel

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

_prepare_frozen_caches()

# =========================================
# Global config / settings
# =========================================
DATA_FILE = "records.csv"
GLOSSARY_FILE = "glossary.json"
SETTINGS_FILE = "settings.json"

# Language options
WHISPER_LANG_CHOICES = [
    ("Auto (detect)", "auto"),
    ("Russian",   "ru"),
    ("Ukrainian", "uk"),
    ("Polish",    "pl"),
    ("English",   "en"),
    ("German",    "de"),
]
_LANG_LABELS = [lbl for (lbl, _) in WHISPER_LANG_CHOICES]
_LABEL_TO_CODE = {lbl: code for (lbl, code) in WHISPER_LANG_CHOICES}
_CODE_TO_LABEL = {code: lbl for (lbl, code) in WHISPER_LANG_CHOICES}

# Model picker: UI label ↔ key ↔ faster-whisper name
MODEL_CHOICES = [
    ("Tiny",     "tiny"),
    ("Base",     "base"),
    ("Small",    "small"),
    ("Medium",   "medium"),
    ("Large", "large-v3"),
    ("Turbo",    "large-v3-turbo"), 
]
_MODEL_LABELS = [x[0] for x in MODEL_CHOICES]
_LABEL_TO_MODELKEY = {lbl: key for (lbl, key) in MODEL_CHOICES}
_MODELKEY_TO_LABEL = {key: lbl for (lbl, key) in MODEL_CHOICES}

# Map model_key to actual checkpoint id
def _model_id_for_key(model_key: str) -> str:
    return model_key

# UI tuning
UI_SCALE = 1.0
TABLE_ZOOM_PCT = 135
PREVIEW_CLEAR_DELAY_MS = 1800

# Audio
SAMPLERATE = 16000
AUDIO_BLOCK_SEC = 0.10   # capture block size for low latency
BLOCK_DURATION = 5       # seconds to force a commit if user speaks continuously

# Silero VAD (only)
SILERO_THRESHOLD = 0.40
SILERO_MIN_SPEECH_MS = 120
SILERO_MIN_SILENCE_MS = 250
SILERO_PAD_MS = 250

# Preview + speed knobs
SPEED_MODE = False         # exposed in Settings
PREVIEW_MIN_INTERVAL_SEC = 0.35
PREVIEW_TAIL_SEC_DEFAULT = 0.60
PREVIEW_TAIL_SEC_SPEED   = 0.50

def _preview_tail_sec() -> float:
    return PREVIEW_TAIL_SEC_SPEED if SPEED_MODE else PREVIEW_TAIL_SEC_DEFAULT

# Autosave
AUTOSAVE_EVERY_MS = 5 * 60 * 1000   # 5 minutes

# Columns / cleaning
NAME_COLUMNS = {"Имя", "Фамилия", "Имя отца", "Имя матери", "Имя Матери"}

# Outro ban list
BAN_PHRASES = tuple(s.lower() for s in (
    "Субтитры сделал DimaTorzok",
    "Субтитры создал DimaTorzok",
    "Субтитры создавал DimaTorzok",
    "Субтитры сделал DimaTorzhok",
    "Субтитры создал DimaTorzhok",
    "Dima Torzhok", "DimaTorzok", "DimaTorzhok",
    "Продолжение следует", "Субтитры",
    "Редактор субтитров А.Семкин Корректор А.Егорова",
    "Редактор субтитров","Редактор", "Спасибо", "Thank you", "Смотрите На Видео", "Увидимся"
))
MALE_EXCEPTIONS = {
    "акила","арефа","вавила","варнава","иеремия","иона","исая","иуда",
    "калина","лука","осия","оссия","папа","фока","фома","никита",
    "савва","илья","кузьма","мина","сила",
}

# =========================================
# Devices, threading, compute types
# =========================================
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

def _detect_physical_cores() -> Optional[int]:
    """Best-effort physical core count without requiring psutil."""
    # 1) Try psutil via dynamic import (no Pylance warning; safe if missing)
    try:
        import importlib
        psutil = importlib.import_module("psutil")  # optional dep
        n = psutil.cpu_count(logical=False) or 0
        if n > 0:
            return int(n)
    except Exception:
        pass

    # 2) Windows WMIC fallback
    try:
        if sys.platform.startswith("win"):
            import subprocess
            out = subprocess.check_output(["wmic", "cpu", "get", "NumberOfCores"], text=True)
            nums = [int(x) for x in re.findall(r"\d+", out)]
            if nums:
                return sum(nums)
    except Exception:
        pass

    # 3) Linux fallback via lscpu (counts unique core IDs)
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

    # 4) Last resort: logical cores (not physical, but better than None)
    try:
        n = os.cpu_count()
        return int(n) if n else None
    except Exception:
        return None

CPU_THREADS: Optional[int] = None
NUM_WORKERS: Optional[int] = None
_physical = _detect_physical_cores()
if _physical:
    # Keep it modest to avoid starving UI
    CPU_THREADS = max(2, min(_physical, 8))
    NUM_WORKERS = 1
    # Only set env if user didn't
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def _resolve_compute_type(device: str) -> Tuple[List[str], str]:
    """
    Returns (fallback_chain, initial_picked_string)
    CPU: prefer int8 -> float32
    CUDA: prefer float16 -> int8_float16 -> float32
    """
    env_ct = os.getenv("WHISPER_COMPUTE_TYPE") or os.getenv("FAST_WHISPER_COMPUTE_TYPE")
    if env_ct:
        return [env_ct], env_ct

    if device == "cuda":
        chain = ["float16", "int8_float16", "float32"]
        return chain, chain[0]
    else:
        chain = ["int8", "float32"]
        return chain, chain[0]

# =========================================
# Settings helpers
# =========================================
LANGUAGE = "ru"  # may be overridden by settings
MODEL_KEY = None # decided from settings or default per device

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
    global LANGUAGE, AUTOSAVE_EVERY_MS, UI_SCALE, MODEL_KEY, SPEED_MODE
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
    except Exception as e:
        print("settings apply error:", e)

def _save_settings_file(s: dict):
    try:
        tmp = os.path.join(os.path.dirname(SETTINGS_FILE) or ".", "~settings.tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        os.replace(tmp, SETTINGS_FILE)
    except Exception as e:
        print("settings save error:", e)

# =========================================
# Glossary + text utilities (unchanged core logic)
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

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    # Update last activity (silence guard): consider any incoming frame as activity
    global LAST_ACTIVITY_TS
    try:
        x = np.asarray(indata, dtype=np.float32)
        if x.size > 0:
            if rms_db(x.flatten()) > -55.0:
                LAST_ACTIVITY_TS = time.monotonic()
    except Exception:
        LAST_ACTIVITY_TS = time.monotonic()
    audio_queue.put(indata.copy())

def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    x = x.astype(np.float64)
    rms = np.sqrt(np.mean(x**2))
    return 20*np.log10(rms + 1e-12)

def is_silence(buf: np.ndarray, threshold_db: float = -45.0) -> bool:
    return rms_db(buf.flatten()) < threshold_db

# ===========================
# Silero VAD only
# ===========================
_has_silero = False
_silero_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_silero_vad():
    global _has_silero, _silero_model, _get_speech_ts
    try:
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            force_reload=False
        )
        (_get_speech_ts, *_rest) = _silero_utils
        _silero_model.to(_silero_device)
        _silero_model.eval()
        # keep PyTorch from hogging CPU threads if VAD is on CPU
        if _silero_device == "cpu":
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
        _has_silero = True
        print(f"Silero VAD ready on {_silero_device}")
    except Exception as e:
        print("Silero VAD not available:", e)
        _has_silero = False

def _silero_vad_trim(buf_f32: np.ndarray, sr: int = SAMPLERATE) -> Optional[np.ndarray]:
    if not _has_silero or buf_f32 is None or getattr(buf_f32, "size", 0) == 0:
        return buf_f32
    # quick RMS pre-gate: if too silent, skip VAD entirely
    try:
        if rms_db(buf_f32) < -50.0:
            return None
    except Exception:
        pass
    wav = torch.from_numpy(buf_f32).float()
    ts = _get_speech_ts(
        wav, _silero_model,
        sampling_rate=sr,
        threshold=SILERO_THRESHOLD,
        min_speech_duration_ms=SILERO_MIN_SPEECH_MS,
        min_silence_duration_ms=SILERO_MIN_SILENCE_MS,
        speech_pad_ms=SILERO_PAD_MS,
    )
    if not ts:
        return None
    start = ts[0]["start"]; end = ts[-1]["end"]
    trimmed = buf_f32[start:end]
    if len(trimmed) < int(sr * 0.15):
        return None
    return trimmed

# ===========================
# Whisper model loading + wrapper
# ===========================
model: Optional[WhisperModel] = None
COMPUTE_TYPE = None

def _pick_default_model_key() -> str:
    # CPU → small ; CUDA → turbo (large-v3-turbo)
    return "turbo" if device == "cuda" else "small"

def _load_whisper_model(model_key: str):
    global model, COMPUTE_TYPE
    model_id = _model_id_for_key(model_key)
    chain, first = _resolve_compute_type(device)
    last_err = None
    for ct in chain:
        try:
            print(f"Loading Whisper model '{model_id}' on {device} with compute_type={ct} ...")
            model = WhisperModel(model_id, device=device, compute_type=ct)
            COMPUTE_TYPE = ct
            print(f"Whisper ready: compute_type={ct}")
            return
        except Exception as e:
            print(f"compute_type '{ct}' failed -> {e}")
            last_err = e
    raise last_err

def _warmup_model():
    try:
        dummy = np.zeros((SAMPLERATE // 2,), dtype=np.float32)
        list(fw_transcribe(dummy, beam_size=1, temperature=0.0, without_timestamps=True))
    except Exception as e:
        print("Warmup failed (non-fatal):", e)

def fw_transcribe(audio, **kwargs):
    """
    Call faster-whisper's transcribe() with only supported kwargs.
    Also inject auto threading args if available.
    """
    if CPU_THREADS is not None:
        kwargs.setdefault("cpu_threads", CPU_THREADS)
    if NUM_WORKERS is not None:
        kwargs.setdefault("num_workers", NUM_WORKERS)

    params = inspect.signature(model.transcribe).parameters
    safe_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return model.transcribe(audio, **safe_kwargs)

# ===========================
# Language helpers + outro
# ===========================
def _effective_language():
    if LANGUAGE is None:
        return None
    if isinstance(LANGUAGE, str) and LANGUAGE.strip().lower() in ("", "auto"):
        return None
    return LANGUAGE

def _norm_text_basic(s: str) -> str:
    t = (s or "").lower()
    t = t.replace("ё", "е")
    t = re.sub(r"[^a-zа-я0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def looks_like_outro(s: str) -> bool:
    t = _norm_text_basic(s)
    if any(_norm_text_basic(p) in t for p in BAN_PHRASES):
        return True
    if "продолжение следует" in t:
        return True
    if "субтитры" in t and ("сделал" in t or "создал" in t or "создавал" in t):
        return True
    t_c = t.replace(" ", "")
    if ("dimator" in t_c) or ("dimatorzhok" in t_c) or ("dimatorzok" in t_c) or ("диматоржок" in t_c):
        return True
    return False

def strip_trailing_dot(text: str) -> str:
    if text is None:
        return ""
    s = str(text).rstrip()
    while s.endswith(".") or s.endswith("…"):
        s = s[:-1]
    return s

# ===========================
# Decoding functions (faster settings)
# ===========================
def transcribe_buffer(buffer):
    if buffer.size == 0:
        return ""
    segments, _ = fw_transcribe(
        buffer,
        language=_effective_language(),
        beam_size=1,
        temperature=0.0,
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.75,
        compression_ratio_threshold=2.3,
    )
    return " ".join((seg.text or "").strip() for seg in segments).strip()

def transcribe_buffer_commit(buffer):
    if buffer is None or buffer.size == 0:
        return ""

    mono = buffer[:, 0] if getattr(buffer, "ndim", 0) > 1 else buffer
    mono = np.asarray(mono, dtype=np.float32, order="C")

    # Trim with Silero if available (behind RMS pre-gate happens inside)
    trimmed = mono
    try:
        if _has_silero:
            t = _silero_vad_trim(mono)
            if t is not None:
                trimmed = t
            else:
                if not is_silence(mono, threshold_db=-50.0):
                    trimmed = mono
                else:
                    return ""
    except Exception:
        pass

    # simple pre-emphasis for clarity
    if trimmed.shape[0] > 1:
        x = trimmed.copy()
        x[1:] = x[1:] - 0.97 * x[:-1]
        trimmed = x

    samples = trimmed.flatten()

    # First pass (fast & strict)
    segments, info = fw_transcribe(
        samples,
        language=_effective_language(),
        beam_size=1,
        temperature=0.0,
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.85,
        compression_ratio_threshold=2.1,
    )

    segs = list(segments)
    full_text = " ".join((s.text or "").strip() for s in segs).strip()

    def _seg_suspicious(seg) -> bool:
        try:
            cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
            lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
            nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            return (cr > 2.2) or (lp < -0.5 and nsp > 0.5)
        except Exception:
            return False

    suspicious = False
    if full_text and looks_like_outro(full_text):
        suspicious = True
    elif not full_text:
        suspicious = True
    else:
        bad_flags = [_seg_suspicious(sg) for sg in segs] if segs else []
        if bad_flags and (sum(bad_flags) >= max(1, len(bad_flags)//2) or bad_flags[-1]):
            suspicious = True

    # In SPEED_MODE, skip the retry pass entirely to save time
    if suspicious and not SPEED_MODE:
        # Retry with slightly relaxed thresholds
        segments2, info2 = fw_transcribe(
            samples,
            language=_effective_language(),
            beam_size=1,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.90,
            compression_ratio_threshold=2.0,
        )
        segs2 = list(segments2)
        alt = " ".join((s.text or "").strip() for s in segs2).strip()

        def _seg_suspicious_retry(seg) -> bool:
            try:
                cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
                lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
                nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                return (cr > 2.0) or (lp < -0.35 and nsp > 0.60)
            except Exception:
                return False

        bad2 = [_seg_suspicious_retry(sg) for sg in segs2] if segs2 else []
        looks_outro = bool(alt and looks_like_outro(alt))
        metrics_bad = bool(bad2 and (sum(bad2) >= max(1, len(bad2)//2) or bad2[-1]))
        if (not alt) or looks_outro or metrics_bad:
            return ""
        full_text = alt
    elif suspicious and SPEED_MODE:
        return ""

    try:
        full_text = strip_trailing_dot(full_text)
    except Exception:
        pass
    return full_text

# ===========================
# Glossary correction (full logic retained)
# ===========================
GLOSSARIES = {}
GLOSSARY_VER = 0
BEST_CACHE_SINGLE: dict[tuple[str, str, int], tuple[Optional["_Best"], Optional["_Best"]]] = {}
BEST_CACHE_MERGE: dict[tuple[str, str, int], tuple[Optional["_Best"], Optional["_Best"]]] = {}

def _clear_best_caches():
    BEST_CACHE_SINGLE.clear()
    BEST_CACHE_MERGE.clear()

def load_glossaries(path: str = GLOSSARY_FILE):
    global GLOSSARY_VER
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                norm = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        norm[k] = [str(x) for x in v]
                    elif isinstance(v, dict):
                        norm[k] = [str(x) for x in v.keys()]
                if norm:
                    GLOSSARIES.clear()
                    GLOSSARIES.update(norm)
                    GLOSSARY_VER += 1
                    _clear_best_caches()
    except Exception as e:
        print("Glossary load error:", e)

def _strip_diacritics(s: str) -> str:
    if not s:
        return ""
    n = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in n if not unicodedata.combining(ch))

def _ru_norm(s: str) -> str:
    t = (s or "").lower().replace("ё", "е")
    t = _strip_diacritics(t)
    t = re.sub(r"[^a-zа-яіїєґ]+", "", t)
    return t

def _ru_norm_merge(s: str) -> str:
    t = _ru_norm(s)
    return t.replace("тс", "ц")

CONFUSION_CLASSES = [
    set("бп"), set("вф"), set("гкх"), set("дт"), set("зсц"),
    set("жшщч"),
]
KEYBOARD_NEIGHBORS_RU: dict[str, set[str]] = {}
VOWELS_RU = set("аеёиоуыэюяіїеaeiouy")

def _is_vowel_ru(ch: str) -> bool:
    return ch in VOWELS_RU

def _same_confusion_class(a: str, b: str) -> bool:
    if a == b:
        return True
    for grp in CONFUSION_CLASSES:
        if a in grp and b in grp:
            return True
    return False

@lru_cache(maxsize=8192)
def ru_phon_key(s: str) -> str:
    s = s.replace("ё", "е")
    out = []
    for ch in s:
        if ch in "бп":
            out.append("b")
        elif ch in "вф":
            out.append("v")
        elif ch in "гкхґ":
            out.append("k")
        elif ch in "дт":
            out.append("t")
        elif ch in "жшщч":
            out.append("x")
        elif ch in "зсц":
            out.append("s")
        elif ch in "й":
            out.append("j")
        elif ch in "а":
            out.append("a")
        elif ch in "еэє":
            out.append("e")
        elif ch in "иыії":
            out.append("i")
        elif ch in "о":
            out.append("o")
        elif ch in "ую":
            out.append("u")
        else:
            out.append(ch)
    return "".join(out)

_WDL_INS_COST   = 1.0
_WDL_DEL_COST   = 1.0
_WDL_SWAP_COST  = 0.8
_WDL_CONF_COST  = 0.55
_WDL_VOWEL_COST = 0.70
_WDL_NEAR_COST  = 0.65
_WDL_BASE_COST  = 1.0

_SPECIAL_PAIRS = {
    ("й","и"): 0.45, ("и","й"): 0.45,
    ("о","а"): 0.75, ("е","и"): 0.75, ("и","е"): 0.75,
}

def _sub_cost_ru(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if (a, b) in _SPECIAL_PAIRS:
        return _SPECIAL_PAIRS[(a, b)]
    if _same_confusion_class(a, b):
        return _WDL_CONF_COST
    if (_is_vowel_ru(a) and _is_vowel_ru(b)):
        return _WDL_VOWEL_COST
    if b in KEYBOARD_NEIGHBORS_RU.get(a, ()) or a in KEYBOARD_NEIGHBORS_RU.get(b, ()):
        return _WDL_NEAR_COST
    return _WDL_BASE_COST

@lru_cache(maxsize=32768)
def _wdl(a: str, b: str) -> float:
    n, m = len(a), len(b)
    if n == 0: return float(m)
    if m == 0: return float(n)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i * _WDL_DEL_COST
    for j in range(1, m+1):
        dp[0][j] = j * _WDL_INS_COST
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            bj = b[j-1]
            sub = dp[i-1][j-1] + _sub_cost_ru(ai, bj)
            ins = dp[i][j-1] + _WDL_INS_COST
            dele= dp[i-1][j] + _WDL_DEL_COST
            best = sub if sub <= ins and sub <= dele else (ins if ins <= dele else dele)
            if i >= 2 and j >= 2 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                swap = dp[i-2][j-2] + _WDL_SWAP_COST
                if swap < best:
                    best = swap
            dp[i][j] = best
    return dp[n][m]

def _bigrams(s: str) -> set[str]:
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else set()

def _dice_sim(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return (2.0 * inter) / (len(A) + len(B))

def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _lcs_len(a: str, b: str) -> int:
    ia, ib = len(a)-1, len(b)-1
    k = 0
    while ia >= 0 and ib >= 0 and a[ia] == b[ib]:
        k += 1; ia -= 1; ib -= 1
    return k

def _lev(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = list(range(m+1))
    for i in range(1, n+1):
        cur = [i] + [0]*m
        ca = a[i-1]
        for j in range(1, m+1):
            cb = b[j-1]
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    return prev[m]

def _len_aware_max_edits(L: int) -> int:
    if L <= 4:  return 1
    if L <= 6:  return 1
    if L <= 8:  return 2
    if L <= 10: return 2
    return 3

def _min_dice_for_len(L: int) -> float:
    if L <= 4:  return 0.57
    if L <= 6:  return 0.69
    if L <= 8:  return 0.72
    if L <= 10: return 0.77
    return 0.80

_Best = namedtuple("_Best", "orig norm d dice lcp lcs wdl phon")

def _combined_score(a_norm: str, b_norm: str) -> tuple[float, float, float, int]:
    wdl  = _wdl(a_norm, b_norm)
    dice = _dice_sim(a_norm, b_norm)
    a_ph = ru_phon_key(a_norm)
    b_ph = ru_phon_key(b_norm)
    phon = _wdl(a_ph, b_ph)
    lcp  = _lcp_len(a_norm, b_norm)
    lcs  = _lcs_len(a_norm, b_norm)
    anchor = max(lcp, lcs)
    score = wdl + 0.45 * phon + (1.0 - dice) - 0.10 * min(anchor, 5)
    return score, wdl, dice, anchor

BRIDGE_WORDS = {"в", "во", "и", "й", "а"}

def _best_two_candidates(a_norm: str, g_norm: list[tuple[str, str]]) -> tuple[Optional[_Best], Optional[_Best]]:
    if not a_norm:
        return None, None
    best: Optional[_Best] = None
    second: Optional[_Best] = None
    first = a_norm[0] if a_norm else ""
    last  = a_norm[-1] if a_norm else ""
    use_coarse = len(a_norm) >= 5

    def _edge_ok(a_ch: str, b_ch: str) -> bool:
        if not a_ch or not b_ch: return False
        return a_ch == b_ch or _same_confusion_class(a_ch, b_ch)

    for orig, b in g_norm:
        if not b:
            continue
        if use_coarse and not (_edge_ok(first, b[0]) or _edge_ok(last, b[-1])):
            continue
        d_lev = _lev(a_norm, b)
        L = max(len(a_norm), len(b))
        if d_lev > _len_aware_max_edits(L):
            wdl_soft = _wdl(a_norm, b)
            if wdl_soft > _len_aware_max_edits(L) + 0.5:
                continue
        dice = _dice_sim(a_norm, b)
        lcp  = _lcp_len(a_norm, b)
        lcs  = _lcs_len(a_norm, b)
        wdl  = _wdl(a_norm, b)
        phon = _wdl(ru_phon_key(a_norm), ru_phon_key(b))
        cand = _Best(orig=orig, norm=b, d=d_lev, dice=dice, lcp=lcp, lcs=lcs, wdl=wdl, phon=phon)

        def better(x: _Best, y: _Best) -> bool:
            if x.wdl != y.wdl: return x.wdl < y.wdl
            if x.dice != y.dice: return x.dice > y.dice
            return max(x.lcp, x.lcs) > max(y.lcp, y.lcs)

        if best is None or better(cand, best):
            second = best; best = cand
        elif second is None or better(cand, second):
            second = cand
    return best, second

def _best_two_cached(header: str, a_norm: str, g_norm: list[tuple[str, str]], *, is_merge: bool) -> tuple[Optional[_Best], Optional[_Best]]:
    key = (header, a_norm, GLOSSARY_VER)
    cache = BEST_CACHE_MERGE if is_merge else BEST_CACHE_SINGLE
    if key in cache:
        return cache[key]
    res = _best_two_candidates(a_norm, g_norm)
    cache[key] = res
    return res

def _should_correct_from_norm(a_norm: str, best: Optional[_Best], runner_up: Optional[_Best]) -> bool:
    if not best or not a_norm:
        return False
    L = max(len(a_norm), len(best.norm))
    wdl  = best.wdl
    dice = best.dice
    anchor = max(best.lcp, best.lcs)
    phon = best.phon

    max_ed = _len_aware_max_edits(L) + 0.5
    min_d  = _min_dice_for_len(L) - 0.03
    req_anchor = 2 if L <= 5 else 3

    if dice < min_d and phon <= 1.0:
        min_d -= 0.07
    if wdl > max_ed: return False
    if dice < min_d: return False
    if anchor < req_anchor: return False

    if runner_up is None:
        return True

    score_best, _, _, _   = _combined_score(a_norm, best.norm)
    score_run , _, _, _   = _combined_score(a_norm, runner_up.norm)
    return (score_run - score_best) >= 0.35

def _should_correct(token: str, best: Optional[_Best], runner_up: Optional[_Best]) -> bool:
    a_norm = _ru_norm(token)
    return _should_correct_from_norm(a_norm, best, runner_up)

def correct_text_for_column(text: str, header: str) -> str:
    if not text:
        return ""
    if header not in NAME_COLUMNS:
        return text

    glossary = GLOSSARIES.get(header) or []
    if not glossary:
        return text

    g_norm = [(g, _ru_norm(g)) for g in glossary if g and _ru_norm(g)]
    g_norm_merge = [(g, _ru_norm_merge(g)) for g in glossary if g and _ru_norm_merge(g)]
    if not g_norm:
        return text

    tokens = text.split()
    out: List[str] = []
    i = 0

    def _score_from_norm(a_norm: str, cand: _Best) -> float:
        sc, _, _, _ = _combined_score(a_norm, cand.norm)
        return sc

    while i < len(tokens):
        tok = tokens[i]
        a = _ru_norm(tok)

        cand2 = None
        cand3 = None

        if i + 1 < len(tokens):
            tok2 = tokens[i + 1]
            joined2 = tok + tok2
            a_join2 = _ru_norm_merge(joined2)
            if a_join2:
                bestm2, runnerm2 = _best_two_cached(header, a_join2, g_norm_merge, is_merge=True)
                if _should_correct_from_norm(a_join2, bestm2, runnerm2):
                    if len(a_join2) >= 5 and (bestm2.lcp >= 3 or bestm2.lcs >= 3):
                        cand2 = (bestm2, _score_from_norm(a_join2, bestm2))

        if i + 2 < len(tokens):
            tok2 = tokens[i + 1]
            tok3 = tokens[i + 2]
            joined123 = tok + tok2 + tok3
            a_join123 = _ru_norm_merge(joined123)
            if a_join123:
                bestm3a, runnerm3a = _best_two_cached(header, a_join123, g_norm_merge, is_merge=True)
                if _should_correct_from_norm(a_join123, bestm3a, runnerm3a):
                    if len(a_join123) >= 6 and (bestm3a.lcp >= 3 or bestm3a.lcs >= 3):
                        cand3 = (bestm3a, _score_from_norm(a_join123, bestm3a))
            tok2_clean = _ru_norm(tok2)
            if tok2_clean in {"в","во","и","й","а"}:
                joined13 = tok + tok3
                a_join13 = _ru_norm_merge(joined13)
                if a_join13:
                    bestm3b, runnerm3b = _best_two_cached(header, a_join13, g_norm_merge, is_merge=True)
                    if _should_correct_from_norm(a_join13, bestm3b, runnerm3b):
                        if len(a_join13) >= 6 and (bestm3b.lcp >= 3 or bestm3b.lcs >= 3):
                            cand3 = (bestm3b, _score_from_norm(a_join13, bestm3b))

        best_merge = cand2 if (cand2 and (not cand3 or cand2[1] <= cand3[1])) else cand3
        if best_merge is not None:
            ok1 = ok2 = False
            score1 = score2 = float("inf")
            if a:
                best1, runner1 = _best_two_cached(header, a, g_norm, is_merge=False)
                ok1 = _should_correct(tok, best1, runner1)
                if ok1 and best1:
                    score1 = _score_from_norm(a, best1)
            if i + 1 < len(tokens):
                a2 = _ru_norm(tokens[i + 1])
                if a2:
                    best2, runner2 = _best_two_cached(header, a2, g_norm, is_merge=False)
                    ok2 = _should_correct(tokens[i + 1], best2, runner2)
                    if ok2 and best2:
                        score2 = _score_from_norm(a2, best2)
            score_m = best_merge[1]
            if (not ok1 and not ok2) or (score_m + 0.30 <= min(score1, score2)):
                out.append(best_merge[0].orig)
                i += 3 if (cand3 and best_merge == cand3) else 2
                continue

        if a:
            best, runner = _best_two_cached(header, a, g_norm, is_merge=False)
            if _should_correct(tok, best, runner):
                out.append(best.orig)
            else:
                out.append(tok)
        else:
            out.append(tok)
        i += 1

    return " ".join(out)

# Dates
RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalize_date(text: str) -> Optional[str]:
    t = (_clean_spaces(text or "")).lower()
    m = re.search(r"\b(\d{1,2})[.\-\/](\d{1,2})[.\-\/](\d{2,4})\b", t)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = 2000 + y if y < 100 else y
        try:
            return f"{date(y, mth, d):%d.%m.%Y}"
        except ValueError:
            return None
    m2 = re.search(r"\b(\d{1,2})\s+([а-яё]+)(?:\s+(\d{2,4}))?\b", t)
    if m2:
        mon = m2.group(2)
        if mon in RU_MONTHS:
            d = int(m2.group(1)); mth = RU_MONTHS[mon]
            y = m2.group(3)
            if y is None:
                y = date.today().year
            else:
                y = int(y); y = 2000 + y if y < 100 else y
            try:
                return f"{date(y, mth, d):%d.%m.%Y}"
            except ValueError:
                return None
    return None

# ===========================
# UI theming
# ===========================
try:
    from tkinter import ttk
    import tkinter.font as tkfont
except Exception:
    ttk = None
    tkfont = None

def _gfm_palette(dark=True):
    if dark:
        return {
            "bg": "#0f1216",
            "surface": "#151a21",
            "elevated": "#1b212a",
            "text": "#e6e6e6",
            "muted": "#a8b2bd",
            "accent": "#479d7b",
            "grid": "#2b323c",
            "border": "#2b323c",
            "entry_bg": "#0f1216",
            "entry_fg": "#e6e6e6",
            "sel_bg": "#263a33",
            "sel_fg": "#eaf5f0",
            "button_bg": "#222833",
            "button_active_bg": "#2a3140",
            "button_fg": "#eef1f4",
            "button_border": "#333a46",
            "button_disabled_bg": "#1f2430",
            "button_disabled_fg": "#717b89",
            "header_bg": "#151a21",
            "header_fg": "#d9dee5",
            "index_bg": "#151a21",
            "index_fg": "#cbd3dc",
        }
    else:
        return {
            "bg": "#ffffff",
            "surface": "#f5f7f9",
            "elevated": "#ffffff",
            "text": "#0f141a",
            "muted": "#5b6876",
            "accent": "#479d7b",
            "grid": "#e3e8ef",
            "border": "#d8dee6",
            "entry_bg": "#ffffff",
            "entry_fg": "#0f141a",
            "sel_bg": "#d8efe6",
            "sel_fg": "#0f141a",
            "button_bg": "#e9edf2",
            "button_active_bg": "#dfe6ee",
            "button_fg": "#0f141a",
            "button_border": "#cbd3dc",
            "header_bg": "#f6f8fa",
            "header_fg": "#0f141a",
            "index_bg": "#f6f8fa",
            "index_fg": "#334155",
        }

def _set_windows_dpi_awareness():
    try:
        import ctypes
        if sys.platform.startswith("win"):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
    except Exception:
        pass

def _auto_scaling(root: "tk.Tk"):
    try:
        dpi = root.winfo_fpixels('1i')
        scaling = max(1.0, float(dpi) / 72.0)
        scaling = min(2.5, scaling * UI_SCALE)
        root.tk.call('tk', 'scaling', scaling)
    except Exception:
        pass

def _apply_fonts(root: "tk.Tk"):
    if not tkfont:
        return
    try:
        default = tkfont.nametofont("TkDefaultFont")
        text_f = tkfont.nametofont("TkTextFont")
        heading = tkfont.nametofont("TkHeadingFont")
        fixed = tkfont.nametofont("TkFixedFont")
    except Exception:
        return
    base = max(10, int(round(11 * UI_SCALE)))
    hsize = max(11, int(round(12 * UI_SCALE)))
    mono  = max(10, int(round(11 * UI_SCALE)))
    fam  = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"
    hfam = "Segoe UI Semibold" if sys.platform.startswith("win") else "Helvetica"
    mfam = "Consolas" if sys.platform.startswith("win") else "Menlo"
    for fnt in (default, text_f):
        fnt.configure(size=base, family=fam)
    heading.configure(size=hsize, weight="bold", family=hfam)
    fixed.configure(size=mono, family=mfam)

def _recolor_tk_widgets(root: "tk.Tk", dark=True):
    colors = _gfm_palette(dark)
    def walk(w):
        if isinstance(w, (tk.Tk, tk.Toplevel, tk.Frame, tk.LabelFrame, tk.PanedWindow)):
            try: w.configure(bg=colors["bg"])
            except Exception: pass
        if isinstance(w, tk.Label):
            try: w.configure(bg=colors["bg"], fg=colors["text"])
            except Exception: pass
        if isinstance(w, tk.Entry):
            try:
                w.configure(
                    bg=colors["surface"],
                    fg=colors["entry_fg"],
                    insertbackground=colors["text"],
                    relief="flat",
                    highlightthickness=1,
                    highlightbackground=colors["border"],
                    highlightcolor=colors["border"],
                )
            except Exception:
                pass
        if isinstance(w, tk.Text):
            try: w.configure(bg=colors["surface"], fg=colors["text"], insertbackground=colors["text"], highlightthickness=0, relief="flat")
            except Exception: pass
        if isinstance(w, tk.Button):
            try:
                w.configure(bg=colors["button_bg"], fg=colors["button_fg"],
                            activebackground=colors["button_active_bg"], activeforeground=colors["button_fg"],
                            relief="flat", highlightthickness=1,
                            highlightbackground=colors["button_border"], highlightcolor=colors["button_border"])
            except Exception: pass
        for c in w.winfo_children():
            walk(c)
    walk(root)

    def _enforce_disabled():
        try:
            def deep_children(w):
                for c in w.winfo_children():
                    yield c
                    yield from deep_children(c)
            for btn in deep_children(root):
                if isinstance(btn, tk.Button):
                    state = str(btn.cget("state"))
                    if state == "disabled":
                        try:
                            btn.configure(bg=colors["button_disabled_bg"], fg=colors["button_disabled_fg"],
                                          activebackground=colors["button_disabled_bg"], activeforeground=colors["button_disabled_fg"])
                        except Exception: pass
        except Exception:
            pass
        finally:
            try: root.after(300, _enforce_disabled)
            except Exception: pass
    _enforce_disabled()

def _style_ttk(dark=True):
    if not ttk:
        return
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    c = _gfm_palette(dark)
    style.configure(".", background=c["bg"], foreground=c["text"])
    style.configure("TFrame", background=c["bg"])
    style.configure("TLabel", background=c["bg"], foreground=c["text"])
    style.configure("TEntry", fieldbackground=c["surface"], foreground=c["entry_fg"], bordercolor=c["border"])
    style.map("TEntry", fieldbackground=[("disabled", c["button_disabled_bg"])], foreground=[("disabled", c["button_disabled_fg"])])
    style.configure(
        "Settings.TCombobox",
        background=c["surface"],
        fieldbackground=c["surface"],
        foreground=c["entry_fg"],
        bordercolor=c["border"],
        lightcolor=c["border"],
        darkcolor=c["border"],
    )
    style.map(
        "Settings.TCombobox",
        fieldbackground=[
            ("readonly", c["surface"]),
            ("!disabled", c["surface"]),
        ],
        foreground=[("readonly", c["entry_fg"])],
        selectbackground=[("readonly", c["sel_bg"])],
        selectforeground=[("readonly", c["sel_fg"])],
    )

def _try_style_tksheet(root: "tk.Tk", dark=True):
    try:
        import tksheet
    except Exception:
        return
    colors = _gfm_palette(dark)
    def _sz(px):
        return max(8, int(round(px * UI_SCALE)))
    def walk(w):
        for c in w.winfo_children():
            try:
                if isinstance(c, tksheet.Sheet):
                    c.set_options(
                        table_bg=colors["surface"],
                        table_grid_fg=colors["grid"],
                        table_fg=colors["text"],
                        index_bg=colors["index_bg"],
                        index_fg=colors["index_fg"],
                        header_bg=colors["header_bg"],
                        header_fg=colors["header_fg"],
                        header_border_fg=colors["grid"],
                        index_border_fg=colors["grid"],
                        top_left_bg=colors["surface"],
                        top_left_fg=colors["muted"],
                        selected_cells_bg=colors["sel_bg"],
                        selected_cells_fg=colors["sel_fg"],
                        row_height=_sz(24),
                        header_height=_sz(26),
                        row_index_width=_sz(60),
                    )
                else:
                    walk(c)
            except Exception:
                walk(c)
    walk(root)

def _apply_dark_ui(root: "tk.Tk", dark=True):
    _set_windows_dpi_awareness()
    _auto_scaling(root)
    _apply_fonts(root)
    _style_ttk(dark=dark)
    _recolor_tk_widgets(root, dark=dark)
    _try_style_tksheet(root, dark=dark)

def enable_crisp_dark_mode(root: "tk.Tk", dark=True, delay_ms=350):
    try:
        root.after(delay_ms, lambda: _apply_dark_ui(root, dark=dark))
    except Exception:
        _apply_dark_ui(root, dark=dark)

# ===========================
# GUI Application
# ===========================
class SpeechSheetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GeneoGraph VoIx")
        self.is_listening = False

        self._autosave_job = None
        self._last_autosave_ok = True
        self._start_autosave()

        self._bind_toggle_hotkeys()
        self._last_preview_text = ""
        self._last_preview_ts = 0.0
        self._last_preview_decode_ts = 0.0
        self._settings_applying = False

        # Buttons frame
        btn_frame = tk.Frame(root)
        btn_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.start_btn = tk.Button(btn_frame, text="🎤 Start Listening", command=self.start_listening)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="⛔ Stop", command=self.stop_listening, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.export_btn = tk.Button(btn_frame, text="💾 Export", command=self.export_data)
        self.export_btn.grid(row=0, column=2, padx=5)

        self.add_row_btn = tk.Button(btn_frame, text="➕ Add Rows", command=lambda: self.add_rows(100))
        self.add_row_btn.grid(row=0, column=3, padx=5)

        self.clear_btn = tk.Button(btn_frame, text="🧹 Clear All", command=self.clear_all_cells)
        self.clear_btn.grid(row=0, column=4, padx=5)

        self.glossary_btn = tk.Button(btn_frame, text="📚 Glossary", command=self.open_glossary_editor)
        self.glossary_btn.grid(row=0, column=5, padx=5)

        self.auto_num_btn = tk.Button(btn_frame, text="🔢 Auto Numerate", command=self.auto_numerate)
        self.auto_num_btn.grid(row=0, column=6, padx=5)

        self.settings_btn = tk.Button(btn_frame, text="⚙️ Settings", command=self.open_settings)
        self.settings_btn.grid(row=0, column=7, padx=5)

        # Preview label
        self.preview_label = tk.Label(
            root,
            text="Preview:",
            anchor="w",
            fg="#e6e6e6",
            font=("Calibri", max(12, int(round(16 * UI_SCALE))), "bold")
        )
        self.preview_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self._preview_clear_after = None

        # Sheet
        self.headers = ["№","№ М","№ Ж","Дата","Имя","Фамилия","Имя отца",
                        "Имя матери","Восприемник","Страница","Комментарий"]
        load_glossaries()

        self.sheet = Sheet(root, headers=self.headers, height=400, width=1000, zoom=TABLE_ZOOM_PCT)

        # Column widths
        try:
            self.load_column_widths()
        except Exception as e:
            print("initial load_column_widths error:", e)

        self.sheet.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.sheet.enable_bindings(("single_select","row_select","column_select","arrowkeys","edit_cell",
                                    "rc_popup_menu","drag_select","column_width_resize","row_height_resize",
                                    "copy","cut","paste","delete","undo","double_click_column_resize",
                                    "double_click_row_resize","rc_insert_column","rc_delete_column",
                                    "rc_insert_row","rc_delete_row"))

        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        self.root.after(60, self.load_column_widths)
        self.root.after(500, self.load_column_widths)

        self._style_action_buttons()
        self.root.after(600, self._style_action_buttons)

        # Data
        self.load_data()

        # Audio state
        self.buffer = np.zeros((0,1), dtype=np.float32)
        self._tail_chunks: deque[np.ndarray] = deque()
        self._tail_total_samples: int = 0
        self._commit_chunks: list[np.ndarray] = []
        self._commit_total_samples: int = 0

        # Safety net timer
        self._silence_guard_job = None
        self._start_silence_guard()

        # Handle close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Info line
        try:
            print(f"[Devices] Whisper device={device}; compute_type={COMPUTE_TYPE}; "
                  f"Silero device={_silero_device}; torch.cuda={torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print("[CUDA] device name:", torch.cuda.get_device_name(0))
        except Exception as _e:
            print("Device log error:", _e)

    # ------- Preview flash -------
    def _flash_preview_note(self, msg: str, ms: int = 1200):
        try:
            old = self.preview_label.cget("text")
            self.preview_label.config(text=msg)
            def _restore():
                try:
                    self.preview_label.config(text=old)
                except Exception:
                    pass
            self.root.after(ms, _restore)
        except Exception:
            pass

    # Column widths helpers
    def _snapshot_column_widths(self):
        try:
            if hasattr(self.sheet, "get_column_widths"):
                return self.sheet.get_column_widths()
            elif hasattr(self.sheet, "column_widths"):
                return list(self.sheet.column_widths)
        except Exception as e:
            print("snapshot widths error:", e)
        return None

    def apply_column_widths(self, widths):
        if widths is None:
            return
        try:
            if isinstance(widths, dict):
                iterable = widths.items()
            elif isinstance(widths, list):
                iterable = enumerate(widths)
            else:
                print("Unknown width data format:", type(widths))
                return
            for col, width in iterable:
                try:
                    self.sheet.column_width(int(col), int(width))
                except Exception as e:
                    print(f"column_width set failed for {col} -> {width}:", e)
        except Exception as e:
            print("apply_column_widths error:", e)

    @contextmanager
    def preserve_column_widths(self):
        widths = self._snapshot_column_widths()
        try:
            yield
        finally:
            if widths is not None:
                try:
                    self.apply_column_widths(widths)
                except Exception as e:
                    print("restore widths error:", e)

    # Sheet ops
    def _get_sheet_data_copy(self):
        try:
            return self.sheet.get_sheet_data(return_copy=True)
        except TypeError:
            data = self.sheet.get_sheet_data()
            return [list(row) for row in data]

    def add_rows(self, count: int = 1):
        if count <= 0:
            return
        with self.preserve_column_widths():
            data = self._get_sheet_data_copy()
            ncols = len(self.headers)
            data.extend([[""] * ncols for _ in range(count)])
            self.sheet.set_sheet_data(data)

    def clear_all_cells(self):
        with self.preserve_column_widths():
            rows = len(self.sheet.get_sheet_data())
            self.sheet.set_sheet_data([[""] * len(self.headers) for _ in range(rows)])

    def _persist_column_widths_to_settings(self):
        try:
            if hasattr(self.sheet, "get_column_widths"):
                widths = self.sheet.get_column_widths()
            elif hasattr(self.sheet, "column_widths"):
                widths = list(self.sheet.column_widths)
            else:
                return
        except Exception:
            return

        try:
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    s = json.load(f)
            except Exception:
                s = {}
            s["column_widths"] = widths
            tmp = os.path.join(os.path.dirname(SETTINGS_FILE) or ".", "~settings.tmp.json")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
            os.replace(tmp, SETTINGS_FILE)
        except Exception as e:
            print("persist widths error:", e)

    def reset_column_widths(self):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            s = {}
        if "column_widths" in s:
            s.pop("column_widths", None)
            try:
                tmp = os.path.join(os.path.dirname(SETTINGS_FILE) or ".", "~settings.tmp.json")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(s, f, ensure_ascii=False, indent=2)
                os.replace(tmp, SETTINGS_FILE)
            except Exception as e:
                print("reset widths persist error:", e)

        data = self._get_sheet_data_copy()
        try:
            sel = list(self.sheet.get_selected_cells())[:1]
        except Exception:
            sel = []

        try:
            self.sheet.destroy()
        except Exception:
            pass

        self.sheet = Sheet(self.root, headers=self.headers, height=400, width=1000, zoom=TABLE_ZOOM_PCT)
        self.sheet.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.sheet.enable_bindings((
            "single_select","row_select","column_select","arrowkeys","edit_cell",
            "rc_popup_menu","drag_select","column_width_resize","row_height_resize",
            "copy","cut","paste","delete","undo",
            "double_click_column_resize","double_click_row_resize",
            "rc_insert_column","rc_delete_column","rc_insert_row","rc_delete_row"
        ))
        self.sheet.set_sheet_data(data)

        try:
            _try_style_tksheet(self.root, dark=True)
        except Exception:
            pass

        if sel:
            r, c = sel[0]
            try:
                self.sheet.select_cell(r, c)
            except Exception:
                pass

        try:
            self.root.bell()
        except Exception:
            pass

    def add_text_to_cell(self, text):
        selected = self.sheet.get_selected_cells()
        if not selected:
            return
        row, col = list(selected)[0]
        header = self.headers[col]

        raw = (text or "").strip()
        if header == "Дата":
            value = normalize_date(raw)
            if value is None:
                try: self.root.bell()
                except Exception: pass
                self.preview_label.config(text="🎤 Preview: (дата не распознана — повторите)")
                return
            self.sheet.set_cell_data(row, col, value)
            return

        raw = correct_text_for_column(raw, header)
        if header in NAME_COLUMNS:
            raw = clean_person_field(raw)

        self.sheet.set_cell_data(row, col, raw)

    def open_glossary_editor(self):
        win = GlossaryEditor(self.root, GLOSSARY_FILE, on_saved=lambda: load_glossaries())
        try:
            enable_crisp_dark_mode(win, dark=True, delay_ms=0)
        except Exception:
            pass

    # Autosave
    def _start_autosave(self):
        try:
            if self._autosave_job:
                self.root.after_cancel(self._autosave_job)
        except Exception:
            pass
        self._autosave_job = self.root.after(AUTOSAVE_EVERY_MS, self._autosave_tick)

    def _autosave_tick(self):
        try:
            self.save_data()
            self._last_autosave_ok = True
        except Exception as e:
            self._last_autosave_ok = False
            print("Autosave error:", e)
        finally:
            self._start_autosave()

    def _cancel_autosave(self):
        try:
            if self._autosave_job:
                self.root.after_cancel(self._autosave_job)
                self._autosave_job = None
        except Exception:
            pass

    # Name helpers (for auto-numerate)
    def _ru_norm_name(self, s: str) -> str:
        if not s:
            return ""
        s = s.replace("ё", "е").replace("Ё", "Е")
        s = re.sub(r"[^А-Яа-я\- ]+", "", s).strip()
        return s

    def _first_name_token(self, name: str) -> str:
        n = self._ru_norm_name(name)
        if not n:
            return ""
        first = n.split()[0]
        return first.split("-")[-1]

    def _gender_of_name(self, name: str) -> str:
        tok = self._first_name_token(name)
        if not tok:
            return ""
        low = tok.lower()
        if low in MALE_EXCEPTIONS:
            return "м"
        last = low[-1]
        if last in ("а", "я"):
            return "ж"
        return "м"

    def auto_numerate(self):
        try:
            idx_m = self.headers.index("№ М")
            idx_f = self.headers.index("№ Ж")
            idx_name = self.headers.index("Имя")
        except ValueError:
            messagebox.showerror("Error", "Expected headers '№ М', '№ Ж', and 'Имя' were not found.")
            return

        with self.preserve_column_widths():
            rows_count = len(self.sheet.get_sheet_data())

            for r in range(rows_count):
                try:
                    self.sheet.set_cell_data(r, idx_m, "")
                    self.sheet.set_cell_data(r, idx_f, "")
                except Exception:
                    pass

            male_counter = 0
            female_counter = 0
            for r in range(rows_count):
                try:
                    name_val = self.sheet.get_cell_data(r, idx_name)
                except Exception:
                    name_val = ""

                g = self._gender_of_name(name_val)
                if g == "м":
                    male_counter += 1
                    try: self.sheet.set_cell_data(r, idx_m, str(male_counter))
                    except Exception: pass
                elif g == "ж":
                    female_counter += 1
                    try: self.sheet.set_cell_data(r, idx_f, str(female_counter))
                    except Exception: pass

        try: self.root.bell()
        except Exception: pass

    # Hotkeys
    def _bind_toggle_hotkeys(self):
        self._hotkey_cooldown_until = 0.0
        def _cooldown_ok():
            now = time.monotonic()
            if now < self._hotkey_cooldown_until:
                return False
            self._hotkey_cooldown_until = now + 0.25
            return True
        def _in_text_edit():
            w = self.root.focus_get()
            if w is None:
                return False
            try:
                cls = (w.winfo_class() or "").lower()
            except Exception:
                cls = ""
            return isinstance(w, (tk.Entry, tk.Text)) or "entry" in cls or "text" in cls
        def toggle_evt(_=None):
            if _in_text_edit():
                self._flash_preview_note("⚠ Горячая клавиша отключена во время редактирования ячейки")
                try: self.root.bell()
                except Exception: pass
                return
            if not _cooldown_ok():
                return
            if self.is_listening:
                self.stop_listening()
            else:
                self.start_listening()
        def stop_evt(_=None):
            if not _cooldown_ok():
                return
            if self.is_listening:
                self.stop_listening()
        self.root.bind_all("<F1>", toggle_evt, add="+")
        for seq in ("<Escape>", "<KeyPress-Escape>"):
            self.root.bind_all(seq, stop_evt, add="+")

    def _current_settings(self) -> dict:
        return {
            "language": LANGUAGE if LANGUAGE else "auto",
            "autosave_minutes": max(1, AUTOSAVE_EVERY_MS // 60000),
            "ui_scale": UI_SCALE,
            "model_key": MODEL_KEY or _pick_default_model_key(),
            "speed_mode": SPEED_MODE,
        }

    def _refresh_preview_font(self):
        try:
            sz = max(12, int(round(16 * UI_SCALE)))
            self.preview_label.configure(font=("Calibri", sz, "bold"))
        except Exception:
            pass

    def _maybe_update_sheet_zoom(self):
        try:
            new_zoom = int(round(TABLE_ZOOM_PCT * float(UI_SCALE)))
            if hasattr(self.sheet, "set_sheet_zoom"):
                self.sheet.set_sheet_zoom(new_zoom)
            elif hasattr(self.sheet, "set_zoom"):
                self.sheet.set_zoom(new_zoom)
            elif hasattr(self.sheet, "zoom"):
                try:
                    self.sheet.zoom = new_zoom
                    if hasattr(self.sheet, "refresh"):
                        self.sheet.refresh()
                except Exception:
                    pass
        except Exception:
            pass

    def apply_settings(self, s: dict):
        prev = self._current_settings()
        self._settings_applying = True
        try:
            # Determine if model switch is requested
            requested_model_key = s.get("model_key", prev["model_key"])
            model_changed = (requested_model_key != prev["model_key"])

            # Save globals
            _apply_settings_to_globals(s)
            _save_settings_file(self._current_settings())

            # UI scale changes
            if float(prev.get("ui_scale", 1.0)) != float(s.get("ui_scale", prev.get("ui_scale", 1.0))):
                _apply_dark_ui(self.root, dark=True)
                self._refresh_preview_font()
                self._maybe_update_sheet_zoom()
                self._style_action_buttons()

            # Speed mode might change preview tail; no reload needed

            # If listening and model changed, stop recording before reload
            if model_changed:
                if self.is_listening:
                    self.stop_listening()
                self._show_loading("Loading model, please wait…")
                try:
                    _load_whisper_model(MODEL_KEY)
                    _warmup_model()
                finally:
                    self._hide_loading()

            self._start_autosave()
        except Exception as e:
            print("apply_settings error:", e)
            messagebox.showerror("Settings", f"Failed to apply settings:\n{e}")
        finally:
            self._settings_applying = False

    # Settings dialog
    def open_settings(self):
        SettingsDialog(
            self.root,
            initial=self._current_settings(),
            on_apply=self.apply_settings,
            on_reset_widths=self.reset_column_widths,
        )

    # Listening
    def start_listening(self):
        global LAST_ACTIVITY_TS
        self.reset_audio_state()
        LAST_ACTIVITY_TS = time.monotonic()
        self._listening_started_ts = LAST_ACTIVITY_TS

        # reset rolling structures
        self._tail_chunks.clear()
        self._tail_total_samples = 0
        self._commit_chunks.clear()
        self._commit_total_samples = 0
        self._last_preview_decode_ts = 0.0

        self.is_listening = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._style_action_buttons()
        threading.Thread(target=self.audio_capture_thread, daemon=True).start()
        threading.Thread(target=self.transcribe_thread, daemon=True).start()

    def stop_listening(self):
        self.is_listening = False
        self.reset_audio_state()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._style_action_buttons()

    def audio_capture_thread(self):
        bs = int(SAMPLERATE * AUDIO_BLOCK_SEC)
        if bs < 256:
            bs = 256
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='float32',
                            blocksize=bs, latency='low', callback=audio_callback):
            while self.is_listening:
                sd.sleep(100)

    def _tail_limit_samples(self) -> int:
        return int(_preview_tail_sec() * SAMPLERATE)

    def _push_tail(self, mono_1d: np.ndarray):
        """Append to rolling tail (deque of 1D float32 chunks) and enforce length."""
        if mono_1d.size == 0:
            return
        self._tail_chunks.append(mono_1d)
        self._tail_total_samples += mono_1d.shape[0]
        limit = self._tail_limit_samples()
        # trim from the left if we exceed the limit
        while self._tail_total_samples > limit and self._tail_chunks:
            excess = self._tail_total_samples - limit
            left = self._tail_chunks[0]
            if left.shape[0] <= excess:
                self._tail_total_samples -= left.shape[0]
                self._tail_chunks.popleft()
            else:
                # split the left chunk, drop only the needed prefix
                remain = left[excess:].copy()
                self._tail_chunks[0] = remain
                self._tail_total_samples -= excess
                break

    def _concat_tail(self) -> np.ndarray:
        if not self._tail_chunks:
            return np.zeros((0,), dtype=np.float32)
        if len(self._tail_chunks) == 1:
            return self._tail_chunks[0]
        return np.concatenate(list(self._tail_chunks), axis=0)

    def transcribe_thread(self):
        last_audio_time = time.monotonic()

        while self.is_listening:
            if getattr(self, "_settings_applying", False):
                time.sleep(0.03)
                continue

            # Drain queue quickly
            got_any = False
            while not audio_queue.empty():
                blk = audio_queue.get()
                got_any = True
                # Add to commit accumulator (keep original 2D)
                self._commit_chunks.append(blk)
                self._commit_total_samples += len(blk)

                # Add to tail as 1D (mono)
                mono = blk[:, 0] if getattr(blk, "ndim", 0) > 1 else blk
                mono = np.asarray(mono, dtype=np.float32, order="C")
                self._push_tail(mono)

            if got_any:
                last_audio_time = time.monotonic()

            # Preview gating: only if enough tail and time since last preview
            now = time.monotonic()
            tail_needed = int(_preview_tail_sec() * SAMPLERATE * 0.6)  # need at least ~60% of window
            if self._tail_total_samples >= tail_needed and (now - self._last_preview_decode_ts) >= PREVIEW_MIN_INTERVAL_SEC:
                tail = self._concat_tail()
                # quick silence gate
                if not is_silence(tail, threshold_db=-48.0):
                    preview_text = transcribe_buffer(tail)
                    if preview_text and _preview_is_banned(preview_text):
                        preview_text = ""
                    if preview_text:
                        preview_text = strip_trailing_dot(preview_text)
                        if preview_text != self._last_preview_text:
                            self._last_preview_text = preview_text
                            self._last_preview_ts = now
                            self.root.after(0, lambda t=preview_text: self.preview_label.config(text="Preview: " + t))
                self._last_preview_decode_ts = now

            # Commit conditions
            timeout = self._commit_total_samples >= int(BLOCK_DURATION * SAMPLERATE)

            # Tail silence?
            tail = self._concat_tail() if self._tail_total_samples > 0 else None
            tail_is_silence = bool(tail is not None and tail.shape[0] >= int(0.5 * SAMPLERATE) and is_silence(tail))

            # Safety net: stop after 60s of inactivity
            if (time.monotonic() - last_audio_time) > 60.0:
                self.root.after(0, lambda: self.preview_label.config(text="Preview: (auto-stopped after inactivity)"))
                self.stop_listening()
                return

            if tail_is_silence or timeout:
                # Concatenate commit window only now
                if not self._commit_chunks:
                    # nothing to commit; just clear tail
                    self._tail_chunks.clear()
                    self._tail_total_samples = 0
                    time.sleep(0.03)
                    continue

                buf = np.concatenate(self._commit_chunks, axis=0)
                full_text = transcribe_buffer_commit(buf)
                full_text = strip_trailing_dot(full_text)

                if not full_text:
                    # fallback to tail-only decode if recent preview was good
                    recent_preview = (time.monotonic() - getattr(self, "_last_preview_ts", 0.0)) < 3.0
                    if recent_preview and getattr(self, "_last_preview_text", ""):
                        try:
                            alt = transcribe_buffer(tail)
                            alt = strip_trailing_dot(alt)
                            if alt and not looks_like_outro(alt):
                                full_text = alt
                        except Exception:
                            pass

                if full_text and not looks_like_outro(full_text):
                    self.root.after(0, self.add_text_to_cell, full_text)
                    # update preview label only if changed
                    if full_text != self._last_preview_text:
                        self._last_preview_text = full_text
                        self._last_preview_ts = time.monotonic()
                        self.root.after(0, lambda t=full_text: self.preview_label.config(text="Preview: " + t))

                # schedule preview clear
                try:
                    if getattr(self, "_preview_clear_after", None):
                        self.root.after_cancel(self._preview_clear_after)
                except Exception:
                    pass
                def _clear_preview():
                    try: self.preview_label.config(text="Preview:")
                    except Exception: pass
                    finally: setattr(self, "_preview_clear_after", None)
                self._preview_clear_after = self.root.after(PREVIEW_CLEAR_DELAY_MS, _clear_preview)

                # reset commit + tail efficiently
                self._commit_chunks.clear()
                self._commit_total_samples = 0
                self._tail_chunks.clear()
                self._tail_total_samples = 0

            time.sleep(0.02)  # tight loop for low latency, still cooperative

    def reset_audio_state(self):
        global audio_queue
        try:
            while not audio_queue.empty():
                audio_queue.get_nowait()
        except Exception:
            pass
        self.buffer = np.zeros((0,1), dtype=np.float32)
        self._tail_chunks.clear()
        self._tail_total_samples = 0
        self._commit_chunks.clear()
        self._commit_total_samples = 0
        try:
            self.preview_label.config(text="Preview:")
        except Exception:
            pass

    # Silence-guard timer (ensure we keep counting even if not listening)
    def _start_silence_guard(self):
        def tick():
            if self.is_listening:
                now = time.monotonic()
                started = getattr(self, "_listening_started_ts", now)
                if (now - started) >= SILENCE_GUARD_GRACE_SEC and (now - LAST_ACTIVITY_TS) > 60.0:
                    self.stop_listening()
                    self._flash_preview_note("⏸ Авто-стоп: нет входа 60с")
            self._silence_guard_job = self.root.after(1000, tick)
        self._silence_guard_job = self.root.after(1000, tick)

    # Console detach (Windows)
    def _detach_console_if_any(self):
        if sys.platform.startswith("win"):
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                if kernel32.GetConsoleWindow():
                    kernel32.FreeConsole()
            except Exception:
                pass

    # Styling
    def _style_action_buttons(self):
        START_BG  = "#2e7d32"
        START_BG_H= "#2b7030"
        STOP_BG   = "#c62828"
        STOP_BG_H = "#b12525"
        TXT_LIGHT = "#ffffff"
        def apply(btn, *, bg, bg_h, fg, disabled: bool):
            if disabled:
                btn.configure(
                    bg="#3a4352", fg="#9aa5b1",
                    activebackground="#3a4352", activeforeground="#9aa5b1"
                )
            else:
                btn.configure(
                    bg=bg, fg=fg,
                    activebackground=bg_h, activeforeground=fg,
                    highlightthickness=1
                )
        try:
            apply(self.start_btn, bg=START_BG, bg_h=START_BG_H, fg=TXT_LIGHT,
                  disabled=(str(self.start_btn.cget("state")) == "disabled"))
            apply(self.stop_btn,  bg=STOP_BG,  bg_h=STOP_BG_H,  fg=TXT_LIGHT,
                  disabled=(str(self.stop_btn.cget("state")) == "disabled"))
        except Exception:
            pass

    # Export / save / load
    def export_data(self):
        data = self.sheet.get_sheet_data()
        df = pd.DataFrame(data, columns=self.headers)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")]
        )
        if not file_path:
            return
        if file_path.endswith(".csv"):
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Exported", f"Data exported to:\n{file_path}")
            return
        try:
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Exported", f"Data exported to:\n{file_path}")
        except Exception as e:
            try:
                alt_csv = re.sub(r"\.xlsx$", ".csv", file_path, flags=re.I)
                df.to_csv(alt_csv, index=False, encoding="utf-8-sig")
                messagebox.showwarning(
                    "Exported as CSV",
                    f"Excel export requires 'openpyxl' or 'xlsxwriter' which may not be bundled.\n"
                    f"Saved CSV instead:\n{alt_csv}\n\nDetails: {e}"
                )
            except Exception as e2:
                messagebox.showerror("Export failed", f"Could not export:\n{e2}")

    def save_data(self):
        data = self.sheet.get_sheet_data()
        df = pd.DataFrame(data, columns=self.headers)
        df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig", na_rep="")

    def load_data(self):
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", keep_default_na=False)
            df = df.fillna("")
            if list(df.columns) != self.headers and len(df.columns) == len(self.headers):
                df.columns = self.headers
            self.sheet.set_sheet_data(df.values.tolist())
        else:
            self.add_rows(20)

    def save_column_widths(self):
        self._persist_column_widths_to_settings()

    def load_column_widths(self):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:
            print("load_column_widths error (read):", e)
            return
        widths = settings.get("column_widths")
        if widths is None:
            return
        try:
            self.apply_column_widths(widths)
        except Exception as e:
            print("load_column_widths error (apply):", e)

    def on_close(self):
        try:
            self.is_listening = False
            self._cancel_autosave()
            try:
                if getattr(self, "_preview_clear_after", None):
                    self.root.after_cancel(self._preview_clear_after)
                    self._preview_clear_after = None
            except Exception:
                pass
            if self._silence_guard_job:
                try: self.root.after_cancel(self._silence_guard_job)
                except Exception: pass
                self._silence_guard_job = None
            self.save_data()
            self.save_column_widths()
        finally:
            try:
                self._detach_console_if_any()
            except Exception:
                pass
            self.root.destroy()

    # Loading overlay
    def _show_loading(self, text: str = "Loading…"):
        self._loading = tk.Toplevel(self.root)
        self._loading.title("Please wait")
        self._loading.geometry("320x120")
        self._loading.transient(self.root)
        self._loading.grab_set()
        lbl = tk.Label(self._loading, text=text)
        lbl.pack(expand=True, fill="both", padx=20, pady=20)
        try: enable_crisp_dark_mode(self._loading, dark=True, delay_ms=0)
        except Exception: pass
        self._loading.update()

    def _hide_loading(self):
        try:
            if hasattr(self, "_loading") and self._loading.winfo_exists():
                self._loading.destroy()
        except Exception:
            pass

# ===========================
# Glossary Editor (unchanged features)
# ===========================
class GlossaryEditor(tk.Toplevel):
    def __init__(self, master, path: str, on_saved=None):
        super().__init__(master)
        self.title("Glossary Editor")
        self.geometry("820x500")
        self.minsize(700, 420)
        self.transient(master)
        self.path = path
        self.on_saved = on_saved or (lambda: None)

        try:
            c = _gfm_palette(True)
        except Exception:
            c = {
                "bg":"#0f1216","surface":"#151a21","elevated":"#1b212a","text":"#e6e6e6","muted":"#a8b2bd",
                "accent":"#479d7b","border":"#2b323c","sel_bg":"#263a33","sel_fg":"#eaf5f0",
                "button_bg":"#222833","button_active_bg":"#2a3140","button_fg":"#eef1f4","button_border":"#333a46",
            }

        self.configure(bg=c["bg"])

        topbar = tk.Frame(self, bg=c["surface"])
        topbar.pack(side="top", fill="x", padx=10, pady=(10, 6))

        def _style_btn(b):
            try:
                b.configure(bg=c["button_bg"], fg=c["button_fg"],
                            activebackground=c["button_active_bg"], activeforeground=c["button_fg"],
                            relief="flat", highlightthickness=1,
                            highlightbackground=c["button_border"], highlightcolor=c["button_border"])
            except Exception:
                pass

        btn_save = tk.Button(topbar, text="💾 Save", command=self._save)
        btn_reload = tk.Button(topbar, text="↺ Reload", command=self._reload)
        btn_close = tk.Button(topbar, text="Save & Close", command=self._on_close)
        for w in (btn_save, btn_reload):
            _style_btn(w); w.pack(side="left", padx=4, pady=6)
        _style_btn(btn_close); btn_close.pack(side="right", padx=4, pady=6)

        content = tk.Frame(self, bg=c["bg"])
        content.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        left = tk.Frame(content, bg=c["bg"], width=260)
        left.pack(side="left", fill="y", expand=False, padx=(0, 10))
        left.pack_propagate(True)

        right = tk.Frame(content, bg=c["bg"])
        right.pack(side="left", fill="both", expand=True)

        self.data: dict[str, list[str]] = {}
        self._load_from_file()

        tk.Label(left, text="Columns", bg=c["bg"], fg=c["text"]).pack(anchor="w")
        self.headers_lb = tk.Listbox(
            left, exportselection=False, bg=c["surface"], fg=c["text"],
            selectbackground=c["sel_bg"], selectforeground=c["sel_fg"],
            highlightthickness=1, highlightbackground=c["border"], relief="flat"
        )
        self.headers_lb.pack(fill="both", expand=True)

        btns_h = tk.Frame(left, bg=c["bg"]); btns_h.pack(fill="x", pady=(6, 0))
        for txt, cmd in (("➕ Add", self._add_header),
                         ("✏ Rename", self._rename_header),
                         ("🗑 Delete", self._delete_header)):
            b = tk.Button(btns_h, text=txt, command=cmd)
            _style_btn(b); b.pack(side="left", padx=2)

        tk.Label(right, text="Terms for selected column", bg=c["bg"], fg=c["text"]).pack(anchor="w")

        terms_wrap = tk.Frame(right, bg=c["bg"])
        terms_wrap.pack(fill="both", expand=True)

        self.terms_lb = tk.Listbox(
            terms_wrap, exportselection=False, bg=c["surface"], fg=c["text"],
            selectbackground=c["sel_bg"], selectforeground=c["sel_fg"],
            highlightthickness=1, highlightbackground=c["border"], relief="flat"
        )
        self.terms_lb.pack(side="left", fill="both", expand=True)

        tscroll_y = tk.Scrollbar(terms_wrap, orient="vertical")
        tscroll_y.pack(side="right", fill="y")
        self.terms_lb.configure(yscrollcommand=tscroll_y.set)
        tscroll_y.configure(command=self.terms_lb.yview)
        try:
            tscroll_y.configure(
                bg=c["surface"],
                activebackground=c["button_active_bg"],
                troughcolor=c["border"],
                highlightthickness=0,
                bd=0,
                relief="flat",
                width=12
            )
        except Exception:
            pass

        term_box = tk.Frame(right, bg=c["bg"])
        term_box.pack(fill="x", pady=(10, 0))

        tk.Label(term_box, text="New term", bg=c["bg"], fg=c["muted"]).pack(anchor="w")

        self._term_card = tk.Frame(term_box, bg=c.get("elevated", "#1b212a"),
                                   highlightthickness=1, highlightbackground=c["accent"], relief="flat", bd=0)
        self._term_card.pack(fill="x", expand=False, pady=(4, 8))

        self.term_entry = tk.Entry(self._term_card, font=("Segoe UI", 12), relief="flat", bd=0)
        self.term_entry.pack(fill="x", expand=True, padx=10, pady=10)

        self._restyle_term_entry(c)

        def _on_focus_in(_):
            try:
                self._term_card.configure(highlightbackground=c["accent"], highlightcolor=c["accent"])
            except Exception:
                pass

        def _on_focus_out(_):
            try:
                self._term_card.configure(highlightbackground=c["border"], highlightcolor=c["border"])
            except Exception:
                pass

        self.term_entry.bind("<FocusIn>", _on_focus_in)
        self.term_entry.bind("<FocusOut>", _on_focus_out)
        self.term_entry.bind("<Return>", lambda e: self._add_term())

        term_btns = tk.Frame(term_box, bg=c["bg"]); term_btns.pack(fill="x")
        b_add = tk.Button(term_btns, text="➕ Add term", command=self._add_term)
        b_del = tk.Button(term_btns, text="🗑 Remove term", command=self._remove_term)
        def _style_btn2(b):
            try:
                b.configure(bg=c["button_bg"], fg=c["button_fg"],
                            activebackground=c["button_active_bg"], activeforeground=c["button_fg"],
                            relief="flat", highlightthickness=1,
                            highlightbackground=c["button_border"], highlightcolor=c["button_border"])
            except Exception:
                pass
        _style_btn2(b_add); _style_btn2(b_del)
        b_add.pack(side="left"); b_del.pack(side="left", padx=(6, 0))

        self.headers_lb.bind("<<ListboxSelect>>", lambda e: self._refresh_terms())
        self._refresh_headers()
        if self.headers_lb.size() > 0:
            self.headers_lb.selection_set(0)
            self.headers_lb.activate(0)
            self._refresh_terms()
            try: self.term_entry.focus_set()
            except Exception: pass

        try:
            enable_crisp_dark_mode(self, dark=True, delay_ms=0)
        except Exception:
            pass
        try:
            self.after(30, lambda: self._restyle_term_entry(c))
        except Exception:
            pass

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_from_file(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                norm: dict[str, list[str]] = {}
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(v, list):
                            norm[str(k)] = [str(x) for x in v]
                        elif isinstance(v, dict):
                            norm[str(k)] = [str(x) for x in v.keys()]
                self.data = norm
            else:
                self.data = {}
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load glossary:\n{e}")
            self.data = {}

    def _write_to_file(self):
        tmp = os.path.join(os.path.dirname(self.path) or ".", "~glossary.tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def _restyle_term_entry(self, c: dict):
        card_bg = c.get("elevated", "#1b212a")
        field_fg = c.get("text", "#e6e6e6")
        try:
            self._term_card.configure(bg=card_bg, highlightbackground=c.get("border", "#2b323c"))
            self.term_entry.configure(
                bg=card_bg,
                fg=field_fg,
                insertbackground=field_fg,
                relief="flat",
                bd=0
            )
        except Exception:
            pass

    def _autosize_lists(self):
        try:
            from tkinter import font as _tkfont
            f = _tkfont.nametofont("TkTextFont")
        except Exception:
            f = None
        def fit(lb: tk.Listbox):
            items = lb.get(0, "end")
            if not items:
                lb.configure(width=24); return
            if f:
                px = max(f.measure(str(s)) for s in items) + 24
                w0 = max(16, f.measure("0"))
                ch = max(24, int(px / max(1, w0)))
                lb.configure(width=ch)
            else:
                ch = max(24, max(len(str(s)) for s in items))
                lb.configure(width=ch)
        fit(self.headers_lb)
        fit(self.terms_lb)

    def _refresh_headers(self):
        self.headers_lb.delete(0, "end")
        for k in sorted(self.data.keys(), key=str.lower):
            self.headers_lb.insert("end", k)
        self._autosize_lists()

    def _current_header(self) -> Optional[str]:
        sel = self.headers_lb.curselection()
        if not sel:
            return None
        return self.headers_lb.get(sel[0])

    def _ensure_header_selected(self) -> Optional[str]:
        hdr = self._current_header()
        if hdr:
            return hdr
        if self.headers_lb.size() == 1:
            self.headers_lb.selection_set(0)
            self.headers_lb.activate(0)
            hdr = self.headers_lb.get(0)
        return hdr

    def _refresh_terms(self):
        hdr = self._current_header()
        self.terms_lb.delete(0, "end")
        if hdr is None:
            self._autosize_lists()
            return
        for t in sorted(self.data.get(hdr, []), key=str.lower):
            self.terms_lb.insert("end", t)
        self._autosize_lists()

    def _add_header(self):
        name = simpledialog.askstring("New column", "Column header:")
        if not name: return
        name = name.strip()
        if not name: return
        if name in self.data:
            messagebox.showinfo("Info", "This column already exists.")
            return
        self.data[name] = []
        self._refresh_headers()
        headers_sorted = sorted(self.data.keys(), key=str.lower)
        idx = headers_sorted.index(name)
        self.headers_lb.selection_clear(0, "end")
        self.headers_lb.selection_set(idx); self.headers_lb.activate(idx)
        self._refresh_terms()
        try: self.term_entry.focus_set()
        except Exception: pass

    def _rename_header(self):
        hdr = self._current_header()
        if not hdr: return
        new = simpledialog.askstring("Rename column", "New name:", initialvalue=hdr)
        if not new: return
        new = new.strip()
        if not new or new == hdr: return
        if new in self.data:
            messagebox.showinfo("Info", "A column with that name already exists.")
            return
        self.data[new] = self.data.pop(hdr)
        self._refresh_headers()
        headers_sorted = sorted(self.data.keys(), key=str.lower)
        idx = headers_sorted.index(new)
        self.headers_lb.selection_clear(0, "end")
        self.headers_lb.selection_set(idx); self.headers_lb.activate(idx)
        self._refresh_terms()

    def _delete_header(self):
        hdr = self._current_header()
        if not hdr: return
        if not messagebox.askyesno("Confirm", f"Delete column '{hdr}' and all its terms?"):
            return
        self.data.pop(hdr, None)
        self._refresh_headers()
        self._refresh_terms()

    def _add_term(self):
        hdr = self._ensure_header_selected()
        if not hdr:
            messagebox.showinfo("Info", "Add or select a column first.")
            return
        term = self.term_entry.get().strip()
        if not term:
            return
        arr = self.data.setdefault(hdr, [])
        if term.lower() in (t.lower() for t in arr):
            messagebox.showinfo("Info", "This term already exists in the column.")
            return
        arr.append(term)
        self.term_entry.delete("end", "end")
        self._refresh_terms()

    def _remove_term(self):
        hdr = self._current_header()
        if not hdr: return
        sel = self.terms_lb.curselection()
        if not sel: return
        term = self.terms_lb.get(sel[0])
        arr = self.data.get(hdr, [])
        self.data[hdr] = [t for t in arr if t != term]
        self._refresh_terms()

    def _save(self):
        try:
            self._write_to_file()
            try:
                self.on_saved()
            except Exception:
                pass
            messagebox.showinfo("Saved", f"Glossary saved to:\n{self.path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save glossary:\n{e}")

    def _reload(self):
        self._load_from_file()
        self._refresh_headers()
        self._refresh_terms()

    def _on_close(self):
        try:
            self._write_to_file()
            try:
                self.on_saved()
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Error", f"Failed to save glossary on close:\n{e}")
            except Exception:
                pass
        finally:
            self.destroy()

# ===========================
# Settings dialog (adds model picker + speed mode)
# ===========================
class SettingsDialog(tk.Toplevel):
    def __init__(self, master, initial: dict, on_apply: Callable[[dict], None], on_reset_widths: Optional[Callable[[], None]] = None):
        super().__init__(master)
        self.title("Settings")
        self.geometry("560x460")
        self.minsize(380, 300)
        self.transient(master)
        self.on_apply = on_apply
        self.on_reset_widths = on_reset_widths

        c = _gfm_palette(True)
        self.configure(bg=c["bg"])
        frm = tk.Frame(self, bg=c["bg"])
        frm.pack(fill="both", expand=True, padx=14, pady=12)

        def lab(parent, txt):
            return tk.Label(parent, text=txt, bg=c["bg"], fg=c["text"])

        # Language
        lab(frm, "Recognition language").grid(row=0, column=0, sticky="w", pady=(0,4))
        cur_code = str(initial.get("language", "auto")).lower() if initial.get("language", None) is not None else "auto"
        cur_label = _CODE_TO_LABEL.get(cur_code, _CODE_TO_LABEL["auto"])
        self.var_lang_label = tk.StringVar(value=cur_label)
        if ttk:
            self.cmb_lang = ttk.Combobox(frm, state="readonly", values=_LANG_LABELS,
                                         textvariable=self.var_lang_label, style="Settings.TCombobox")
            self.cmb_lang.grid(row=1, column=0, sticky="ew", padx=(0,6), pady=(0,10))
        else:
            self.cmb_lang = tk.OptionMenu(frm, self.var_lang_label, *_LANG_LABELS)
            self.cmb_lang.configure(bg=c["surface"], fg=c["entry_fg"], highlightthickness=1,
                                    highlightbackground=c["border"], relief="flat")
            self.cmb_lang.grid(row=1, column=0, sticky="ew", padx=(0,6), pady=(0,10))

        # Model
        lab(frm, "Model").grid(row=2, column=0, sticky="w", pady=(0,4))
        cur_model_key = initial.get("model_key", _pick_default_model_key())
        cur_model_label = _MODELKEY_TO_LABEL.get(cur_model_key, "Small")
        self.var_model_label = tk.StringVar(value=cur_model_label)
        if ttk:
            self.cmb_model = ttk.Combobox(frm, state="readonly", values=_MODEL_LABELS,
                                          textvariable=self.var_model_label, style="Settings.TCombobox")
            self.cmb_model.grid(row=3, column=0, sticky="ew", padx=(0,6), pady=(0,10))
        else:
            self.cmb_model = tk.OptionMenu(frm, self.var_model_label, *_MODEL_LABELS)
            self.cmb_model.configure(bg=c["surface"], fg=c["entry_fg"], highlightthickness=1,
                                     highlightbackground=c["border"], relief="flat")
            self.cmb_model.grid(row=3, column=0, sticky="ew", padx=(0,6), pady=(0,10))

        # Autosave
        lab(frm, "Autosave period (minutes)").grid(row=4, column=0, sticky="w", pady=(0,4))
        self.var_auto = tk.StringVar(value=str(initial.get("autosave_minutes", 5)))
        e_auto = tk.Entry(frm, textvariable=self.var_auto, relief="flat",
                          bg=c["surface"], fg=c["entry_fg"], insertbackground=c["entry_fg"])
        e_auto.grid(row=5, column=0, sticky="ew", padx=(0,6), pady=(0,10))

        # UI scale
        lab(frm, "UI Scale (e.g., 1.00, 1.25)").grid(row=6, column=0, sticky="w", pady=(0,4))
        self.var_scale = tk.StringVar(value=str(initial.get("ui_scale", 1.0)))
        e_scale = tk.Entry(frm, textvariable=self.var_scale, relief="flat",
                           bg=c["surface"], fg=c["entry_fg"], insertbackground=c["entry_fg"])
        e_scale.grid(row=7, column=0, sticky="ew", padx=(0,6), pady=(0,10))

        # Speed mode
        self.var_speed = tk.BooleanVar(value=bool(initial.get("speed_mode", False)))
        speed_box = tk.Checkbutton(frm, text="Speed mode (lower latency, may reduce accuracy)",
                                   variable=self.var_speed, bg=c["bg"], fg=c["text"],
                                   activebackground=c["bg"], activeforeground=c["text"],
                                   selectcolor=c.get("surface", "#151a21"))
        speed_box.grid(row=8, column=0, sticky="w", pady=(4,10))

        frm.grid_columnconfigure(0, weight=1)

        btns = tk.Frame(self, bg=c["bg"])
        btns.pack(fill="x", padx=14, pady=(0,12))

        def _style_btn(b):
            try:
                b.configure(bg=c["button_bg"], fg=c["button_fg"],
                            activebackground=c["button_active_bg"], activeforeground=c["button_fg"],
                            relief="flat", highlightthickness=1,
                            highlightbackground=c["button_border"], highlightcolor=c["button_border"])
            except Exception:
                pass

        def _save_and_close():
            sel_label = self.var_lang_label.get()
            lang_code = _LABEL_TO_CODE.get(sel_label, "auto")

            try:
                auto_m = max(1, int(self.var_auto.get()))
            except Exception:
                messagebox.showerror("Invalid value", "Autosave must be an integer ≥ 1.")
                return

            try:
                scale = float(self.var_scale.get())
                if scale < 0.75 or scale > 2.5:
                    raise ValueError()
            except Exception:
                messagebox.showerror("Invalid value", "UI Scale must be between 0.75 and 2.5.")
                return

            model_label = self.var_model_label.get()
            model_key = _LABEL_TO_MODELKEY.get(model_label, _pick_default_model_key())

            s = {
                "language": lang_code,
                "autosave_minutes": auto_m,
                "ui_scale": scale,
                "model_key": model_key,
                "speed_mode": bool(self.var_speed.get()),
            }
            try:
                self.on_apply(s)
            finally:
                self.destroy()

        def _reset_widths():
            if callable(self.on_reset_widths):
                self.on_reset_widths()
                messagebox.showinfo("Column widths", "Column widths have been reset to defaults.")
            else:
                messagebox.showwarning("Unavailable", "Reset action is not available.")

        b_reset = tk.Button(btns, text="Reset column widths", command=_reset_widths)
        _style_btn(b_reset)
        b_reset.pack(side="left")

        b_ok = tk.Button(btns, text="Save", command=_save_and_close)
        b_cancel = tk.Button(btns, text="Cancel", command=self.destroy)
        _style_btn(b_ok); _style_btn(b_cancel)
        b_ok.pack(side="right", padx=(6,0))
        b_cancel.pack(side="right")

        try:
            enable_crisp_dark_mode(self, dark=True, delay_ms=0)
        except Exception:
            pass

# ===========================
# App entry
# ===========================
LAST_ACTIVITY_TS = time.monotonic()
SILENCE_GUARD_GRACE_SEC = 5.0  # don't auto-stop within first 5s after starting

def _boost_process_priority_windows():
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        # https://learn.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities
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

    # Optional: boost process priority on Windows (safe level)
    _boost_process_priority_windows()

    # Log devices
    try:
        print(f"[Devices] Whisper device={device}; compute_type={COMPUTE_TYPE}; "
              f"Silero device={_silero_device}; torch.cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("[CUDA] device name:", torch.cuda.get_device_name(0))
    except Exception as _e:
        print("Device log error:", _e)

    root = tk.Tk()
    try:
        enable_crisp_dark_mode(root, dark=True, delay_ms=350)
    except Exception:
        pass
    app = SpeechSheetApp(root)
    root.mainloop()
