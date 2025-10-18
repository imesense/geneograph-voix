import tkinter as tk


from tksheet import Sheet
from tkinter import filedialog, messagebox, simpledialog
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import pandas as pd
from faster_whisper import WhisperModel
import torch
import warnings
import os
import json
import sys
import re
from collections import namedtuple
from datetime import date
from typing import Optional, List, Tuple, Callable
from contextlib import contextmanager
import unicodedata
from functools import lru_cache

# Try optional VAD packages safely
try:
    import webrtcvad  # pip install webrtcvad-wheels
except Exception:
    webrtcvad = None

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_SIZE = "large-v3"
LANGUAGE = "ru"
SAMPLERATE = 16000
BLOCK_DURATION = 7  # seconds to force a commit if user speaks continuously
COMPUTE_TYPE = "int8_float16"
AUTOSAVE_EVERY_MS = 5 * 60 * 1000   # 5 minutes

# --- WebRTC VAD settings (commit-time speech trimming) ---
USE_WEBRTC_VAD     = True   # enable/disable trimming (falls back gracefully if module missing)
VAD_AGGRESSIVENESS = 2      # 0..3 (higher = stricter)
VAD_FRAME_MS       = 20     # 10/20/30 ms
VAD_HANG_MS        = 700    # ms of hangover after speech ends
MIN_COMMIT_SEC     = 0.15   # skip decoding if trimmed speech is shorter than this

# --- Silero VAD (preferred) ---
USE_SILERO_VAD = True
SILERO_THRESHOLD = 0.35            # 0..1, higher = stricter
SILERO_MIN_SPEECH_MS = 100
SILERO_MIN_SILENCE_MS = 200
SILERO_PAD_MS = 300

# UI scaling & preview timing
UI_SCALE = 1                  # 100% scale
TABLE_ZOOM_PCT = 135          # 100 = native, 125 = 1.25×
PREVIEW_CLEAR_DELAY_MS = 1800 # preview clears ~1.8s after commit

DATA_FILE = "records.csv"
GLOSSARY_FILE = "glossary.json"  # external glossary mapping: { "Header": ["term1", "term2", ...], ... }
SETTINGS_FILE = "settings.json"
NAME_COLUMNS = {"Имя", "Фамилия", "Имя отца", "Имя матери", "Имя Матери"}

# --- Whisper language dropdown (labels ↔ codes) ---
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

# Male names that end with "а/я" (should count as male)
MALE_EXCEPTIONS = {
    "акила","арефа","вавила","варнава","иеремия","иона","исая","иуда",
    "калина","лука","осия","оссия","папа","фока","фома","никита",
    "савва","илья","кузьма","мина","сила",
}

# Phrases frequently hallucinated from outros / meme credits
BAN_PHRASES = tuple(s.lower() for s in (
    "Субтитры сделал DimaTorzok",
    "Субтитры создал DimaTorzok",
    "Субтитры создавал DimaTorzok",
    "Субтитры сделал DimaTorzhok",
    "Субтитры создал DimaTorzhok",
    "Dima Torzhok", "DimaTorzok", "DimaTorzhok",
    "Продолжение следует", "Субтитры",
    "Редактор субтитров А.Семкин Корректор А.Егорова",
    "Редактор субтитров","Редактор"
))

def _preview_is_banned(text: str) -> bool:
    """Cheap guard for preview: true if text looks like a banned/outro phrase."""
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
    """
    Keep only letters (Latin/Cyrillic), collapse spaces, and Proper-case each word.
    Removes dots, commas, quotes, digits, etc.
    """
    t = text or ""
    t = re.sub(r"[^A-Za-zА-Яа-яЁё\-]+", " ", t)  # keep letters and hyphen
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    return " ".join(w[:1].upper() + w[1:].lower() if w else "" for w in t.split())

# ----------------------------
# AUDIO QUEUE
# ----------------------------
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(indata.copy())

def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    x = x.astype(np.float64)
    rms = np.sqrt(np.mean(x**2))
    return 20*np.log10(rms + 1e-12)

def is_silence(buf: np.ndarray, threshold_db: float = -45.0) -> bool:
    return rms_db(buf.flatten()) < threshold_db

# ----------------------------
# WHISPER MODEL
# ----------------------------
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper model ({MODEL_SIZE}) on {device}...")
model = WhisperModel(MODEL_SIZE, device=device, compute_type=COMPUTE_TYPE)

# ----------------------------
#  Settings helpers
# ----------------------------

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
    """Apply dict to module-level knobs (no side-effects like UI/VAD here)."""
    global LANGUAGE, AUTOSAVE_EVERY_MS, VAD_AGGRESSIVENESS, UI_SCALE
    try:
        if "language" in s and isinstance(s["language"], str) and s["language"].strip():
            LANGUAGE = s["language"].strip()
        if "autosave_minutes" in s:
            m = int(s["autosave_minutes"])
            AUTOSAVE_EVERY_MS = max(1, m) * 60 * 1000
        if "vad_aggr" in s:
            VAD_AGGRESSIVENESS = max(0, min(3, int(s["vad_aggr"])))
        if "ui_scale" in s:
            UI_SCALE = float(s["ui_scale"])
    except Exception as e:
        print("settings apply error:", e)

def _reinit_webrtc_vad():
    """Recreate WebRTC VAD object after aggressiveness change (thread-safe)."""
    global _vad, _has_vad
    try:
        with _vad_lock:
            if USE_WEBRTC_VAD and webrtcvad is not None:
                _vad = webrtcvad.Vad(int(VAD_AGGRESSIVENESS))
                _has_vad = True
            else:
                _vad = None
                _has_vad = False
    except Exception as e:
        print("reinit VAD error:", e)
        with _vad_lock:
            _vad = None
            _has_vad = False

# ----------------------------
#  Glossary helpers
# ----------------------------
GLOSSARIES = {}  # populated by load_glossaries()

# Versioning + caches for best-candidate lookups
GLOSSARY_VER = 0
BEST_CACHE_SINGLE: dict[tuple[str, str, int], tuple[Optional["_Best"], Optional["_Best"]]] = {}
BEST_CACHE_MERGE: dict[tuple[str, str, int], tuple[Optional["_Best"], Optional["_Best"]]] = {}

def _clear_best_caches():
    BEST_CACHE_SINGLE.clear()
    BEST_CACHE_MERGE.clear()

def load_glossaries(path: str = GLOSSARY_FILE):
    """
    Load external glossary JSON file. Format:
    {
      "Имя": ["Иван", "Мария", "..."],
      "Фамилия": ["Иванов", "Петрова"]
    }
    Unknown or missing file -> keep existing GLOSSARIES.
    """
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

# ---- Conservative glossary correction ----

def _strip_diacritics(s: str) -> str:
    if not s:
        return ""
    # Normalize and strip combining marks (for Latin-based diacritics)
    n = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in n if not unicodedata.combining(ch))

def _ru_norm(s: str) -> str:
    """
    Language-aware normalization:
      - lowercasing
      - ё→е
      - keep Cyrillic letters incl. Ukrainian: а-я + і ї є ґ
      - strip Latin diacritics to base ASCII and keep a–z
      - drop everything else
    """
    t = (s or "").lower().replace("ё", "е")
    t = _strip_diacritics(t)
    # Keep ASCII a-z and Cyrillic incl. Ukrainian extras
    t = re.sub(r"[^a-zа-яіїєґ]+", "", t)
    return t

def _ru_norm_merge(s: str) -> str:
    """
    Merge-only normalization used when trying to glue tokens:
    - start with _ru_norm
    - collapse 'тс' -> 'ц' (helps 'Кот Сага' → 'Коцага')
    """
    t = _ru_norm(s)
    return t.replace("тс", "ц")

# --- Confusion / phonetic helpers for Russian/UA core ---

CONFUSION_CLASSES = [
    set("бп"), set("вф"), set("гкх"), set("дт"), set("зсц"),
    set("жшщч"),  # merge strong sibilants
]

# Keyboard adjacency disabled (previous noisy map removed intentionally)
KEYBOARD_NEIGHBORS_RU: dict[str, set[str]] = {}

# Vowels: Cyrillic (ru/ua) + ASCII
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
    """
    Very light 'phonetic' reduction for Russian-like strings:
      - ё→е already handled in _ru_norm
      - collapse voiced/voiceless pairs and sibilants
      - collapse vowels to 5 buckets: a/e/i/o/u
    """
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
            out.append("x")   # strong sibilants
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

# --- Weighted Damerau–Levenshtein (restricted) ---

# Tunable costs
_WDL_INS_COST   = 1.0
_WDL_DEL_COST   = 1.0
_WDL_SWAP_COST  = 0.8
_WDL_CONF_COST  = 0.55   # same confusion class (в↔ф, к↔г, ...)
_WDL_VOWEL_COST = 0.70   # vowel↔vowel
_WDL_NEAR_COST  = 0.65   # keyboard neighbors
_WDL_BASE_COST  = 1.0    # unrelated substitution

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
    """
    Restricted Damerau–Levenshtein with weighted substitutions
    and constant insertion/deletion/swap costs.
    Operates on already-normalized lowercase strings.
    """
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

            # transposition (restricted)
            if i >= 2 and j >= 2 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                swap = dp[i-2][j-2] + _WDL_SWAP_COST
                if swap < best:
                    best = swap

            dp[i][j] = best
    return dp[n][m]

def _combined_score(a_norm: str, b_norm: str) -> tuple[float, float, float, int]:
    """
    Return (score, wdl, dice, anchor) where lower score is better.
    """
    wdl  = _wdl(a_norm, b_norm)
    dice = _dice_sim(a_norm, b_norm)
    # phonetic rescue
    a_ph = ru_phon_key(a_norm)
    b_ph = ru_phon_key(b_norm)
    phon = _wdl(a_ph, b_ph)

    # anchor = max shared prefix/suffix
    lcp  = _lcp_len(a_norm, b_norm)
    lcs  = _lcs_len(a_norm, b_norm)
    anchor = max(lcp, lcs)

    # weighted mixture (tuned for short RU/UA names)
    score = (
        wdl
        + 0.45 * phon
        + (1.0 - dice)
        - 0.10 * min(anchor, 5)
    )
    return score, wdl, dice, anchor


BRIDGE_WORDS = {"в", "во", "и", "й", "а"}  # tiny words we can skip in 3-token merges

@lru_cache(maxsize=8192)
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
            cur[j] = min(prev[j] + 1,      # deletion
                         cur[j-1] + 1,      # insertion
                         prev[j-1] + cost)  # substitution
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

def _best_two_candidates(a_norm: str, g_norm: list[tuple[str, str]]) -> tuple[Optional[_Best], Optional[_Best]]:
    """
    Find best and runner-up candidates for given normalized input `a_norm`.
    Ranking primarily by weighted DL, then by lower (1 - Dice), then anchor.
    """
    if not a_norm:
        return None, None

    best: Optional[_Best] = None
    second: Optional[_Best] = None

    first = a_norm[0] if a_norm else ""
    last  = a_norm[-1] if a_norm else ""
    use_coarse = len(a_norm) >= 5

    def _edge_ok(a_ch: str, b_ch: str) -> bool:
        # relaxed: allow same confusion class at edges
        if not a_ch or not b_ch:
            return False
        return a_ch == b_ch or _same_confusion_class(a_ch, b_ch)

    for orig, b in g_norm:
        if not b:
            continue
        if use_coarse and not (_edge_ok(first, b[0]) or _edge_ok(last, b[-1])):
            continue

        # integer Levenshtein (legacy signal)
        d_lev = _lev(a_norm, b)

        # prune by classic len-aware max edits
        L = max(len(a_norm), len(b))
        if d_lev > _len_aware_max_edits(L):
            # allow a soft pass if the weighted DL is still very small
            # (e.g., confusable pairs): compute quick wdl and keep only if <= cap + 0.5
            wdl_soft = _wdl(a_norm, b)
            if wdl_soft > _len_aware_max_edits(L) + 0.5:
                continue

        # main metrics
        dice = _dice_sim(a_norm, b)
        lcp  = _lcp_len(a_norm, b)
        lcs  = _lcs_len(a_norm, b)
        wdl  = _wdl(a_norm, b)
        # phonetic DL for acceptance check (cheap to cache here)
        phon = _wdl(ru_phon_key(a_norm), ru_phon_key(b))

        cand = _Best(orig=orig, norm=b, d=d_lev, dice=dice, lcp=lcp, lcs=lcs, wdl=wdl, phon=phon)

        def better(x: _Best, y: _Best) -> bool:
            # prefer smaller wdl, then higher dice, then stronger anchor
            if x.wdl != y.wdl:
                return x.wdl < y.wdl
            if x.dice != y.dice:
                return x.dice > y.dice
            return max(x.lcp, x.lcs) > max(y.lcp, y.lcs)

        if best is None or better(cand, best):
            second = best
            best = cand
        elif second is None or better(cand, second):
            second = cand

    return best, second

def _best_two_cached(header: str, a_norm: str, g_norm: list[tuple[str, str]], *, is_merge: bool) -> tuple[Optional[_Best], Optional[_Best]]:
    """
    Cached wrapper keyed by (header, a_norm, GLOSSARY_VER, is_merge).
    """
    key = (header, a_norm, GLOSSARY_VER)
    cache = BEST_CACHE_MERGE if is_merge else BEST_CACHE_SINGLE
    if key in cache:
        return cache[key]
    res = _best_two_candidates(a_norm, g_norm)
    cache[key] = res
    return res

def _should_correct_from_norm(a_norm: str, best: Optional[_Best], runner_up: Optional[_Best]) -> bool:
    """
    Hybrid acceptance with weighted DL (wdl), phonetic backstop, Dice and anchor.
    Slightly lenient to catch 'Афгсентий' -> 'Авксентий', but still conservative.
    """
    if not best or not a_norm:
        return False

    L = max(len(a_norm), len(best.norm))
    wdl  = best.wdl
    dice = best.dice
    anchor = max(best.lcp, best.lcs)
    phon = best.phon

    # Baseline thresholds
    max_ed = _len_aware_max_edits(L) + 0.5   # allow half-point slack with weighted DL
    min_d  = _min_dice_for_len(L) - 0.03     # tiny softening for hybrid scorer
    req_anchor = 2 if L <= 5 else 3

    # Phonetic rescue: if Dice is a bit low, accept with strong phonetic match
    if dice < min_d and phon <= 1.0:
        min_d -= 0.07   # allow slightly lower dice if phonetic distance is minimal

    if wdl > max_ed:
        return False
    if dice < min_d:
        return False
    if anchor < req_anchor:
        return False

    if runner_up is None:
        return True

    # Margin vs runner-up: compare combined scores
    score_best, _, _, _   = _combined_score(a_norm, best.norm)
    score_run , _, _, _   = _combined_score(a_norm, runner_up.norm)

    # Need a clear margin to avoid flip-flop corrections
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
        # lower is better; use full combined score for consistency
        sc, _, _, _ = _combined_score(a_norm, cand.norm)
        return sc

    while i < len(tokens):
        tok = tokens[i]
        a = _ru_norm(tok)

        cand2 = None
        cand3 = None

        # 2-token merge
        if i + 1 < len(tokens):
            tok2 = tokens[i + 1]
            joined2 = tok + tok2
            a_join2 = _ru_norm_merge(joined2)
            if a_join2:
                bestm2, runnerm2 = _best_two_cached(header, a_join2, g_norm_merge, is_merge=True)
                if _should_correct_from_norm(a_join2, bestm2, runnerm2):
                    if len(a_join2) >= 5 and (bestm2.lcp >= 3 or bestm2.lcs >= 3):
                        cand2 = (bestm2, _score_from_norm(a_join2, bestm2))

        # 3-token merge (with/without bridge)
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
            if tok2_clean in BRIDGE_WORDS:
                joined13 = tok + tok3
                a_join13 = _ru_norm_merge(joined13)
                if a_join13:
                    bestm3b, runnerm3b = _best_two_cached(header, a_join13, g_norm_merge, is_merge=True)
                    if _should_correct_from_norm(a_join13, bestm3b, runnerm3b):
                        if len(a_join13) >= 6 and (bestm3b.lcp >= 3 or bestm3b.lcs >= 3):
                            cand3 = (bestm3b, _score_from_norm(a_join13, bestm3b))

        # choose best merge
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

        # single-token fallback
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

# ----------------------------
# Date normalization for "Дата"
# ----------------------------
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
            d = int(m2.group(1))
            mth = RU_MONTHS[mon]
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

# --- Trailing dot cleanup ---
def strip_trailing_dot(text: str) -> str:
    if text is None:
        return ""
    s = str(text).rstrip()
    while s.endswith(".") or s.endswith("…"):
        s = s[:-1]
    return s

# ----------------------------
# VAD INIT & HELPERS
# ----------------------------
_has_vad = False
try:
    if USE_WEBRTC_VAD and webrtcvad is not None:
        _vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        _has_vad = True
    else:
        _vad = None
        _has_vad = False
except Exception:
    _vad = None
    _has_vad = False
    
# Protect access to _vad / _has_vad across threads
_vad_lock = threading.RLock()

def _float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return np.rint(x * 32767.0).astype(np.int16).tobytes()

def _webrtc_vad_trim(buf_f32: np.ndarray, sr: int = SAMPLERATE) -> Optional[np.ndarray]:
    # snapshot the VAD pointer & flag under lock
    with _vad_lock:
        has_vad = _has_vad
        vad_obj = _vad

    if not has_vad or buf_f32 is None or getattr(buf_f32, "size", 0) == 0:
        return buf_f32

    samples_per_frame = int(sr * (VAD_FRAME_MS / 1000.0))
    if samples_per_frame <= 0:
        return buf_f32
    total = len(buf_f32)
    if total < samples_per_frame:
        return None

    bytes_all = _float32_to_pcm16_bytes(buf_f32)
    frame_bytes = samples_per_frame * 2
    frames = [bytes_all[i:i+frame_bytes] for i in range(0, len(bytes_all) - frame_bytes + 1, frame_bytes)]
    speech_flags = [False] * len(frames)
    for i, fb in enumerate(frames):
        try:
            speech_flags[i] = vad_obj.is_speech(fb, sr)
        except Exception:
            speech_flags[i] = False

    if sum(speech_flags) < max(2, int(0.1 * len(speech_flags))):
        return None

    hang_frames = int((VAD_HANG_MS / 1000.0) / (VAD_FRAME_MS / 1000.0))
    first = next((i for i, f in enumerate(speech_flags) if f), None)
    last = next((i for i in range(len(speech_flags)-1, -1, -1) if speech_flags[i]), None)
    if first is None or last is None:
        return None
    start = max(0, first - hang_frames); end = min(len(speech_flags) - 1, last + hang_frames)
    start_samp = start * samples_per_frame; end_samp = (end + 1) * samples_per_frame

    trimmed = buf_f32[start_samp:end_samp]
    if len(trimmed) < int(sr * MIN_COMMIT_SEC):
        return None
    return trimmed


_has_silero = False
try:
    if USE_SILERO_VAD:
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            force_reload=False
        )
        (get_speech_ts, _, read_audio, *_) = _silero_utils  # read_audio unused
        _has_silero = True
except Exception as e:
    print("Silero VAD not available:", e)
    _has_silero = False

def _silero_vad_trim(buf_f32: np.ndarray, sr: int = SAMPLERATE) -> Optional[np.ndarray]:
    if not _has_silero or buf_f32 is None or getattr(buf_f32, "size", 0) == 0:
        return buf_f32
    wav = torch.from_numpy(buf_f32).float()
    ts = get_speech_ts(
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
    if len(trimmed) < int(sr * MIN_COMMIT_SEC):
        return None
    return trimmed

# ----------------------------
# Outro/Ban detectors
# ----------------------------
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

# ----------------------------
# TRANSCRIBE FUNCTIONS
# ----------------------------
def _effective_language():
    if LANGUAGE is None:
        return None
    if isinstance(LANGUAGE, str) and LANGUAGE.strip().lower() in ("", "auto"):
        return None
    return LANGUAGE

def transcribe_buffer(buffer):
    if buffer.shape[0] == 0:
        return ""
    samples = buffer.flatten()
    segments, _ = model.transcribe(
        samples,
        language=_effective_language(),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
        no_speech_threshold=0.7,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False
    )
    text = " ".join([seg.text for seg in segments]).strip()
    return text

def transcribe_buffer_commit(buffer):
    if buffer is None or buffer.size == 0:
        return ""

    mono = buffer[:, 0] if getattr(buffer, "ndim", 0) > 1 else buffer
    mono = np.asarray(mono, dtype=np.float32, order="C")

    use_trim = False
    trimmed = mono
    try:
        if _has_silero:
            use_trim = True
            trimmed = _silero_vad_trim(mono)
        elif USE_WEBRTC_VAD and _has_vad:
            use_trim = True
            trimmed = _webrtc_vad_trim(mono)
    except Exception:
        pass

    if use_trim and trimmed is None:
        # second chance: if the raw buffer has energy, try untrimmed
        try:
            if not is_silence(mono, threshold_db=-50.0):
                trimmed = mono
            else:
                return ""
        except Exception:
            return ""

    mono = trimmed

    if mono.shape[0] > 1:
        mono = mono.copy()
        mono[1:] = mono[1:] - 0.97 * mono[:-1]

    samples = mono.flatten()

    try:
        segments, info = model.transcribe(
            samples,
            language=_effective_language(),
            beam_size=5,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
            no_speech_threshold=0.85,
            compression_ratio_threshold=2.2,
            logprob_threshold=-0.5,
            suppress_blank=True,
            initial_prompt="Диктовка для таблицы: имена, фамилии, даты. Не писать титры или подписи.",
        )
    except TypeError:
        segments, info = model.transcribe(
            samples,
            language=_effective_language(),
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
            no_speech_threshold=0.85,
            compression_ratio_threshold=2.2,
            condition_on_previous_text=False,
        )

    segs = list(segments)
    full_text = " ".join((s.text or "").strip() for s in segs).strip()

    def _seg_suspicious(seg) -> bool:
        try:
            cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
            lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
            nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            return (cr > 2.2) or (lp < -0.5 and nsp > 0.50)
        except Exception:
            return False

    bad_flags = [_seg_suspicious(sg) for sg in segs] if segs else []
    reason = None
    suspicious = False

    if full_text and looks_like_outro(full_text):
        suspicious = True; reason = "banlist"
    elif not full_text:
        suspicious = True; reason = "empty"
    elif bad_flags and (sum(bad_flags) >= max(1, len(bad_flags)//2) or bad_flags[-1]):
        suspicious = True; reason = "metrics"

    DEBUG_COMMIT = False
    if suspicious:
        if DEBUG_COMMIT:
            print(f"[commit:retry] reason={reason} text='{full_text}'")
        try:
            segments2, info2 = model.transcribe(
                samples,
                language=_effective_language(),
                beam_size=1,
                temperature=0.0,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
                no_speech_threshold=0.90,
                compression_ratio_threshold=2.0,
                logprob_threshold=-0.3,
                suppress_blank=True,
            )
        except TypeError:
            segments2, info2 = model.transcribe(
                samples,
                language=_effective_language(),
                beam_size=1,
                temperature=0.0,
                vad_filter=True,
                no_speech_threshold=0.90,
                compression_ratio_threshold=2.0,
                condition_on_previous_text=False,
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

        bad_flags2 = [_seg_suspicious_retry(sg) for sg in segs2] if segs2 else []
        looks_outro = bool(alt and looks_like_outro(alt))
        metrics_bad = bool(bad_flags2 and (sum(bad_flags2) >= max(1, len(bad_flags2)//2) or bad_flags2[-1]))
        drop = (not alt) or looks_outro or metrics_bad

        if DEBUG_COMMIT:
            print(f"[commit:retry-result] drop={drop} alt='{alt}' "
                  f"looks_outro={looks_outro} metrics_bad={metrics_bad}")
        if drop:
            return ""
        full_text = alt

    try:
        full_text = strip_trailing_dot(full_text)
    except Exception:
        pass
    return full_text

# ----------------------------
# HiDPI & Dark Theme
# ----------------------------
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

    # ttk.Entry style (in case it’s used)
    style.configure("TEntry",
                    fieldbackground=c["surface"],
                    foreground=c["entry_fg"],
                    bordercolor=c["border"])
    style.map("TEntry",
              fieldbackground=[("disabled", c["button_disabled_bg"])],
              foreground=[("disabled", c["button_disabled_fg"])])

    # Combobox (dropdown) – visually distinct like entries
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

# ----------------------------
# GUI APP
# ----------------------------
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

        self.add_row_btn = tk.Button(btn_frame, text="➕ Add Row", command=self.add_row)
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

        # Create the Sheet but do NOT grid yet
        self.sheet = Sheet(root, headers=self.headers, height=400, width=1000, zoom=TABLE_ZOOM_PCT)

        # Apply saved widths immediately using the existing function
        try:
            self.load_column_widths()
        except Exception as e:
            print("initial load_column_widths error:", e)

        # Now grid the Sheet (first paint uses the saved widths)
        self.sheet.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.sheet.enable_bindings(("single_select","row_select","column_select","arrowkeys","edit_cell",
                                    "rc_popup_menu","drag_select","column_width_resize","row_height_resize",
                                    "copy","cut","paste","delete","undo","double_click_column_resize",
                                    "double_click_row_resize","rc_insert_column","rc_delete_column",
                                    "rc_insert_row","rc_delete_row"))

        # Responsive layout
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Optional tiny fallback reapply in case the first map nudges sizes
        self.root.after(60, self.load_column_widths)

        # Load saved column widths after rendering
        self.root.after(500, self.load_column_widths)

        # High-contrast action buttons
        self._style_action_buttons()
        self.root.after(600, self._style_action_buttons)

        # Data
        self.load_data()

        # Audio buffer
        self.buffer = np.zeros((0,1), dtype=np.float32)

        # Handle close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------- small helper: transient preview note -------
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

    # ----------------------------
    # Column width helpers
    # ----------------------------
    def _snapshot_column_widths(self):
        """Return current widths (list or dict) from tksheet."""
        try:
            if hasattr(self.sheet, "get_column_widths"):
                return self.sheet.get_column_widths()
            elif hasattr(self.sheet, "column_widths"):
                return list(self.sheet.column_widths)
        except Exception as e:
            print("snapshot widths error:", e)
        return None

    def apply_column_widths(self, widths):
        """Apply width structure returned by _snapshot_column_widths / settings."""
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
        """Snapshot widths and restore them after the enclosed block."""
        widths = self._snapshot_column_widths()
        try:
            yield
        finally:
            if widths is not None:
                try:
                    self.apply_column_widths(widths)
                except Exception as e:
                    print("restore widths error:", e)

    # ----------------------------
    # SHEET HELPERS
    # ----------------------------
    def _get_sheet_data_copy(self):
        try:
            return self.sheet.get_sheet_data(return_copy=True)
        except TypeError:
            data = self.sheet.get_sheet_data()
            return [list(row) for row in data]

    def add_row(self):
        # Preserve widths during structural change
        with self.preserve_column_widths():
            data = self._get_sheet_data_copy()
            data.append([""] * len(self.headers))
            self.sheet.set_sheet_data(data)

    def clear_all_cells(self):
        # Preserve widths during mass update
        with self.preserve_column_widths():
            rows = len(self.sheet.get_sheet_data())
            self.sheet.set_sheet_data([[""] * len(self.headers) for _ in range(rows)])

    def _persist_column_widths_to_settings(self):
        """Save current widths into settings.json under 'column_widths'."""
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
        """Reset to tksheet defaults: remove saved widths and rebuild Sheet."""
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

    def _bind_toggle_hotkeys(self):
        """F1 toggles start/stop; Esc stops. Safe around text editing widgets."""
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
            "vad_aggr": VAD_AGGRESSIVENESS,
            "ui_scale": UI_SCALE,
        }

    def _save_settings_file(self, s: dict):
        try:
            tmp = os.path.join(os.path.dirname(SETTINGS_FILE) or ".", "~settings.tmp.json")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
            os.replace(tmp, SETTINGS_FILE)
        except Exception as e:
            print("settings save error:", e)

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
            # 1) globals + persist
            _apply_settings_to_globals(s)
            self._save_settings_file(self._current_settings())

            # 2) VAD reinit (now thread-safe + workers paused)
            _reinit_webrtc_vad()

            # 3) autosave tick reset
            self._start_autosave()

            # 4) Only restyle UI if UI-affecting knobs changed
            ui_scale_changed = float(prev.get("ui_scale", 1.0)) != float(s.get("ui_scale", prev.get("ui_scale", 1.0)))
            if ui_scale_changed:
                _apply_dark_ui(self.root, dark=True)
                self._refresh_preview_font()
                self._maybe_update_sheet_zoom()
                self._style_action_buttons()
            # If only VAD/aggr changed, skip heavy restyle to keep UI snappy
        except Exception as e:
            print("apply_settings error:", e)
        finally:
            self._settings_applying = False

    def open_settings(self):
        SettingsDialog(
            self.root,
            initial=self._current_settings(),
            on_apply=self.apply_settings,
            on_reset_widths=self.reset_column_widths,
        )

    # ----------------------------
    # AUDIO RESET
    # ----------------------------
    def reset_audio_state(self):
        global audio_queue
        try:
            while not audio_queue.empty():
                audio_queue.get_nowait()
        except Exception:
            pass
        self.buffer = np.zeros((0,1), dtype=np.float32)
        try:
            self.preview_label.config(text="Preview:")
        except Exception:
            pass

    # ----------------------------
    # WINDOWS CONSOLE AUTO-CLOSE
    # ----------------------------
    def _detach_console_if_any(self):
        if sys.platform.startswith("win"):
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                if kernel32.GetConsoleWindow():
                    kernel32.FreeConsole()
            except Exception:
                pass

    # ----------------------------
    # ACTION BUTTON STYLING
    # ----------------------------
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
            apply(
                self.start_btn,
                bg=START_BG, bg_h=START_BG_H, fg=TXT_LIGHT,
                disabled=(str(self.start_btn.cget("state")) == "disabled")
            )
            apply(
                self.stop_btn,
                bg=STOP_BG, bg_h=STOP_BG_H, fg=TXT_LIGHT,
                disabled=(str(self.stop_btn.cget("state")) == "disabled")
            )
        except Exception:
            pass

    # ----------------------------
    # LISTENING
    # ----------------------------
    def start_listening(self):
        self.reset_audio_state()
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
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='float32',
                            blocksize=int(SAMPLERATE * 0.2), callback=audio_callback):
            while self.is_listening:
                sd.sleep(100)

    def transcribe_thread(self):
        temp_buffer = np.zeros((0, 1), dtype=np.float32)
        while self.is_listening:
            # If settings are being applied, pause work to avoid races / jank
            if getattr(self, "_settings_applying", False):
                time.sleep(0.05)
                continue
            while not audio_queue.empty():
                temp_buffer = np.concatenate((temp_buffer, audio_queue.get()))

            if len(temp_buffer) >= SAMPLERATE * 1:
                preview_text = transcribe_buffer(temp_buffer[-SAMPLERATE:])
                if preview_text and _preview_is_banned(preview_text):
                    preview_text = ""
                if preview_text:
                    preview_text = strip_trailing_dot(preview_text)
                    now = time.monotonic()
                    self._last_preview_text = preview_text
                    self._last_preview_ts = now
                    self.root.after(0, lambda t=preview_text: self.preview_label.config(text="Preview: " + t))

            silence_tail = temp_buffer[-int(0.6 * SAMPLERATE):]
            timeout = len(temp_buffer) >= SAMPLERATE * BLOCK_DURATION

            if (silence_tail.shape[0] >= int(0.5 * SAMPLERATE) and is_silence(silence_tail)) or timeout:
                full_text = transcribe_buffer_commit(temp_buffer)
                full_text = strip_trailing_dot(full_text)
                
                if not full_text:
                    recent_preview = (time.monotonic() - getattr(self, "_last_preview_ts", 0.0)) < 3.0
                    if recent_preview and getattr(self, "_last_preview_text", ""):
                        try:
                            tail = temp_buffer[-int(1.5 * SAMPLERATE):]
                            alt = transcribe_buffer(tail)
                            alt = strip_trailing_dot(alt)
                            if alt and not looks_like_outro(alt):
                                full_text = alt
                        except Exception:
                            pass

                try:
                    if full_text and looks_like_outro(full_text):
                        full_text = ""
                except NameError:
                    pass

                if full_text:
                    self.root.after(0, self.add_text_to_cell, full_text)
                    self.root.after(0, lambda t=full_text: self.preview_label.config(text="Preview: " + t))

                try:
                    if getattr(self, "_preview_clear_after", None):
                        self.root.after_cancel(self._preview_clear_after)
                except Exception:
                    pass

                def _clear_preview():
                    try:
                        self.preview_label.config(text="Preview:")
                    except Exception:
                        pass
                    finally:
                        self._preview_clear_after = None

                self._preview_clear_after = self.root.after(PREVIEW_CLEAR_DELAY_MS, _clear_preview)

                temp_buffer = np.zeros((0, 1), dtype=np.float32)

            time.sleep(0.1)

    # ----------------------------
    # EXPORT / SAVE / LOAD
    # ----------------------------
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
        else:
            df.to_excel(file_path, index=False)
        messagebox.showinfo("Exported", f"Data exported to:\n{file_path}")

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
            for _ in range(20):
                self.add_row()

    # ----------------------------
    # COLUMN WIDTHS (persist inside settings.json)
    # ----------------------------
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

    # ----------------------------
    # CLOSE
    # ----------------------------
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

            self.save_data()
            self.save_column_widths()
        finally:
            try:
                self._detach_console_if_any()
            except Exception:
                pass
            self.root.destroy()

# ----------------------------
# Glossary Editor
# ----------------------------
class GlossaryEditor(tk.Toplevel):
    """
    Simple editor for glossary.json:
    { "Имя": ["Иван","Мария"], "Фамилия": ["Иванов","Петров"], ... }
    """
    def __init__(self, master, path: str, on_saved=None):
        super().__init__(master)
        self.title("Glossary Editor")
        self.geometry("820x500")
        self.minsize(700, 420)
        self.transient(master)
        self.path = path
        self.on_saved = on_saved or (lambda: None)

        # Theme colors (fixed a typo in sel_bg hex)
        try:
            c = _gfm_palette(True)
        except Exception:
            c = {
                "bg":"#0f1216","surface":"#151a21","elevated":"#1b212a","text":"#e6e6e6","muted":"#a8b2bd",
                "accent":"#479d7b","border":"#2b323c","sel_bg":"#263a33","sel_fg":"#eaf5f0",
                "button_bg":"#222833","button_active_bg":"#2a3140","button_fg":"#eef1f4","button_border":"#333a46",
            }

        self.configure(bg=c["bg"])

        # --- Top action bar ---
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

        # --- Content area ---
        content = tk.Frame(self, bg=c["bg"])
        content.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        left = tk.Frame(content, bg=c["bg"], width=260)
        left.pack(side="left", fill="y", expand=False, padx=(0, 10))
        left.pack_propagate(True)

        right = tk.Frame(content, bg=c["bg"])
        right.pack(side="left", fill="both", expand=True)

        # Data
        self.data: dict[str, list[str]] = {}
        self._load_from_file()

        # Headers list
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

        # Terms list
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

        # New term card
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
        self.term_entry.delete(0, "end")
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
            try: self.on_saved()
            except Exception: pass
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

# ----------------------------
# Settings
# ----------------------------
class SettingsDialog(tk.Toplevel):
    def __init__(self, master, initial: dict, on_apply: Callable[[dict], None], on_reset_widths: Optional[Callable[[], None]] = None):
        super().__init__(master)
        self.title("Settings")
        self.geometry("560x380")
        self.minsize(380, 260)
        self.transient(master)
        self.on_apply = on_apply
        self.on_reset_widths = on_reset_widths

        c = _gfm_palette(True)
        self.configure(bg=c["bg"])

        # Content
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
            self.cmb_lang = ttk.Combobox(
                frm, state="readonly", values=_LANG_LABELS,
                textvariable=self.var_lang_label, style="Settings.TCombobox",
            )
            self.cmb_lang.grid(row=1, column=0, sticky="ew", padx=(0,6), pady=(0,10))
        else:
            self.cmb_lang = tk.OptionMenu(frm, self.var_lang_label, *_LANG_LABELS)
            self.cmb_lang.configure(bg=c["surface"], fg=c["entry_fg"], highlightthickness=1,
                                    highlightbackground=c["border"], relief="flat")
            self.cmb_lang.grid(row=1, column=0, sticky="ew", padx=(0,6), pady=(0,10))

        # Autosave
        lab(frm, "Autosave period (minutes)").grid(row=2, column=0, sticky="w", pady=(0,4))
        self.var_auto = tk.StringVar(value=str(initial.get("autosave_minutes", 5)))
        e_auto = tk.Entry(frm, textvariable=self.var_auto, relief="flat",
                          bg=c["surface"], fg=c["entry_fg"], insertbackground=c["entry_fg"])
        e_auto.grid(row=3, column=0, sticky="ew", padx=(0,6), pady=(0,10))

        # VAD
        lab(frm, "VAD aggressiveness (0–3)").grid(row=4, column=0, sticky="w", pady=(0,4))
        self.var_vad = tk.StringVar(value=str(initial.get("vad_aggr", 1)))
        e_vad = tk.Entry(frm, textvariable=self.var_vad, relief="flat",
                         bg=c["surface"], fg=c["entry_fg"], insertbackground=c["entry_fg"])
        e_vad.grid(row=5, column=0, sticky="ew", padx=(0,6), pady=(0,10))

        # UI scale
        lab(frm, "UI Scale (e.g., 1.00, 1.25)").grid(row=6, column=0, sticky="w", pady=(0,4))
        self.var_scale = tk.StringVar(value=str(initial.get("ui_scale", 1.0)))
        e_scale = tk.Entry(frm, textvariable=self.var_scale, relief="flat",
                           bg=c["surface"], fg=c["entry_fg"], insertbackground=c["entry_fg"])
        e_scale.grid(row=7, column=0, sticky="ew", padx=(0,6), pady=(0,10))

        frm.grid_columnconfigure(0, weight=1)

        # Buttons
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
                vad = int(self.var_vad.get())
                if vad < 0 or vad > 3:
                    raise ValueError()
            except Exception:
                messagebox.showerror("Invalid value", "VAD aggressiveness must be 0, 1, 2, or 3.")
                return
            try:
                scale = float(self.var_scale.get())
                if scale < 0.75 or scale > 2.5:
                    raise ValueError()
            except Exception:
                messagebox.showerror("Invalid value", "UI Scale must be between 0.75 and 2.5.")
                return

            s = {
                "language": lang_code,
                "autosave_minutes": auto_m,
                "vad_aggr": vad,
                "ui_scale": scale,
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

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    _apply_settings_to_globals(_read_settings_from_file())
    _reinit_webrtc_vad()

    root = tk.Tk()
    try:
        enable_crisp_dark_mode(root, dark=True, delay_ms=350)
    except Exception:
        pass
    app = SpeechSheetApp(root)
    root.mainloop()
