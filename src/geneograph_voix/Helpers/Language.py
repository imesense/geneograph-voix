import re
import unicodedata

from geneograph_voix.Models.Config import BAN_PHRASES
from geneograph_voix.Models.Settings import LANGUAGE

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

def _remove_punct_keep_hyphen(text: str) -> str:
    """
    Remove all punctuation but keep hyphens/dashes. Collapses spaces.
    Keeps: '-' U+002D, '–' U+2013, '—' U+2014.
    """
    if not text:
        return ""
    keep = "-–—"
    out = []
    for ch in str(text):
        cat = unicodedata.category(ch)
        if cat and cat.startswith("P") and ch not in keep:
            out.append(" ")
        else:
            out.append(ch)
    t = "".join(out)
    t = re.sub(r"\s+", " ", t).strip()
    return t
