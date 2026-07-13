import re
import unicodedata

from collections import namedtuple
from datetime import date
from functools import lru_cache
from typing import List, Optional

from geneograph_voix.Models.ModelCoefficients import GLOSSARY_STRICTNESS

# ===========================
# Glossary correction engine (same logic; now applicable to any mapped column)
# ===========================
# We keep all original helpers below; the key change is we REMOVED the early gate
# "if header not in NAME_COLUMNS: return text" so mapped non-name columns can use lists.

def _glossary_thresholds(L: int, level: int | None = None):
    level = int(max(1, min(5, level if level is not None else GLOSSARY_STRICTNESS)))
    base_ed  = _len_aware_max_edits(L)
    base_d   = _min_dice_for_len(L)
    ed_bonus = {1:0.3, 2:0.5, 3:0.7, 4:1.0, 5:1.4}[level]
    dice_adj = {1:+0.06, 2:+0.03, 3:0.00, 4:-0.05, 5:-0.10}[level]
    anchor_base = 2 if L <= 5 else 3
    anchor_adj  = {1:0, 2:0, 3:0, 4:-1, 5:-1}[level]
    score_gap   = {1:0.40, 2:0.38, 3:0.35, 4:0.22, 5:0.15}[level]
    max_ed   = base_ed + ed_bonus
    min_d    = max(0.0, min(0.95, base_d + dice_adj))
    req_anchor = max(1, anchor_base + anchor_adj)
    return max_ed, min_d, req_anchor, score_gap

GLOSSARIES = {}  # dynamic per-column synthetic key: "col:<uid>" -> [terms]

def _clear_best_caches():
    BEST_CACHE_SINGLE.clear()
    BEST_CACHE_MERGE.clear()

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

def _trim_soft(s: str) -> str:
    return re.sub(r"[ьъ]+$", "", s or "")

def _cons_skeleton(s: str) -> str:
    t = _trim_soft(s.lower().replace("ё", "е"))
    return "".join(ch for ch in t if ch.isalpha() and not _is_vowel_ru(ch) and ch not in "й")

CONFUSION_CLASSES = [
    set("бп"), set("вф"), set("гкх"), set("дт"), set("зсц"),
    set("жшщч"),
]
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

    def _edge_ok(a_ch: str, b_ch: str, *, is_last: bool = False) -> bool:
        if not a_ch or not b_ch:
            return False
        if a_ch == b_ch:
            return True
        if is_last and (a_ch in "ьъ" or b_ch in "ьъ"):
            return True
        if _same_confusion_class(a_ch, b_ch):
            return True
        if _is_vowel_ru(a_ch) and _is_vowel_ru(b_ch):
            return True
        return False

    for orig, b in g_norm:
        if not b:
            continue
        if use_coarse and not (_edge_ok(first, b[0]) or _edge_ok(last, b[-1], is_last=True)):
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
        cand = _Best(orig=orig, norm=b, d=dice, dice=dice, lcp=lcp, lcs=lcs, wdl=wdl, phon=phon)

        def better(x: _Best, y: _Best) -> bool:
            if x.wdl != y.wdl: return x.wdl < y.wdl
            if x.dice != y.dice: return x.dice > y.dice
            return max(x.lcp, x.lcs) > max(y.lcp, y.lcs)

        if best is None or better(cand, best):
            second = best; best = cand
        elif second is None or better(cand, second):
            second = cand
    return best, second

BEST_CACHE_SINGLE: dict[tuple[str, str, int], tuple[Optional[_Best], Optional[_Best]]] = {}
BEST_CACHE_MERGE: dict[tuple[str, str, int], tuple[Optional[_Best], Optional[_Best]]] = {}

def _best_two_cached(header: str, a_norm: str, g_norm: list[tuple[str, str]], *, is_merge: bool) -> tuple[Optional[_Best], Optional[_Best]]:
    key = (header, a_norm, 1 if is_merge else 0)
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

    max_ed, min_d, req_anchor, score_gap = _glossary_thresholds(L, GLOSSARY_STRICTNESS)

    sk_a = _cons_skeleton(a_norm)
    sk_b = _cons_skeleton(best.norm)
    sk_dice = _dice_sim(sk_a, sk_b) if sk_a and sk_b else 0.0
    sk_len  = max(len(sk_a), len(sk_b))

    if dice < min_d and phon <= 1.0:
        min_d -= 0.05
    if GLOSSARY_STRICTNESS >= 4 and sk_len >= 3 and sk_dice >= 0.80:
        min_d = min(min_d, 0.50)

    if wdl > max_ed:
        return False
    if dice >= min_d and anchor >= req_anchor:
        return True


    if runner_up is None:
        # Levels 1–3: no correction if it failed the main (dice+anchor) gate above.
        if GLOSSARY_STRICTNESS <= 3:
            return False
        # Levels 4–5: keep the original permissive fallback.
        return (anchor >= req_anchor) or (sk_len >= 3 and sk_dice >= 0.80 and wdl <= (max_ed + 0.2))

    score_best, _, _, _ = _combined_score(a_norm, best.norm)
    score_run , _, _, _ = _combined_score(a_norm, runner_up.norm)
    if (score_run - score_best) >= score_gap and anchor >= max(1, req_anchor - 1):
        return True

    if GLOSSARY_STRICTNESS >= 4:
        if (_lev(a_norm, best.norm) <= 2 and max(best.lcp, best.lcs) >= (L - 2)):
            return True
        if sk_len >= 3 and sk_dice >= 0.85 and _lev(sk_a, sk_b) <= 1:
            return True
        if _trim_soft(a_norm) == _trim_soft(best.norm) and _lev(a_norm, best.norm) <= 2:
            return True
    return False

def _should_correct(token: str, best: Optional[_Best], runner_up: Optional[_Best]) -> bool:
    a_norm = _ru_norm(token)
    return _should_correct_from_norm(a_norm, best, runner_up)

def correct_text_for_column(text: str, header_key: str) -> str:
    """
    Apply glossary correction if GLOSSARIES[header_key] exists and has terms.
    Unlike the old version, we don't limit this to NAME columns only —
    we allow any column that has mapped lists.
    """
    if not text:
        return ""
    glossary = GLOSSARIES.get(header_key) or []
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
                bestm2, runnerm2 = _best_two_cached(header_key, a_join2, g_norm_merge, is_merge=True)
                if bestm2 and _should_correct_from_norm(a_join2, bestm2, runnerm2):
                    if len(a_join2) >= 5 and (bestm2.lcp >= 3 or bestm2.lcs >= 3):
                        cand2 = (bestm2, _score_from_norm(a_join2, bestm2))

        if i + 2 < len(tokens):
            tok2 = tokens[i + 1]
            tok3 = tokens[i + 2]
            joined123 = tok + tok2 + tok3
            a_join123 = _ru_norm_merge(joined123)
            if a_join123:
                bestm3a, runnerm3a = _best_two_cached(header_key, a_join123, g_norm_merge, is_merge=True)
                if bestm3a and _should_correct_from_norm(a_join123, bestm3a, runnerm3a):
                    if len(a_join123) >= 6 and (bestm3a.lcp >= 3 or bestm3a.lcs >= 3):
                        cand3 = (bestm3a, _score_from_norm(a_join123, bestm3a))
            tok2_clean = _ru_norm(tok2)
            if tok2_clean in {"в","во","и","й","а"}:
                joined13 = tok + tok3
                a_join13 = _ru_norm_merge(joined13)
                if a_join13:
                    bestm3b, runnerm3b = _best_two_cached(header_key, a_join13, g_norm_merge, is_merge=True)
                    if bestm3b and _should_correct_from_norm(a_join13, bestm3b, runnerm3b):
                        if len(a_join13) >= 6 and (bestm3b.lcp >= 3 or bestm3b.lcs >= 3):
                            cand3 = (bestm3b, _score_from_norm(a_join13, bestm3b))

        best_merge = cand2 if (cand2 and (not cand3 or cand2[1] <= cand3[1])) else cand3
        if best_merge is not None:
            ok1 = ok2 = False
            score1 = score2 = float("inf")
            if a:
                best1, runner1 = _best_two_cached(header_key, a, g_norm, is_merge=False)
                ok1 = _should_correct(tok, best1, runner1)
                if ok1 and best1:
                    score1 = _score_from_norm(a, best1)
            if i + 1 < len(tokens):
                a2 = _ru_norm(tokens[i + 1])
                if a2:
                    best2, runner2 = _best_two_cached(header_key, a2, g_norm, is_merge=False)
                    ok2 = _should_correct(tokens[i + 1], best2, runner2)
                    if ok2 and best2:
                        score2 = _score_from_norm(a2, best2)
            score_m = best_merge[1]
            if (not ok1 and not ok2) or (score_m + 0.30 <= min(score1, score2)):
                out.append(best_merge[0].orig)
                i += 3 if (cand3 and best_merge == cand3) else 2
                continue

        if a:
            best, runner = _best_two_cached(header_key, a, g_norm, is_merge=False)
            if best and _should_correct(tok, best, runner):
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
