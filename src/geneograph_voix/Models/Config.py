import os
import time

# Application settings

LAST_ACTIVITY_TS = time.monotonic()
SILENCE_GUARD_GRACE_SEC = 5.0

# =========================================
# Global config / settings
# =========================================

# New files / dirs
SETTINGS_FILE = "settings.json"
TEMPLATES_FILE = "templates.json"
GLOSSARIES_FILE = "glossaries.json"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "default.csv")

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

def _model_id_for_key(model_key: str) -> str:
    return model_key

# UI tuning
UI_SCALE = 1.0
TABLE_ZOOM_PCT = 135
PREVIEW_CLEAR_DELAY_MS = 1800

# Audio
SAMPLERATE = 16000
AUDIO_BLOCK_SEC = 0.10
BLOCK_DURATION = 5

# Preview + speed knobs
SPEED_MODE = False
PREVIEW_MIN_INTERVAL_SEC = 0.35
PREVIEW_TAIL_SEC_DEFAULT = 0.60
PREVIEW_TAIL_SEC_SPEED   = 0.50

COMMIT_TAIL_SILENCE_SEC = 0.60
COMMIT_MIN_SILENCE_SEC  = 0.50

def _preview_tail_sec() -> float:
    return PREVIEW_TAIL_SEC_SPEED if SPEED_MODE else PREVIEW_TAIL_SEC_DEFAULT

# Autosave
AUTOSAVE_EVERY_MS = 5 * 60 * 1000   # 5 minutes

# Columns / cleaning
NAME_COLUMNS = {"Имя", "Фамилия", "Имя отца", "Имя матери", "Имя Матери"}

# Text normalization
REMOVE_PUNCT = True  # controlled by Settings: "Remove punctuation"

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
    "Редактор субтитров","Редактор", "Спасибо", "Thank you", "Смотрите На Видео", "Увидимся", "Стук в дверь", "Динамичная музыка"
))
MALE_EXCEPTIONS = {
    "акила","арефа","вавила","варнава","иеремия","иона","исая","иуда",
    "калина","лука","осия","оссия","папа","фока","фома","никита",
    "савва","илья","кузьма","мина","сила",
}
