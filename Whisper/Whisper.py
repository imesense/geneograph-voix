
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
import webrtcvad
from datetime import date

# --- Safety shim: ensure commit helper name exists at import time ---
if 'transcribe_buffer_commit' not in globals():
    def transcribe_buffer_commit(buffer):
        # Fallback to the standard transcribe if the specialized helper isn't defined yet.
        try:
            return transcribe_buffer(buffer, fast=False)
        except TypeError:
            return transcribe_buffer(buffer)

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_SIZE = "large-v3"
LANGUAGE = "ru"
SAMPLERATE = 16000
BLOCK_DURATION = 5  # seconds
COMPUTE_TYPE = "int8_float16"
# --- WebRTC VAD settings (commit-time speech trimming) ---
USE_WEBRTC_VAD     = True   # enable/disable trimming (falls back gracefully if module missing)
VAD_AGGRESSIVENESS = 3      # 0..3 (higher = stricter)
VAD_FRAME_MS       = 30     # 10/20/30 ms
VAD_HANG_MS        = 300    # ms of hangover after speech ends
MIN_COMMIT_SEC     = 0.4    # skip decoding if trimmed speech is shorter than this

# --- Silero VAD (preferred) ---
USE_SILERO_VAD = True
SILERO_THRESHOLD = 0.60            # 0..1, higher = stricter
SILERO_MIN_SPEECH_MS = 150
SILERO_MIN_SILENCE_MS = 250
SILERO_PAD_MS = 200


DATA_FILE = "records.csv"
COLUMN_WIDTHS_FILE = "column_widths.json"
GLOSSARY_FILE = "glossary.json"  # external glossary mapping: { "Header": ["term1", "term2", ...], ... }


# Phrases frequently hallucinated from outros / meme credits (lowercase)
BAN_PHRASES = (
    "Субтитры сделал DimaTorzok",
    "Субтитры создал DimaTorzok",
    "Субтитры создавал DimaTorzok",
    "Субтитры сделал DimaTorzhok",
    "Субтитры создал DimaTorzhok",
    "Dima Torzhok", "DimaTorzok", "DimaTorzhok",
    "Продолжение следует", "Субтитры",
)


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
# TRANSCRIBE FUNCTION
# ----------------------------
def transcribe_buffer(buffer):
    if buffer.shape[0] == 0:
        return ""
    samples = buffer.flatten()
    segments, _ = model.transcribe(
        samples,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
        no_speech_threshold=0.7,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False
    )
    text = " ".join([seg.text for seg in segments]).strip()
    return text

# ----------------------------
#  Glossary helpers (kept minimal to avoid NameErrors)
# ----------------------------
GLOSSARIES = {}  # populate if needed

def load_glossaries(path: str = GLOSSARY_FILE):
    """
    Load external glossary JSON file. Format:
    {
      "Имя": ["Иван", "Мария", "..."],
      "Фамилия": ["Иванов", "Петрова"]
    }
    Unknown or missing file -> keep existing GLOSSARIES.
    """
    import json, os
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # normalize values to lists of strings
                norm = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        norm[k] = [str(x) for x in v]
                    elif isinstance(v, dict):
                        # allow dict with any values; flatten keys as canonical terms
                        norm[k] = [str(x) for x in v.keys()]
                    else:
                        continue
                if norm:
                    GLOSSARIES.clear()
                    GLOSSARIES.update(norm)
    except Exception as e:
        print("Glossary load error:", e)

def levenshtein_distance(a: str, b: str) -> int:
    a = a or ""
    b = b or ""
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current_row = range(n+1)
    for i, c1 in enumerate(b):
        previous_row, current_row = current_row, [i+1]+[0]*n
        for j, c2 in enumerate(a):
            insertions = previous_row[j+1] + 1
            deletions  = current_row[j] + 1
            substitutions = previous_row[j-1] + (c1 != c2)
            current_row[j+1] = min(insertions, deletions, substitutions)
    return current_row[n]

def correct_text_for_column(text: str, header: str) -> str:
    # basic passthrough unless GLOSSARIES is populated
    words = (text or "").split()
    if header in GLOSSARIES and GLOSSARIES[header]:
        fixed = []
        for w in words:
            best = min(GLOSSARIES[header], key=lambda cand: levenshtein_distance(w.lower(), cand.lower()))
            if levenshtein_distance(w.lower(), best.lower()) <= max(1, len(w)//3):
                fixed.append(best)
            else:
                fixed.append(w)
        return " ".join(fixed)
    return text or ""

# ----------------------------
# Date normalization for "Дата"
# ----------------------------
RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalize_date(text: str) -> str | None:
    t = (_clean_spaces(text or "")).lower()

    # 31.08.2025  | 31/8/25 | 31-08-2025
    m = re.search(r"\b(\d{1,2})[.\-\/](\d{1,2})[.\-\/](\d{2,4})\b", t)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = 2000 + y if y < 100 else y
        try:
            return f"{date(y, mth, d):%d.%m.%Y}"
        except ValueError:
            return None

    # "30 августа 2025" or "30 августа"
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

# --- Commit-only trailing dot cleanup ---
def strip_trailing_dot(text: str) -> str:
    if text is None:
        return ""
    s = str(text).rstrip()
    # remove any final run of dots ('.' or '…') at the very end
    while s.endswith(".") or s.endswith("…"):
        s = s[:-1]
    return s

# --- HiDPI & Dark Theme Injection BEGIN ---
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
            "sel_fg": "#eaf5f0",          # high-contrast selection text
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
            "button_disabled_bg": "#f1f3f6",
            "button_disabled_fg": "#98a2ad",
            "header_bg": "#f6f8fa",
            "header_fg": "#0f141a",
            "index_bg": "#f6f8fa",
            "index_fg": "#334155",
        }

def _ui_buttons_palette(dark=True):
    # High-contrast Start/Stop + disabled/hover colors
    if dark:
        return {
            "start_bg": "#2e7d32",   # green
            "start_hover": "#2b7030",
            "start_fg": "#ffffff",
            "stop_bg":  "#c62828",   # red
            "stop_hover":"#b12525",
            "stop_fg":  "#ffffff",
            "disabled_bg": "#1f2430",
            "disabled_fg": "#717b89",
        }
    else:
        return {
            "start_bg": "#2e7d32",
            "start_hover": "#2b7030",
            "start_fg": "#ffffff",
            "stop_bg":  "#c62828",
            "stop_hover":"#b12525",
            "stop_fg":  "#ffffff",
            "disabled_bg": "#e5e7eb",
            "disabled_fg": "#9ca3af",
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
        scaling = min(2.5, scaling)
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
    base = 11; hsize = 12; mono = 11
    fam = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"
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
            try: w.configure(bg=colors["entry_bg"], fg=colors["entry_fg"], insertbackground=colors["text"], highlightthickness=0, relief="flat")
            except Exception: pass
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
                    else:
                        # DO NOT reset enabled buttons here; let custom styles persist
                        pass
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
    try: style.theme_use("clam")
    except Exception: pass
    c = _gfm_palette(dark)
    style.configure(".", background=c["bg"], foreground=c["text"])
    style.configure("TFrame", background=c["bg"])
    style.configure("TLabel", background=c["bg"], foreground=c["text"])
    style.configure("TEntry", fieldbackground=c["entry_bg"], foreground=c["entry_fg"], bordercolor=c["border"])
    style.map("TEntry", fieldbackground=[("disabled", c["button_disabled_bg"])], foreground=[("disabled", c["button_disabled_fg"])])

def _try_style_tksheet(root: "tk.Tk", dark=True):
    try:
        import tksheet
    except Exception:
        return
    colors = _gfm_palette(dark)
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
                    )
                    try:
                        c.font(("Helvetica", 11))
                        c.header_font(("Helvetica", 11, "bold"))
                        c.index_font(("Helvetica", 11))
                    except Exception: pass
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
# --- HiDPI & Dark Theme Injection END ---

# ----------------------------
# GUI APP
# ----------------------------
class SpeechSheetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper")
        self.is_listening = False

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


        # Preview label (light text for dark background)
        self.preview_label = tk.Label(root, text="Preview:", anchor="w", fg="#e6e6e6", font=("Calibri", 12, "bold"))
        self.preview_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Sheet (headers + 20 initial rows)
        self.headers = ["№","№ М","№ Ж","Дата","Имя","Фамилия","Имя отца",
                        "Имя матери","Восприемник","Страница","Комментарий"]
        load_glossaries()

        self.sheet = Sheet(root, headers=self.headers, height=400, width=1000)
        self.sheet.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        # Bindings - do NOT bind save on resize; we save only on close
        self.sheet.enable_bindings(("single_select","row_select","column_select","arrowkeys","edit_cell","rc_popup_menu", "drag_select", "column_width_resize", "row_height_resize", "copy", "paste", "delete"))

        # Responsive layout
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Load saved column widths after rendering
        self.root.after(500, self.load_column_widths)

        # High-contrast action buttons (apply now and re-apply after theme)
        self._style_action_buttons()
        self.root.after(600, self._style_action_buttons)

        # Data
        self.load_data()

        # Audio buffer
        self.buffer = np.zeros((0,1), dtype=np.float32)

        # Handle close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------------
    # SHEET HELPERS
    # ----------------------------
    def add_row(self):
        current = self.sheet.get_sheet_data(return_copy=True)
        current.append([""] * len(self.headers))
        self.sheet.set_sheet_data(current)

    def clear_all_cells(self):
        rows = len(self.sheet.get_sheet_data())
        self.sheet.set_sheet_data([[""] * len(self.headers) for _ in range(rows)])

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

        # Default for other columns (unchanged)
        self.sheet.set_cell_data(row, col, raw)
        
    def open_glossary_editor(self):
        win = GlossaryEditor(self.root, GLOSSARY_FILE, on_saved=lambda: load_glossaries())
        try:
            # apply your dark theme to the editor window as well
            enable_crisp_dark_mode(win, dark=True, delay_ms=0)
        except Exception:
            pass



    # ----------------------------
    # AUDIO RESET
    # ----------------------------
    def reset_audio_state(self):
        """Clear queued audio and preview so next session starts clean."""
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
    def _close_console_if_ours(self):
        """Close the attached console window if it was created for this process (Windows only)."""
        if sys.platform.startswith("win"):
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                user32 = ctypes.windll.user32
                hwnd = kernel32.GetConsoleWindow()
                if hwnd:
                    arr = (ctypes.c_ulong * 1)()
                    n = kernel32.GetConsoleProcessList(arr, 1)
                    if n == 1:
                        WM_CLOSE = 0x0010
                        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            except Exception:
                pass

    # ----------------------------
    # ACTION BUTTON STYLING
    # ----------------------------
    def _style_action_buttons(self):
        """High-contrast colors for Start/Stop; dim & readable when disabled."""
        START_BG  = "#2e7d32"   # green
        START_BG_H= "#2b7030"
        STOP_BG   = "#c62828"   # red
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
        temp_buffer = np.zeros((0,1), dtype=np.float32)
        while self.is_listening:
            while not audio_queue.empty():
                temp_buffer = np.concatenate((temp_buffer, audio_queue.get()))

            # Live preview for last 1s
            if len(temp_buffer) >= SAMPLERATE * 1:
                preview_text = transcribe_buffer(temp_buffer[-SAMPLERATE:])
                preview_text = strip_trailing_dot(preview_text) 
                if preview_text:
                    self.root.after(0, lambda t=preview_text: self.preview_label.config(text="🎤 Preview: " + t))

            # Commit when silence tail (~600ms) indicates end-of-speech OR a timeout
            silence_tail = temp_buffer[-int(0.6 * SAMPLERATE):]
            timeout = len(temp_buffer) >= SAMPLERATE * BLOCK_DURATION
            if (silence_tail.shape[0] >= int(0.5 * SAMPLERATE) and is_silence(silence_tail)) or timeout:
                full_text = transcribe_buffer_commit(temp_buffer)
                full_text = strip_trailing_dot(full_text)
                if full_text:
                    self.root.after(0, self.add_text_to_cell, full_text)
                temp_buffer = np.zeros((0,1), dtype=np.float32)
                self.root.after(0, lambda: self.preview_label.config(text="Preview:"))

            time.sleep(0.2)

    # ----------------------------
    # EXPORT / SAVE / LOAD
    # ----------------------------
    def export_data(self):
        data = self.sheet.get_sheet_data()
        df = pd.DataFrame(data)
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
            self.sheet.set_sheet_data(df.values.tolist())
        else:
            for _ in range(20):
                self.add_row()

    # ----------------------------
    # COLUMN WIDTHS
    # ----------------------------
    def save_column_widths(self):
        """Save column widths safely across tksheet versions"""
        try:
            if hasattr(self.sheet, "get_column_widths"):
                widths = self.sheet.get_column_widths()
            elif hasattr(self.sheet, "column_widths"):
                widths = list(self.sheet.column_widths)
            else:
                print("⚠ Could not retrieve column widths — attribute missing.")
                return

            with open(COLUMN_WIDTHS_FILE, "w", encoding="utf-8") as f:
                json.dump(widths, f)

            print("Column widths saved:", widths)
        except Exception as e:
            print("save_column_widths error:", e)

    def load_column_widths(self):
        """Load and apply saved column widths (after UI render)"""
        try:
            with open(COLUMN_WIDTHS_FILE, "r", encoding="utf-8") as f:
                widths = json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:
            print("load_column_widths error:", e)
            return
        self.apply_column_widths(widths)

    def apply_column_widths(self, widths):
        """Apply saved column widths to the sheet"""
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

    # ----------------------------
    # CLOSE
    # ----------------------------
    def on_close(self):
        try:
            self.save_data()
            self.save_column_widths()
        finally:
            try:
                self._close_console_if_ours()
            except Exception:
                pass
            self.root.destroy()

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

        # Theme colors
        try:
            c = _gfm_palette(True)
        except Exception:
            c = {
                "bg":"#0f1216","surface":"#151a21","elevated":"#1b212a","text":"#e6e6e6","muted":"#a8b2bd",
                "accent":"#479d7b","border":"#2b323c","sel_bg":"#263a33","sel_fg":"#eaf5f0",
                "button_bg":"#222833","button_active_bg":"#2a3140","button_fg":"#eef1f4","button_border":"#333a46",
            }

        self.configure(bg=c["bg"])

        # --- Top action bar (ABOVE columns/terms) ---
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

        # --- Content area (Columns left, Terms right) ---
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

        # ---------- Headers (left) ----------
        tk.Label(left, text="Columns", bg=c["bg"], fg=c["text"]).pack(anchor="w")
        self.headers_lb = tk.Listbox(
            left, exportselection=False, bg=c["surface"], fg=c["text"],
            selectbackground=c["sel_bg"], selectforeground=c["sel_fg"],
            highlightthickness=1, highlightbackground=c["border"], relief="flat"
        )
        self.headers_lb.pack(fill="both", expand=True)
        # NOTE: no scrollbars for headers (per request)

        btns_h = tk.Frame(left, bg=c["bg"]); btns_h.pack(fill="x", pady=(6, 0))
        for txt, cmd in (("➕ Add", self._add_header),
                         ("✏ Rename", self._rename_header),
                         ("🗑 Delete", self._delete_header)):
            b = tk.Button(btns_h, text=txt, command=cmd)
            _style_btn(b); b.pack(side="left", padx=2)

        # ---------- Terms (right) ----------
        tk.Label(right, text="Terms for selected column", bg=c["bg"], fg=c["text"]).pack(anchor="w")

        # Wrap list + its vertical scrollbar so the bar sits "inside" the panel
        terms_wrap = tk.Frame(right, bg=c["bg"])
        terms_wrap.pack(fill="both", expand=True)

        self.terms_lb = tk.Listbox(
            terms_wrap, exportselection=False, bg=c["surface"], fg=c["text"],
            selectbackground=c["sel_bg"], selectforeground=c["sel_fg"],
            highlightthickness=1, highlightbackground=c["border"], relief="flat"
        )
        self.terms_lb.pack(side="left", fill="both", expand=True)

        # Vertical scrollbar only (no horizontal)
        tscroll_y = tk.Scrollbar(terms_wrap, orient="vertical")
        tscroll_y.pack(side="right", fill="y")

        # Wire it up and theme it
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

        # --- New term box (clearly visible / its own background) ---
        term_box = tk.Frame(right, bg=c["bg"])
        term_box.pack(fill="x", pady=(10, 0))

        tk.Label(term_box, text="New term", bg=c["bg"], fg=c["muted"]).pack(anchor="w")

        # A visible "card" wrapper around the entry
        self._term_card = tk.Frame(term_box, bg=c.get("elevated", "#1b212a"),
                                   highlightthickness=1, highlightbackground=c["accent"], relief="flat", bd=0)
        self._term_card.pack(fill="x", expand=False, pady=(4, 8))

        # Entry inside the card with its own lighter background
        self.term_entry = tk.Entry(self._term_card, font=("Segoe UI", 12), relief="flat", bd=0)
        self.term_entry.pack(fill="x", expand=True, padx=10, pady=10)

        # Apply distinct field colors (lighter than surface)
        self._restyle_term_entry(c)

        # Focus styling: accent border on focus, normal on blur
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
        _style_btn(b_add); _style_btn(b_del)
        b_add.pack(side="left"); b_del.pack(side="left", padx=(6, 0))

        # Bindings & initial fill
        self.headers_lb.bind("<<ListboxSelect>>", lambda e: self._refresh_terms())
        self._refresh_headers()
        if self.headers_lb.size() > 0:
            self.headers_lb.selection_set(0)
            self.headers_lb.activate(0)
            self._refresh_terms()
            try: self.term_entry.focus_set()
            except Exception: pass

        # Apply dark theme to this window, then re-apply special entry styling so it isn't overwritten
        try:
            enable_crisp_dark_mode(self, dark=True, delay_ms=0)
        except Exception:
            pass
        # Ensure our custom field colors survive palette recoloring
        try:
            self.after(30, lambda: self._restyle_term_entry(c))
        except Exception:
            pass

    # ---------- File I/O ----------
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

    # ---------- Styling helper for the term field ----------
    def _restyle_term_entry(self, c: dict):
        card_bg = c.get("elevated", "#1b212a")   # outer card color
        field_fg = c.get("text", "#e6e6e6")

        try:
            # Outer card background + neutral border by default
            self._term_card.configure(bg=card_bg, highlightbackground=c.get("border", "#2b323c"))

            # Entry uses EXACTLY the same background as the card → single-color look
            self.term_entry.configure(
                bg=card_bg,
                fg=field_fg,
                insertbackground=field_fg,
                relief="flat",
                bd=0
            )
        except Exception:
            pass


    # ---------- Auto-size helpers ----------
    def _autosize_lists(self):
        """Adjust listbox 'width' (char units) to longest item; keep vertical-only scrolling."""
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
                px = max(f.measure(str(s)) for s in items) + 24   # padding
                w0 = max(16, f.measure("0"))
                ch = max(24, int(px / max(1, w0)))
                lb.configure(width=ch)
            else:
                ch = max(24, max(len(str(s)) for s in items))
                lb.configure(width=ch)

        fit(self.headers_lb)
        fit(self.terms_lb)

    # ---------- UI updates ----------
    def _refresh_headers(self):
        self.headers_lb.delete(0, "end")
        for k in sorted(self.data.keys(), key=str.lower):
            self.headers_lb.insert("end", k)
        self._autosize_lists()

    def _current_header(self) -> str | None:
        sel = self.headers_lb.curselection()
        if not sel:
            return None
        return self.headers_lb.get(sel[0])

    def _ensure_header_selected(self) -> str | None:
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

    # ---------- Actions: headers ----------
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

    # ---------- Actions: terms ----------
    def _add_term(self):
        hdr = self._ensure_header_selected()
        if not hdr:
            messagebox.showinfo("Info", "Add or select a column first.")
            return

        term = self.term_entry.get().strip()
        if not term:
            return

        arr = self.data.setdefault(hdr, [])
        if term.lower() in (t.lower() for t in arr):  # case-insensitive dedupe
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

    # ---------- Save / Reload ----------
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
        """
        Auto-save glossary on close without showing a dialog.
        If saving fails, show an error but still close the window.
        """
        try:
            self._write_to_file()
            try:
                self.on_saved()  # refresh in-memory glossary in the main app
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
# MAIN
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    # HiDPI + dark theme
    try:
        enable_crisp_dark_mode(root, dark=True, delay_ms=350)
    except Exception:
        pass
    app = SpeechSheetApp(root)
    root.mainloop()


# --- WebRTC VAD helpers ---
try:
    _vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    _has_vad = True
except Exception:
    _vad = None
    _has_vad = False

def _float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    import numpy as _np
    x = _np.clip(x, -1.0, 1.0)
    return _np.rint(x * 32767.0).astype(_np.int16).tobytes()


# --- Silero VAD (preferred if available) ---
_has_silero = False
try:
    if USE_SILERO_VAD:
        # returns (model, utils)
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            force_reload=False
        )
        (get_speech_ts, _, read_audio, *_) = _silero_utils
        _has_silero = True
except Exception as e:
    print("Silero VAD not available:", e)
    _has_silero = False

def _silero_vad_trim(buf_f32: np.ndarray, sr: int = SAMPLERATE) -> np.ndarray | None:
    """
    Use Silero to keep only speech; returns trimmed mono float32 or None if not enough speech.
    """
    if not _has_silero or buf_f32 is None or getattr(buf_f32, "size", 0) == 0:
        return buf_f32
    import torch as _torch
    wav = _torch.from_numpy(buf_f32).float()
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


def _webrtc_vad_trim(buf_f32: np.ndarray, sr: int = SAMPLERATE) -> np.ndarray | None:
    """
    Keep only frames classified as speech, with small hangover to avoid clipping ends.
    Input: mono float32 [N], Output: mono float32 or None if not enough speech.
    """
    if not _has_vad or buf_f32 is None or getattr(buf_f32, "size", 0) == 0:
        return buf_f32

    samples_per_frame = int(sr * (VAD_FRAME_MS / 1000.0))
    if samples_per_frame <= 0:
        return buf_f32
    total = len(buf_f32)
    if total < samples_per_frame:
        return None

    bytes_all = _float32_to_pcm16_bytes(buf_f32)
    frame_bytes = samples_per_frame * 2  # 2 bytes per sample (16-bit)
    frames = [bytes_all[i:i+frame_bytes] for i in range(0, len(bytes_all) - frame_bytes + 1, frame_bytes)]
    speech_flags = [False] * len(frames)

    for i, fb in enumerate(frames):
        try:
            speech_flags[i] = _vad.is_speech(fb, sr)
        except Exception:
            speech_flags[i] = False

    if sum(speech_flags) < max(2, int(0.1 * len(speech_flags))):
        return None

    hang_frames = int((VAD_HANG_MS / 1000.0) / (VAD_FRAME_MS / 1000.0))
    # find first True
    first = None
    for i, f in enumerate(speech_flags):
        if f:
            first = i; break
    # find last True
    last = None
    for j in range(len(speech_flags)-1, -1, -1):
        if speech_flags[j]:
            last = j; break

    if first is None or last is None:
        return None

    start = max(0, first - hang_frames)
    end   = min(len(speech_flags) - 1, last + hang_frames)

    start_samp = start * samples_per_frame
    end_samp   = (end + 1) * samples_per_frame
    trimmed = buf_f32[start_samp:end_samp]
    if len(trimmed) < int(sr * MIN_COMMIT_SEC):
        return None
    return trimmed

def _norm_text_basic(s: str) -> str:
    t = (s or "").lower()
    t = t.replace("ё", "е")
    t = re.sub(r"[^a-zа-я0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def looks_like_outro(s: str) -> bool:
    """True if text matches known 'credits/outro' patterns or explicit ban phrases."""
    t = _norm_text_basic(s)
    # explicit ban phrases first
    if any(p in t for p in BAN_PHRASES):
        return True
    # structural patterns that show up in these memes
    if "продолжение следует" in t:
        return True
    if "субтитры" in t and ("сделал" in t or "создал" in t):
        return True
    # latin/cyrillic variants glued (no spaces)
    t_c = t.replace(" ", "")
    if ("dimator" in t_c) or ("dimatorzhok" in t_c) or ("dimatorzok" in t_c) or ("диматоржок" in t_c):
        return True
    return False


def transcribe_buffer_commit(buffer):
    """
    Commit-time transcription with Silero/WebRTC VAD + strict decoding + outro gating.
    Preview path remains unchanged elsewhere.
    """
    import numpy as _np

    if buffer is None or buffer.size == 0:
        return ""

    # mono float32
    mono = buffer[:, 0] if getattr(buffer, "ndim", 0) > 1 else buffer
    mono = _np.asarray(mono, dtype=_np.float32, order="C")

    # Prefer Silero VAD; fallback to WebRTC; else proceed
    try:
        trimmed = None
        if _has_silero:
            trimmed = _silero_vad_trim(mono)
        elif USE_WEBRTC_VAD and _has_vad:
            trimmed = _webrtc_vad_trim(mono)
        if trimmed is None:
            return ""
        mono = trimmed
    except Exception:
        pass

    # Light pre-emphasis for SNR (copy to avoid mutating upstream)
    if mono.shape[0] > 1:
        mono = mono.copy()
        mono[1:] = mono[1:] - 0.97 * mono[:-1]

    samples = mono.flatten()

    # ---- Pass 1: deterministic, stricter heuristics ----
    try:
        segments, info = model.transcribe(
            samples,
            language=LANGUAGE,
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
            language=LANGUAGE,
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
            no_speech_threshold=0.85,
            compression_ratio_threshold=2.2,
            condition_on_previous_text=False,
        )

    segs = list(segments)
    full_text = " ".join((s.text or "").strip() for s in segs).strip()
    t_low = (full_text or "").lower()
    
    if full_text and looks_like_outro(full_text):
        # mark as suspicious so we retry (and potentially drop)
        suspicious = True
    else:
        suspicious = False

    # ---- Segment metrics gating ----
# ---- Decide if we should retry ----
    def _seg_suspicious(seg) -> bool:
        try:
            cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
            lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
            nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            return (cr > 2.2) or (lp < -0.5 and nsp > 0.50)
        except Exception:
            return False

    bad_flags = [ _seg_suspicious(sg) for sg in segs ] if segs else []
    reason = None
    suspicious = False

    if full_text and looks_like_outro(full_text):
        suspicious = True; reason = "banlist"
    elif not full_text:
        suspicious = True; reason = "empty"
    elif bad_flags and (sum(bad_flags) >= max(1, len(bad_flags)//2) or bad_flags[-1]):
        suspicious = True; reason = "metrics"

    # ---- Retry (conservative) if needed ----
    DEBUG_COMMIT = False  # set True to print gating details

    if suspicious:
        if DEBUG_COMMIT:
            print(f"[commit:retry] reason={reason} text='{full_text}'")

        # Greedy, tighter thresholds, no prompt. Deterministic and minimal.
        try:
            segments2, info2 = model.transcribe(
                samples,
                language=LANGUAGE,
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
                language=LANGUAGE,
                beam_size=1,
                temperature=0.0,
                vad_filter=True,
                no_speech_threshold=0.90,
                compression_ratio_threshold=2.0,
                condition_on_previous_text=False,
            )

        segs2 = list(segments2)
        alt = " ".join((s.text or "").strip() for s in segs2).strip()

        # Re-check ban/outro + metrics (stricter interpretation on retry)
        def _seg_suspicious_retry(seg) -> bool:
            try:
                cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
                lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
                nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                # slightly stricter on retry
                return (cr > 2.0) or (lp < -0.35 and nsp > 0.60)
            except Exception:
                return False

        bad_flags2 = [ _seg_suspicious_retry(sg) for sg in segs2 ] if segs2 else []
        looks_outro = bool(alt and looks_like_outro(alt))
        metrics_bad = bool(bad_flags2 and (sum(bad_flags2) >= max(1, len(bad_flags2)//2) or bad_flags2[-1]))
        drop = (not alt) or looks_outro or metrics_bad

        if DEBUG_COMMIT:
            print(f"[commit:retry-result] drop={drop} alt='{alt}' "
                  f"looks_outro={looks_outro} metrics_bad={metrics_bad}")

        if drop:
            return ""  # refuse this commit
        full_text = alt


    # trailing dot cleanup ONLY on commit
    try:
        full_text = strip_trailing_dot(full_text)
    except Exception:
        pass
    return full_text
