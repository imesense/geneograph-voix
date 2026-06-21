import threading
import time
import os
import sys
import re
import unicodedata
import torch

import tkinter as tk
import numpy as np
import pandas as pd
import sounddevice as sd

from collections import deque
from contextlib import contextmanager
from typing import Optional, List, Tuple, Callable
from tkinter import filedialog, messagebox, simpledialog
from tksheet import Sheet

from geneograph_voix.Helpers.ScreenHelpers import (
    _fit_to_screen,
    _fit_to_screen_bounds,
    _parse_geometry,
    _set_windows_dpi_awareness
)
from geneograph_voix.Helpers.ResourceHelpers import _resource_path
from geneograph_voix.Models.Cache import _prepare_frozen_caches
from geneograph_voix.Models.Config import (
    _CODE_TO_LABEL,
    _LABEL_TO_CODE,
    _LABEL_TO_MODELKEY,
    _LANG_LABELS,
    _MODEL_LABELS,
    _MODELKEY_TO_LABEL,
    AUDIO_BLOCK_SEC,
    AUTOSAVE_EVERY_MS,
    BLOCK_DURATION,
    COMMIT_MIN_SILENCE_SEC,
    COMMIT_TAIL_SILENCE_SEC,
    DATA_DIR,
    DATA_FILE,
    MALE_EXCEPTIONS,
    NAME_COLUMNS,
    PREVIEW_CLEAR_DELAY_MS,
    PREVIEW_MIN_INTERVAL_SEC,
    REMOVE_PUNCT,
    SAMPLERATE,
    SPEED_MODE,
    TABLE_ZOOM_PCT,
    UI_SCALE,
    _preview_tail_sec
)
from geneograph_voix.Models.Devices import (
    DEVICE,
)
from geneograph_voix.Models.GlobalGlossaryLists import (
    GLOSSARY_LISTS,
    _active_template_record,
    _data_path_for_template,
    _ensure_templates_file,
    _make_default_template,
    _new_uuid,
    _set_active_template_id,
    _write_templates_file,
    load_glossary_lists,
    save_glossary_lists
)
from geneograph_voix.Models.GlossaryCorrectionEngine import (
    GLOSSARIES,
    _clear_best_caches,
    correct_text_for_column,
    normalize_date
)
from geneograph_voix.Models.GlossaryTextUtilities import (
    AUDIO_QUEUE,
    _preview_is_banned,
    audio_callback,
    clean_person_field,
    is_silence,
    rms_db
)
from geneograph_voix.Models.LanguageHelpers import (
    _effective_language,
    _remove_punct_keep_hyphen,
    looks_like_outro,
    strip_trailing_dot
)
from geneograph_voix.Models.ModelCoefficients import (
    GLOSSARY_STRICTNESS,
    VAD_STRICTNESS,
    _energy_gate_db,
    _no_speech_thresholds
)
from geneograph_voix.Models.Settings import (
    LANGUAGE,
    MODEL_KEY,
    _apply_settings_to_globals,
    _read_settings_from_file,
    _save_settings_file
)
from geneograph_voix.Models.Sheets import SHEET_BINDS
from geneograph_voix.Models.SileroVadSettings import (
    HAS_SILERO,
    SILERO_DEVICE,
    _load_silero_vad,
    _silero_vad_trim
)
from geneograph_voix.Models.WhisperModelWrapper import (
    COMPUTE_TYPE,
    _load_whisper_model,
    _pick_default_model_key,
    _warmup_model,
    fw_transcribe
)
from geneograph_voix.Views.Palette import _set_palette

_prepare_frozen_caches()

os.makedirs(DATA_DIR, exist_ok=True)

# ===========================
# Decoding functions
# ===========================
def transcribe_buffer(buffer):
    if buffer.size == 0:
        return ""
    if rms_db(buffer.flatten()) < (_energy_gate_db() + 2.0):
        return ""
    prev_nst, _, _, cr_prev, _ = _no_speech_thresholds()
    segments, _ = fw_transcribe(
        buffer,
        language=_effective_language(),
        beam_size=1,
        temperature=0.0,
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=prev_nst,
        compression_ratio_threshold=cr_prev,
    )
    return " ".join((seg.text or "").strip() for seg in segments).strip()

def transcribe_buffer_commit(buffer):
    if buffer is None or buffer.size == 0:
        return ""
    mono = buffer[:, 0] if getattr(buffer, "ndim", 0) > 1 else buffer
    mono = np.asarray(mono, dtype=np.float32, order="C")
    if rms_db(mono) < _energy_gate_db():
        return ""
    trimmed = mono
    try:
        if HAS_SILERO:
            t = _silero_vad_trim(mono)
            if t is not None:
                trimmed = t
            else:
                if not is_silence(mono):
                    trimmed = mono
                else:
                    return ""
    except Exception:
        pass
    if trimmed.shape[0] > 1:
        x = trimmed.copy()
        x[1:] = x[1:] - 0.97 * x[:-1]
        trimmed = x
    samples = trimmed.flatten()
    prev_nst, commit1_nst, commit2_nst, _, cr_commit = _no_speech_thresholds()
    segments, info = fw_transcribe(
        samples,
        language=_effective_language(),
        beam_size=1,
        temperature=0.0,
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=commit1_nst,
        compression_ratio_threshold=cr_commit,
    )
    segs = list(segments)
    full_text = " ".join((s.text or "").strip() for s in segs).strip()

    def _seg_suspicious(seg) -> bool:
        try:
            cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
            lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
            nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            return (cr > 2.1) or (lp < -0.65 and nsp > 0.45)
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

    if suspicious:
        segments2, info2 = fw_transcribe(
            samples,
            language=_effective_language(),
            beam_size=1,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=commit2_nst,
            compression_ratio_threshold=max(1.95, cr_commit - 0.05),
        )
        segs2 = list(segments2)
        alt = " ".join((s.text or "").strip() for s in segs2).strip()

        def _seg_suspicious_retry(seg) -> bool:
            try:
                cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
                lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
                nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                return (cr > 2.0) or (lp < -0.55 and nsp > 0.50)
            except Exception:
                return False

        bad2 = [_seg_suspicious_retry(sg) for sg in segs2] if segs2 else []
        looks_outro = bool(alt and looks_like_outro(alt))
        metrics_bad = bool(bad2 and (sum(bad2) >= max(1, len(bad2)//2) or bad2[-1]))
        if (not alt) or looks_outro or metrics_bad:
            return ""
        full_text = alt

    low = (full_text or "").lower()
    if len(low) <= 24 and any(w in low for w in ("музык", "аплодисмент", "барабан", "спасибо")):
        return ""
    try:
        full_text = strip_trailing_dot(full_text)
    except Exception:
        pass
    return full_text

# ===========================
# UI theming
# ===========================
try:
    from tkinter import ttk
    import tkinter.font as tkfont
except Exception:
    ttk = None
    tkfont = None

# ---------- App icon helpers ----------
APP_ICON_ICO = "app.ico"        # put the same .ico you use with PyInstaller here
APP_ICON_PNG = "app_256.png"    # optional: for non-Windows, a PNG works best

def _set_app_icon(root: "tk.Tk"):
    """Set the app/window icon robustly across platforms/bundles."""
    try:
        if sys.platform.startswith("win"):
            ico = _resource_path(APP_ICON_ICO)
            if os.path.exists(ico):
                # Titlebar + task switcher icon on Windows
                root.iconbitmap(ico)
        else:
            png = _resource_path(APP_ICON_PNG)
            if os.path.exists(png):
                # Apply to all toplevels created after this call
                img = tk.PhotoImage(file=png)
                root.iconphoto(True, img)
                # Prevent image from being garbage-collected
                root._app_icon_img = img
    except Exception as e:
        print("icon set warn:", e)

# Remember the original Tk scaling so re-applying doesn't multiply it
_BASE_TK_SCALING = None

def _auto_scaling(root: "tk.Tk"):
    global _BASE_TK_SCALING
    try:
        if _BASE_TK_SCALING is None:
            # Tk's 'scaling' is pixels-per-point (1pt = 1/72"). Default is usually 1.0.
            _BASE_TK_SCALING = float(root.tk.call('tk', 'scaling')) or 1.0
        new_scale = max(0.5, min(2.5, float(_BASE_TK_SCALING) * float(UI_SCALE)))
        root.tk.call('tk', 'scaling', new_scale)
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
    colors = _set_palette(dark)
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
    c = _set_palette(dark)
    style.configure(".", background=c["bg"], foreground=c["text"])
    style.configure("TFrame", background=c["bg"])
    style.configure("TLabel", background=c["bg"], foreground=c["text"])
    style.configure("TEntry", fieldbackground=c["surface"], foreground=c["entry_fg"], bordercolor=c["border"])
    style.map("TEntry", fieldbackground=[("disabled", c["button_disabled_bg"])], foreground=[("disabled", c["button_disabled_fg"])])
    style.configure("Settings.TCombobox",
                    background=c["surface"], fieldbackground=c["surface"], foreground=c["entry_fg"],
                    bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"])
    style.map("Settings.TCombobox",
              fieldbackground=[("readonly", c["surface"]), ("!disabled", c["surface"])],
              foreground=[("readonly", c["entry_fg"])],
              selectbackground=[("readonly", c["sel_bg"])],
              selectforeground=[("readonly", c["sel_fg"])])

def _try_style_tksheet(root: "tk.Tk", dark=True):
    try:
        import tksheet
    except Exception:
        return
    colors = _set_palette(dark)
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

        # Load glossary lists
        load_glossary_lists()

        # Template bootstrap (create default if none)
        default_headers = ["№","№ М","№ Ж","Дата","Имя","Фамилия","Имя отца",
                           "Имя матери","Восприемник","Страница","Комментарий"]
        self._templates_cache = _ensure_templates_file(default_headers)
        self.active_template = _active_template_record()
        if not self.active_template:
            # should not happen; ensure one exists
            t = _make_default_template(default_headers)
            self._templates_cache = {"last_template_id": t["id"], "templates": [t]}
            _write_templates_file(self._templates_cache)
            self.active_template = t

        # ---- Date columns state (initialize BEFORE UI reads it) ----
        cols = (self.active_template.get("columns") or [])
        self._date_cols_columns = [(str(c.get("uid", "")), str(c.get("header", ""))) for c in cols]

        dc = self.active_template.get("date_columns")
        if isinstance(dc, list):
            selected = [str(x) for x in dc]
        else:
            # Legacy default: preselect headers named "Дата" if template has no date_columns yet
            selected = [str(c.get("uid", "")) for c in cols
                        if str(c.get("header", "")).strip().lower() == "дата"]

        self._date_cols_original = set(selected)
        self._date_cols_pending = set(selected)

        # build headers from active template
        self.headers = [col["header"] for col in self.active_template["columns"]]

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

        self.add_col_btn = tk.Button(btn_frame, text="➕ Add Column", command=self.add_column)
        self.add_col_btn.grid(row=0, column=4, padx=5)

        self.clear_btn = tk.Button(btn_frame, text="🧹 Clear All", command=self.clear_all_cells)
        self.clear_btn.grid(row=0, column=5, padx=5)

        self.glossary_btn = tk.Button(btn_frame, text="📚 Glossary", command=self.open_glossary_editor)
        self.glossary_btn.grid(row=0, column=6, padx=5)

        self.auto_num_btn = tk.Button(btn_frame, text="🔢 Auto Numerate", command=self.auto_numerate)
        self.auto_num_btn.grid(row=0, column=7, padx=5)

        self.settings_btn = tk.Button(btn_frame, text="⚙️ Settings", command=self.open_settings)
        self.settings_btn.grid(row=0, column=8, padx=5)

        self.template_btn = tk.Button(btn_frame, text="🧩 Templates", command=self.open_template_manager)
        self.template_btn.grid(row=0, column=9, padx=5)

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
        self.sheet = Sheet(root, headers=self.headers, height=400, width=1000, zoom=TABLE_ZOOM_PCT)

        # Column widths (per-template)
        try:
            self.load_column_widths()
        except Exception as e:
            print("initial load_column_widths error:", e)

        self.sheet.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self._attach_sheet_bindings_safely()
        self._hk_in_dispatch = False  # re-entrancy guard for hotkey interceptor

        self._bind_sheet_events()
        self._install_cross_layout_shortcuts(force=True)
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        self.root.after(60, self.load_column_widths)
        self.root.after(500, self.load_column_widths)

        self._style_action_buttons()
        self.root.after(600, self._style_action_buttons)
        self._bind_geometry_persistence()
        self._restore_main_window_geometry()

        # Status bar (bottom)
        status_colors = _set_palette(True)
        self.status_frame = tk.Frame(root, bg=status_colors["surface"])
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0,8))
        self.template_status_label = tk.Label(
            self.status_frame,
            text="",
            anchor="w",
            bg=status_colors["surface"],
            fg=status_colors["muted"]
        )
        self.template_status_label.pack(side="left", fill="x", expand=True)
        self._update_template_status()

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
            print(f"[Devices] Whisper device={DEVICE}; compute_type={COMPUTE_TYPE}; "
                  f"Silero device={SILERO_DEVICE}; torch.cuda={torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print("[CUDA] device name:", torch.cuda.get_device_name(0))
        except Exception as _e:
            print("Device log error:", _e)

    # ------- UI helpers -------
    def _restore_main_window_geometry(self):
        try:
            s = _read_settings_from_file()
            geom = s.get("window_geometry")
            state = s.get("window_state", "normal")

            # Track what we restored so we can detect user-driven changes later
            self._geom_restored = None

            if geom:
                parsed = _parse_geometry(geom)
                if parsed:
                    w, h, x, y = parsed
                    w, h, x, y = _fit_to_screen_bounds(self.root, w, h, x, y)
                    self.root.geometry(f"{w}x{h}+{x}+{y}")
                    self._geom_restored = (w, h, x, y)

            self.root.update_idletasks()
            if state == "zoomed":
                try:
                    self.root.state("zoomed")
                except Exception:
                    pass

            # NOTE: no unconditional post-restore save here anymore.
            # We only save later if the user actually resizes/moves the window.
        except Exception as e:
            print("restore geometry warn:", e)

    def _save_main_window_geometry(self):
        try:
            state = self.root.state()
            s = _read_settings_from_file()
            s = dict(s) if isinstance(s, dict) else {}

            # Save state; only persist geometry when in 'normal' state
            s["window_state"] = state if state in ("zoomed", "normal") else "normal"
            if s["window_state"] == "normal":
                s["window_geometry"] = self.root.winfo_geometry()

            _save_settings_file(s)
        except Exception as e:
            print("save geometry warn:", e)

    def _bind_geometry_persistence(self):
        # Defer geometry autosaves until layout has stabilized
        self._geom_save_job = None
        self._geom_ready = False
        self._geom_boot_deadline = time.monotonic() + 1.2  # wait ~1.2s after launch
        self._last_saved_geom = None

        def _on_cfg(evt=None):
            # Don’t save while applying settings or before the boot deadline
            if getattr(self, "_settings_applying", False):
                return
            if time.monotonic() < self._geom_boot_deadline or not self._geom_ready:
                return

            # Only persist in normal state (zoomed/fullscreen produces junk numbers)
            try:
                if self.root.state() != "normal":
                    return
            except Exception:
                return

            # Ignore obviously bogus/early sizes
            try:
                w = int(getattr(evt, "width", 0) or 0)
                h = int(getattr(evt, "height", 0) or 0)
                if w and h and (w < 400 or h < 300):
                    return
            except Exception:
                pass

            # Throttle writes
            if self._geom_save_job:
                try:
                    self.root.after_cancel(self._geom_save_job)
                except Exception:
                    pass

            def _commit():
                try:
                    geom = self.root.winfo_geometry()
                    if geom != self._last_saved_geom:
                        self._save_main_window_geometry()
                        self._last_saved_geom = geom
                except Exception:
                    pass

            self._geom_save_job = self.root.after(350, _commit)

        # Start listening after first idle so requested sizes are computed
        def _arm_ready():
            # mark “ready” a bit after first paint; dark-mode restyle runs at 350ms
            self._geom_ready = True

        self.root.bind("<Configure>", _on_cfg, add="+")
        self.root.bind("<Map>", _on_cfg, add="+")
        self.root.after(700, _arm_ready)


    # ------- Template helpers -------
    
    def _bind_sheet_events(self):
        """Attach tksheet event handlers once per sheet instance."""
        try:
            self.sheet.extra_bindings(bindings={
                "end_edit_header": self._on_end_edit_header,
                "column_drag_and_drop": lambda e: self._persist_column_order_to_template(),
                "begin_edit_cell": lambda e: self._ensure_editor_hotkey_tag(), 
            })
        except Exception as e:
            print("extra_bindings attach error:", e)

    def _attach_sheet_bindings_safely(self):
        """Enable base tksheet bindings after the widget is realized; supports both iterable and varargs APIs."""
        def _do():
            try:
                self.sheet.enable_bindings(SHEET_BINDS)     # most builds accept iterable
            except TypeError:
                self.sheet.enable_bindings(*SHEET_BINDS)    # some builds require varargs
            except Exception as e:
                print("enable_bindings warning:", e)
        try:
            self.sheet.after(0, _do)
        except Exception as e:
            print("after(0) attach error:", e)
            _do()

    def _destroy_sheet_safely(self):
        """Disable bindings before destroying to avoid stale extra bindings."""
        try:
            # Best effort: stop tksheet from keeping extra bindings alive
            self.sheet.disable_bindings()
        except Exception:
            pass
        try:
            self.sheet.destroy()
        except Exception:
            pass

    def _on_end_edit_header(self, event=None):
        """
        Called after a column header edit in tksheet. Persist headers + order to template,
        preserving UIDs and existing mappings.
        """
        try:
            self._persist_column_order_to_template()
            # Keep our in-memory header cache in sync with the widget
            self.headers = self._current_headers_list()
            # Small visual hint
            self._flash_preview_note("✅ Column header saved", ms=900)
        except Exception as e:
            print("end_edit_header persist error:", e)

    def _rebuild_sheet(self, new_headers: list[str], data: list[list[str]]):
        with self.preserve_column_widths():
            try:
                self._destroy_sheet_safely()
            except Exception:
                pass
            self.headers = list(new_headers)
            self.sheet = Sheet(self.root, headers=self.headers, height=400, width=1000, zoom=TABLE_ZOOM_PCT)
            self.sheet.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
            self._attach_sheet_bindings_safely()
            self._bind_sheet_events()
            self._install_cross_layout_shortcuts(force=True)
            self._try_style()
            self.sheet.set_sheet_data(data)
            self.load_column_widths()

    def _save_current_state_before_duplicate(self):
        try:
            self._persist_column_order_to_template()
            self.save_data()
            self.save_column_widths()
        except Exception as e:
            print("pre-duplicate save error:", e)

    def _update_template_status(self):
        try:
            nm = (self.active_template.get("name") or "(unnamed)") if self.active_template else "—"
            ident = (self.active_template.get("id","")[:8]) if self.active_template else ""
            self.template_status_label.config(text=f"Template: {nm} [{ident}]")
        except Exception:
            pass

    def _reload_templates_cache(self):
        self._templates_cache = _ensure_templates_file(self.headers)

    def _reload_active_template_from_disk(self):
        """Reload latest template+mapping from templates.json into the app."""
        try:
            self._reload_templates_cache()
            self.active_template = _active_template_record()
            self._update_template_status()
        except Exception as e:
            print("reload active template error:", e)

    def _get_template_by_id(self, tid: str) -> Optional[dict]:
        for t in self._templates_cache.get("templates", []):
            if t.get("id") == tid:
                return t
        return None

    def _set_active_template(self, tid: str):
        # Save current data + widths first
        try:
            self.save_data()
            self.save_column_widths()
        except Exception as e:
            print("save before switch err:", e)

        self._persist_column_order_to_template()
        _set_active_template_id(tid)
        self._reload_templates_cache()
        t = self._get_template_by_id(tid)
        if not t:
            messagebox.showerror("Templates", "Template not found.")
            return
        self.active_template = t
        self.headers = [col["header"] for col in t["columns"]]
        try:
            self._destroy_sheet_safely()
        except Exception:
            pass
        self.sheet = Sheet(self.root, headers=self.headers, height=400, width=1000, zoom=TABLE_ZOOM_PCT)
        self.sheet.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self._attach_sheet_bindings_safely()
        self._bind_sheet_events()
        self._install_cross_layout_shortcuts(force=True)
        self._try_style()
        # Load data for the new template
        self.load_data()
        # Apply template-specific widths
        self.load_column_widths()
        self._flash_preview_note(f"🧩 Switched to template: {t.get('name','(unnamed)')}", ms=1300)
        self._update_template_status()
        # Clear glossary state/caches after switching templates
        _clear_best_caches()
        GLOSSARIES.clear()

    # --- Append mode columns helpers (per-template) ---
    def _append_columns_set(self) -> set:
        try:
            arr = self.active_template.get("append_columns", [])
            if isinstance(arr, list):
                return set(str(x) for x in arr)
        except Exception:
            pass
        return set()

    def _is_append_column(self, col_uid: Optional[str]) -> bool:
        return (col_uid is not None) and (col_uid in self._append_columns_set())

    def _current_headers_list(self) -> list[str]:
        # Try multiple tksheet APIs; fall back to our cached headers
        try:
            h = self.sheet.headers()
            if isinstance(h, (list, tuple)):
                return [str(x) for x in h]
        except Exception:
            pass
        try:
            return [str(x) for x in self.sheet.headers]
        except Exception:
            pass
        return list(self.headers)

    def _persist_column_order_to_template(self):
        """Persist current sheet header texts and column order back to the active template,
        preserving UIDs and mappings even after renames."""
        if not self.active_template:
            return

        # What the user currently sees in the grid (order + new names)
        hdrs = self._current_headers_list()

        # Reload templates.json so we don't overwrite someone else's changes
        self._reload_templates_cache()

        for t in self._templates_cache.get("templates", []):
            if t.get("id") != self.active_template.get("id"):
                continue

            old_cols = list(t.get("columns", []))  # [{'uid', 'header'}, ...]
            used_idx = set()

            def take_by_header(h):
                for i, col in enumerate(old_cols):
                    if i in used_idx:
                        continue
                    if str(col.get("header", "")) == h:
                        used_idx.add(i)
                        return dict(col)  # copy
                return None

            def take_by_pos(pos):
                if 0 <= pos < len(old_cols) and pos not in used_idx:
                    used_idx.add(pos)
                    return dict(old_cols[pos])  # copy
                return None

            new_cols = []
            for pos, h in enumerate(hdrs):
                col = take_by_header(h)
                if col is None:
                    # If header text changed (rename), keep the same UID by position
                    col = take_by_pos(pos)
                    if col is None:
                        # Truly a new column that wasn't in template yet
                        col = {"uid": _new_uuid(), "header": h}
                    else:
                        col["header"] = h  # update name
                else:
                    # Found by header; still make sure header text is exact
                    col["header"] = h
                new_cols.append(col)

            t["columns"] = new_cols
            break

        _write_templates_file(self._templates_cache)
        # Refresh our live copy
        self.active_template = _active_template_record()


    def _active_mapping(self) -> dict:
        return self.active_template.get("mapping", {}) if self.active_template else {}
    
    # --- Helpers for date columns ---
    def _date_columns_set(self) -> set:
        """Return set of column UIDs configured as date columns for the active template."""
        try:
            arr = self.active_template.get("date_columns", [])
            if isinstance(arr, list):
                return set(str(x) for x in arr)
        except Exception:
            pass
        return set()

    def _is_date_column(self, col_uid: Optional[str], header_label: str) -> bool:
        """
        True if column is marked as a date column in the active template.
        Legacy fallback: if the template has NO 'date_columns' field yet,
        treat header 'Дата' as a date column to preserve old behavior.
        """
        # If template explicitly has 'date_columns', use it (even if empty)
        if "date_columns" in (self.active_template or {}):
            return (col_uid is not None) and (col_uid in self._date_columns_set())
        # Legacy fallback for older templates
        return (header_label or "").strip().lower() == "дата"


    def _column_uid_for_index(self, col_index: int) -> Optional[str]:
        try:
            return self.active_template["columns"][col_index]["uid"]
        except Exception:
            return None

    def _lists_terms_for_column_uid(self, col_uid: str) -> List[str]:
        if not col_uid:
            return []
        mapping = self._active_mapping()
        list_ids = mapping.get(col_uid, [])
        out: List[str] = []
        lists = GLOSSARY_LISTS.get("lists", {})
        for lid in list_ids:
            lst = lists.get(lid)
            if not lst:
                continue
            terms = lst.get("terms", [])
            out.extend([str(x) for x in terms if isinstance(x, str) and x.strip()])
        # de-dup while preserving order
        seen = set(); uniq = []
        for t in out:
            k = t.lower()
            if k in seen:
                continue
            seen.add(k); uniq.append(t)
        return uniq

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

    # --- Persist column widths per template ---
    def _persist_column_widths_to_template(self):
        widths = None
        try:
            if hasattr(self.sheet, "get_column_widths"):
                widths = self.sheet.get_column_widths()
            elif hasattr(self.sheet, "column_widths"):
                widths = list(self.sheet.column_widths)
        except Exception:
            return
        if self.active_template and widths is not None:
            self._reload_templates_cache()
            for t in self._templates_cache.get("templates", []):
                if t.get("id") == self.active_template.get("id"):
                    t["column_widths"] = widths
                    break
            _write_templates_file(self._templates_cache)

    def reset_column_widths(self):
        # reset for active template only
        if not self.active_template:
            return
        self._reload_templates_cache()
        for t in self._templates_cache.get("templates", []):
            if t.get("id") == self.active_template.get("id"):
                t["column_widths"] = None
                break
        _write_templates_file(self._templates_cache)

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
        self._attach_sheet_bindings_safely()
        self._bind_sheet_events()
        self._install_cross_layout_shortcuts(force=True)
        self.sheet.set_sheet_data(data)
        self._try_style()

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

    def add_column(self):
        name = simpledialog.askstring("Add Column", "Header for the new column:")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self._current_headers_list():
            messagebox.showerror("Add Column", f"A column named “{name}” already exists.")
            return

        # Update template structure
        if self.active_template:
            self._reload_templates_cache()
            for t in self._templates_cache.get("templates", []):
                if t.get("id") == self.active_template.get("id"):
                    t.setdefault("columns", []).append({"uid": _new_uuid(), "header": name})
                    # Extend widths list if present
                    cw = t.get("column_widths")
                    if isinstance(cw, list):
                        cw.append((cw[-1] if cw else 120))
                    break
            _write_templates_file(self._templates_cache)
            self.active_template = _active_template_record()

        # Update visible sheet
        data = self._get_sheet_data_copy()
        for row in data:
            row.append("")
        new_headers = self._current_headers_list() + [name]
        self._rebuild_sheet(new_headers, data)
        self._persist_column_order_to_template()
        self._update_template_status()


    def add_text_to_cell(self, text):
        selected = self.sheet.get_selected_cells()
        if not selected:
            return
        row, col = list(selected)[0]
        header_label = self.headers[col]
        col_uid = self._column_uid_for_index(col)

        incoming = (text or "").strip()
        if not incoming:
            return

        # Decide terms for this column (mapped lists)
        terms = self._lists_terms_for_column_uid(col_uid)

        # Processed value (date normalization OR glossary correction + optional name cleanup)
        processed = None

        if self._is_date_column(col_uid, header_label):
            value = normalize_date(incoming)
            if value is None:
                try: self.root.bell()
                except Exception: pass
                self.preview_label.config(text="🎤 Preview: (дата не распознана — повторите)")
                return
            processed = value
            
        else:
            # Build synthetic glossary key for caching
            key = f"col:{col_uid or col}"
            GLOSSARIES[key] = terms  # dynamic supply of terms per column
            value = correct_text_for_column(incoming, key)
            if header_label in NAME_COLUMNS:
                value = clean_person_field(value)
            processed = value
            if REMOVE_PUNCT:
                processed = _remove_punct_keep_hyphen(processed)

        processed = (processed or "").strip()
        if not processed:
            return

        if self._is_append_column(col_uid):
            try:
                existing = self.sheet.get_cell_data(row, col) or ""
            except Exception:
                existing = ""
            existing = str(existing).strip()
            new_value = (existing + " " + processed).strip() if existing else processed
            self.sheet.set_cell_data(row, col, new_value)
        else:
            self.sheet.set_cell_data(row, col, processed)



    def open_glossary_editor(self):
        # Persist + reload so the editor shows fresh header names
        try:
            self._persist_column_order_to_template()
            self._reload_active_template_from_disk()
        except Exception as e:
            print("open_glossary sync warn:", e)

        def _on_glossary_saved():
            load_glossary_lists()
            self._reload_active_template_from_disk()
            _clear_best_caches()
            GLOSSARIES.clear()
            self._flash_preview_note("📚 Glossary & mapping reloaded", ms=1200)

        win = GlossaryListsAndMappingDialog(
            self.root,
            active_template=self.active_template,
            on_saved=_on_glossary_saved
        )
        try:
            enable_crisp_dark_mode(win, dark=True, delay_ms=0)
        except Exception:
            pass

        try: _fit_to_screen(win, margin=60)
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

    # ===== Cross-layout Ctrl shortcuts (works for RU/UK layouts) =====
    def _install_cross_layout_shortcuts(self, force: bool = False):
        try:
            tag = "HotkeyTag"

            # Attach our tag to every widget that can get focus inside tksheet
            candidates = []
            try:
                candidates.append(self.sheet)
            except Exception:
                pass
            for name in ("MT", "main_table", "top_left", "row_index", "col_index"):
                w = getattr(self.sheet, name, None)
                if w and hasattr(w, "winfo_exists") and w.winfo_exists():
                    candidates.append(w)

            for w in candidates:
                try:
                    tags = list(w.bindtags())
                    if tag in tags:
                        if force:
                            tags.remove(tag)
                        else:
                            continue
                    tags.insert(0, tag)  # run before tksheet
                    w.bindtags(tuple(tags))
                except Exception:
                    pass

            # Bind the tag handlers once
            self.root.bind_class(tag, "<KeyPress>",   self._hotkey_interceptor, add="+")
            self.root.bind_class(tag, "<KeyRelease>", self._hotkey_interceptor, add="+")
        except Exception as e:
            print("hotkeys install error:", e)

        # Track the in-cell editor, which is created/destroyed dynamically
        self._ensure_editor_hotkey_tag()

    def _ensure_editor_hotkey_tag(self):
        try:
            tag = "HotkeyTag"
            mt = getattr(self.sheet, "MT", None) or getattr(self.sheet, "main_table", None)
            ed = getattr(mt, "text_editor", None)
            if ed and ed.winfo_exists():
                tags = list(ed.bindtags())
                if tag not in tags:
                    tags.insert(0, tag)
                    ed.bindtags(tuple(tags))
        except Exception:
            pass
        finally:
            try:
                self.root.after(200, self._ensure_editor_hotkey_tag)
            except Exception:
                pass

    def _ctrl_os_down(self) -> bool:
        try:
            import sys
            if sys.platform.startswith("win"):
                import ctypes
                VK_CONTROL  = 0x11
                VK_LCONTROL = 0xA2
                VK_RCONTROL = 0xA3
                GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
                return bool((GetAsyncKeyState(VK_CONTROL)  & 0x8000) or
                            (GetAsyncKeyState(VK_LCONTROL) & 0x8000) or
                            (GetAsyncKeyState(VK_RCONTROL) & 0x8000))
            # Non-Windows: could maintain a flag from Control press if needed
            return False
        except Exception:
            return False

    def _hotkey_interceptor(self, event):
        # Re-entrancy guard: prevents recursive re-entry from any indirect calls
        if getattr(self, "_hk_in_dispatch", False):
            return

        try:
            et = getattr(event, "type", None)
            # accept numeric and textual KeyPress
            if et not in (2, "2", "KeyPress"):
                return

            # If Control is not physically down (OS-level on Windows), ignore
            if not self._ctrl_os_down():
                return

            code = int(getattr(event, "keycode", 0) or 0)
            action = {
                65: "select_all",  # A
                67: "copy",        # C
                86: "paste",       # V
                88: "cut",         # X
                89: "redo",        # Y
                90: "undo",        # Z
            }.get(code)
            if not action:
                return

            self._hk_in_dispatch = True
            try:
                w = self.root.focus_get()
                handled = False
                if isinstance(w, tk.Text):
                    handled = self._dispatch_text_hotkey(w, action)
                elif isinstance(w, tk.Entry):
                    handled = self._dispatch_entry_hotkey(w, action)
                else:
                    handled = self._dispatch_sheet_hotkey(action)

                if handled:
                    return "break"
            finally:
                self._hk_in_dispatch = False

        except Exception as e:
            # keep app alive; show once if you want
            print("hotkey interceptor error:", e)

    def _dispatch_entry_hotkey(self, entry: tk.Entry, action: str) -> bool:
        try:
            if action == "select_all":
                entry.select_range(0, "end")
                entry.icursor("end")
                return True
            ev = {
                "copy":  "<<Copy>>",
                "paste": "<<Paste>>",
                "cut":   "<<Cut>>",
                "undo":  "<<Undo>>",
                "redo":  "<<Redo>>",
            }.get(action)
            if ev:
                entry.event_generate(ev)
                return True
        except Exception:
            pass
        return False

    def _dispatch_text_hotkey(self, text: tk.Text, action: str) -> bool:
        try:
            if action == "select_all":
                text.tag_add("sel", "1.0", "end-1c")
                text.mark_set("insert", "end-1c")
                text.see("insert")
                return True
            ev = {
                "copy":  "<<Copy>>",
                "paste": "<<Paste>>",
                "cut":   "<<Cut>>",
                "undo":  "<<Undo>>",
                "redo":  "<<Redo>>",
            }.get(action)
            if ev:
                text.event_generate(ev)
                return True
        except Exception:
            pass
        return False

    def _dispatch_sheet_hotkey(self, action: str) -> bool:
        """Dispatch Ctrl+X/C/V/Z/Y/A for the tksheet grid without generating KeyPress events."""
        try:
            mt = getattr(self.sheet, "MT", None) or getattr(self.sheet, "main_table", None) or self.sheet

            # Preferred: call tksheet handlers / public APIs directly (no KeyPress events)
            method_map = {
                "copy":       ("ctrl_c", "copy", "copy_to_clipboard"),
                "paste":      ("ctrl_v", "paste", "paste_from_clipboard"),
                "cut":        ("ctrl_x", "cut"),
                "undo":       ("ctrl_z", "undo"),
                "redo":       ("ctrl_y", "redo"),
                "select_all": ("ctrl_a", "select_all"),
            }
            targets = [mt, self.sheet]
            for target in targets:
                for mname in method_map[action]:
                    fn = getattr(target, mname, None)
                    if callable(fn):
                        try:
                            fn(None)   # many ctrl_* handlers accept an event; None is fine
                        except TypeError:
                            fn()      # some public methods take no args
                        return True

            # As a last resort (shouldn’t be needed), you could implement your own clipboard copy/paste here.
            # But on normal tksheet versions at least one method above exists.
        except Exception as e:
            print("sheet hotkey error:", e)
        return False

    def _current_settings(self) -> dict:
        return {
            "language": LANGUAGE if LANGUAGE else "auto",
            "autosave_minutes": max(1, AUTOSAVE_EVERY_MS // 60000),
            "ui_scale": UI_SCALE,
            "model_key": MODEL_KEY or _pick_default_model_key(),
            "speed_mode": SPEED_MODE,
            "vad_strictness": VAD_STRICTNESS,
            "glossary_strictness": GLOSSARY_STRICTNESS,
            "remove_punct": REMOVE_PUNCT
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
            requested_model_key = s.get("model_key", prev["model_key"])
            model_changed = (requested_model_key != prev["model_key"])

            _apply_settings_to_globals(s)
            _save_settings_file(self._current_settings())

            if float(prev.get("ui_scale", 1.0)) != float(s.get("ui_scale", prev.get("ui_scale", 1.0))):
                _apply_dark_ui(self.root, dark=True)
                self._refresh_preview_font()
                self._maybe_update_sheet_zoom()
                self._style_action_buttons()

            if model_changed:
                if self.is_listening:
                    self.stop_listening()
                self._show_loading("Loading model, please wait…")
                try:
                    _load_whisper_model(MODEL_KEY)
                    _warmup_model()
                finally:
                    self._hide_loading()
            try:
                if int(prev.get("glossary_strictness", 3)) != int(s.get("glossary_strictness", prev.get("glossary_strictness", 3))):
                    _clear_best_caches()
            except Exception:
                pass

            # Apply per-template date columns if provided from Settings
            if isinstance(s.get("date_columns", None), list) and self.active_template:
                try:
                    self._reload_templates_cache()
                    for t in self._templates_cache.get("templates", []):
                        if t.get("id") == self.active_template.get("id"):
                            t["date_columns"] = [str(x) for x in s["date_columns"]]
                            break
                    _write_templates_file(self._templates_cache)
                    self._reload_active_template_from_disk()
                    self._flash_preview_note("📅 Date columns updated", ms=1200)
                except Exception as e:
                    print("apply_settings(date_columns) error:", e)

            # Apply per-template append columns if provided from Settings
            if isinstance(s.get("append_columns", None), list) and self.active_template:
                try:
                    self._reload_templates_cache()
                    for t in self._templates_cache.get("templates", []):
                        if t.get("id") == self.active_template.get("id"):
                            t["append_columns"] = [str(x) for x in s["append_columns"]]
                            break
                    _write_templates_file(self._templates_cache)
                    self._reload_active_template_from_disk()
                    self._flash_preview_note("➕ Append columns updated", ms=1200)
                except Exception as e:
                    print("apply_settings(append_columns) error:", e)

            self._start_autosave()
        except Exception as e:
            print("apply_settings error:", e)
            messagebox.showerror("Settings", f"Failed to apply settings:\n{e}")
        finally:
            self._settings_applying = False

    def open_settings(self):
        # Make sure latest headers are persisted & reloaded
        try:
            self._persist_column_order_to_template()
            self._reload_active_template_from_disk()
        except Exception as e:
            print("open_settings sync warn:", e)

        SettingsDialog(
            self.root,
            initial=self._current_settings(),
            on_apply=self.apply_settings,
            on_reset_widths=self.reset_column_widths,
            active_template=self.active_template,
        )

    # Listening
    def start_listening(self):
        global LAST_ACTIVITY_TS
        self.reset_audio_state()
        LAST_ACTIVITY_TS = time.monotonic()
        self._listening_started_ts = LAST_ACTIVITY_TS

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
        if mono_1d.size == 0:
            return
        self._tail_chunks.append(mono_1d)
        self._tail_total_samples += mono_1d.shape[0]
        limit = self._tail_limit_samples()
        while self._tail_total_samples > limit and self._tail_chunks:
            excess = self._tail_total_samples - limit
            left = self._tail_chunks[0]
            if left.shape[0] <= excess:
                self._tail_total_samples -= left.shape[0]
                self._tail_chunks.popleft()
            else:
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

            got_any = False
            while not AUDIO_QUEUE.empty():
                blk = AUDIO_QUEUE.get()
                got_any = True
                self._commit_chunks.append(blk)
                self._commit_total_samples += len(blk)
                mono = blk[:, 0] if getattr(blk, "ndim", 0) > 1 else blk
                mono = np.asarray(mono, dtype=np.float32, order="C")
                self._push_tail(mono)

            if got_any:
                last_audio_time = time.monotonic()

            now = time.monotonic()
            tail_needed = int(_preview_tail_sec() * SAMPLERATE * 0.6)
            if self._tail_total_samples >= tail_needed and (now - self._last_preview_decode_ts) >= PREVIEW_MIN_INTERVAL_SEC:
                tail = self._concat_tail()
                if rms_db(tail.flatten()) < (_energy_gate_db() + 2.0):
                    preview_text = ""
                else:
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

            timeout = self._commit_total_samples >= int(BLOCK_DURATION * SAMPLERATE)

            if self._tail_total_samples > 0:
                tail = self._concat_tail()
                tail_len_samples = int(COMMIT_TAIL_SILENCE_SEC * SAMPLERATE)
                tail_sil = tail[-tail_len_samples:] if tail.shape[0] >= tail_len_samples else tail
                tail_is_silence = (
                    tail_sil is not None
                    and tail_sil.shape[0] >= int(COMMIT_MIN_SILENCE_SEC * SAMPLERATE)
                    and is_silence(tail_sil, threshold_db=_energy_gate_db())
                )
            else:
                tail = None
                tail_sil = None
                tail_is_silence = False

            if (time.monotonic() - last_audio_time) > 60.0:
                self.root.after(0, lambda: self.preview_label.config(text="Preview: (auto-stopped after inactivity)"))
                self.stop_listening()
                return

            if tail_is_silence or timeout:
                if not self._commit_chunks:
                    self._tail_chunks.clear()
                    self._tail_total_samples = 0
                    time.sleep(0.03)
                    continue

                buf = np.concatenate(self._commit_chunks, axis=0)
                full_text = transcribe_buffer_commit(buf)
                full_text = strip_trailing_dot(full_text)

                if not full_text:
                    recent_preview = (time.monotonic() - getattr(self, "_last_preview_ts", 0.0)) < 3.0
                    if recent_preview and getattr(self, "_last_preview_text", "") and tail is not None:
                        try:
                            alt = transcribe_buffer(tail)
                            alt = strip_trailing_dot(alt)
                            if alt and not looks_like_outro(alt):
                                full_text = alt
                        except Exception:
                            pass

                if full_text and not looks_like_outro(full_text):
                    self.root.after(0, self.add_text_to_cell, full_text)
                    if full_text != self._last_preview_text:
                        self._last_preview_text = full_text
                        self._last_preview_ts = time.monotonic()
                        self.root.after(0, lambda t=full_text: self.preview_label.config(text="Preview: " + t))

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

                self._commit_chunks.clear()
                self._commit_total_samples = 0
                self._tail_chunks.clear()
                self._tail_total_samples = 0

            time.sleep(0.02)

    def reset_audio_state(self):
        try:
            while not AUDIO_QUEUE.empty():
                AUDIO_QUEUE.get_nowait()
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

    # Silence-guard timer
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

    def _detach_console_if_any(self):
        if sys.platform.startswith("win"):
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                if kernel32.GetConsoleWindow():
                    kernel32.FreeConsole()
            except Exception:
                pass

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
        if not self.active_template:
            self._persist_column_order_to_template()
            data = self.sheet.get_sheet_data()
            df = pd.DataFrame(data, columns=self.headers)
            df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig", na_rep="")
            return
        path = _data_path_for_template(self.active_template["id"])
        data = self.sheet.get_sheet_data()
        df = pd.DataFrame(data, columns=self.headers)
        df.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")

    def load_data(self):
        if not self.active_template:
            if os.path.exists(DATA_FILE):
                df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", keep_default_na=False)
                df = df.fillna("")
                if list(df.columns) != self.headers and len(df.columns) == len(self.headers):
                    df.columns = self.headers
                self.sheet.set_sheet_data(df.values.tolist())
            else:
                self.add_rows(20)
            return

        path = _data_path_for_template(self.active_template["id"])
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
            df = df.fillna("")
            # align headers
            if list(df.columns) != self.headers and len(df.columns) == len(self.headers):
                df.columns = self.headers
            self.sheet.set_sheet_data(df.values.tolist())
        else:
            self.add_rows(20)

    def save_column_widths(self):
        self._persist_column_widths_to_template()
        self._persist_column_order_to_template()

    def load_column_widths(self):
        # apply template-specific widths
        try:
            self._reload_templates_cache()
            if not self.active_template:
                return
            for t in self._templates_cache.get("templates", []):
                if t.get("id") == self.active_template.get("id"):
                    widths = t.get("column_widths")
                    if widths:
                        self.apply_column_widths(widths)
                    break
        except Exception as e:
            print("load_column_widths error:", e)

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
            self._persist_column_order_to_template()
            self.save_data()
            self.save_column_widths()
            self._save_main_window_geometry()
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
        except Exception: 
            pass
        try: _fit_to_screen(self._loading, margin=120)
        except Exception: pass
        self._loading.update()

    def _hide_loading(self):
        try:
            if hasattr(self, "_loading") and self._loading.winfo_exists():
                self._loading.destroy()
        except Exception:
            pass

    def _try_style(self):
        try:
            _try_style_tksheet(self.root, dark=True)
        except Exception:
            pass

    # Template Manager dialog
    def open_template_manager(self):
        TemplateManagerDialog(
            self.root,
            on_open=self._set_active_template,
            before_duplicate=self._save_current_state_before_duplicate
        )


# ===========================
# Glossary Lists + Mapping Dialog
# ===========================
class GlossaryListsAndMappingDialog(tk.Toplevel):
    """
    Glossary lists + per-list column assignment (checkbox UI, similar to Date mapping).
    - Left: Lists (add/rename/delete)
    - Middle: Terms for selected list (add/remove)
    - Right: Checklist of all columns in the active template. Checking a box means:
             "apply the selected list to this column".
    """
    def __init__(self, master, active_template: dict, on_saved=None):
        super().__init__(master)
        self.title("Glossary Lists & Mapping")
        self.geometry("980x580")
        self.minsize(900, 540)
        self.transient(master)

        self.active_template = active_template or {}
        # Stage mapping edits here; commit on Save / Save & Close
        base_map = self.active_template.get("mapping", {}) or {}
        self._mapping_work = {str(k): list(v) for k, v in base_map.items()}

        self.on_saved = on_saved or (lambda: None)

        c = _set_palette(True)
        self.configure(bg=c["bg"])

        # Topbar
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

        btn_save = tk.Button(topbar, text="💾 Save", command=self._save_all)
        btn_close = tk.Button(topbar, text="Save & Close", command=self._on_close)
        _style_btn(btn_save); btn_save.pack(side="left", padx=4, pady=6)
        _style_btn(btn_close); btn_close.pack(side="right", padx=4, pady=6)

        # Main layout: three columns
        wrap = tk.Frame(self, bg=c["bg"])
        wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Left: Lists
        left = tk.Frame(wrap, bg=c["bg"], width=280); left.pack(side="left", fill="y", padx=(0,10))
        tk.Label(left, text="Lists", bg=c["bg"], fg=c["text"]).pack(anchor="w")
        self.lb_lists = tk.Listbox(left, exportselection=False, bg=c["surface"], fg=c["text"],
                                   selectbackground=c["sel_bg"], selectforeground=c["sel_fg"],
                                   highlightthickness=1, highlightbackground=c["border"], relief="flat")
        self.lb_lists.pack(fill="both", expand=True)
        btns_l = tk.Frame(left, bg=c["bg"]); btns_l.pack(fill="x", pady=(6,0))

        b_add_l = tk.Button(btns_l, text="➕ Add list", command=self._add_list)
        b_ren_l = tk.Button(btns_l, text="✏ Rename", command=self._rename_list)
        b_del_l = tk.Button(btns_l, text="🗑 Delete", command=self._delete_list)
        for b in (b_add_l, b_ren_l, b_del_l):
            _style_btn(b); b.pack(side="left", padx=3)

        # Middle: Terms for selected list
        mid = tk.Frame(wrap, bg=c["bg"], width=360); mid.pack(side="left", fill="both", expand=True, padx=(0,10))
        tk.Label(mid, text="Terms of selected list", bg=c["bg"], fg=c["text"]).pack(anchor="w")
        self.lb_terms = tk.Listbox(mid, exportselection=False, bg=c["surface"], fg=c["text"],
                                   selectbackground=c["sel_bg"], selectforeground=c["sel_fg"],
                                   highlightthickness=1, highlightbackground=c["border"], relief="flat")
        self.lb_terms.pack(fill="both", expand=True)
        bot_term = tk.Frame(mid, bg=c["bg"]); bot_term.pack(fill="x", pady=(6,0))
        self.term_entry = tk.Entry(bot_term, relief="flat", bg=c["surface"], fg=c["text"])
        self.term_entry.pack(side="left", fill="x", expand=True, padx=(0,6))
        self.term_entry.bind("<Return>", self._add_term)      # main Enter
        self.term_entry.bind("<KP_Enter>", self._add_term)    # keypad Enter
        b_add_t = tk.Button(bot_term, text="Add term", command=self._add_term)
        b_del_t = tk.Button(bot_term, text="Remove term", command=self._remove_term)
        for b in (b_add_t, b_del_t):
            _style_btn(b); b.pack(side="left", padx=3)

        # Right: Per-list column assignment (checkbox UI)
        right = tk.Frame(wrap, bg=c["bg"], width=320); right.pack(side="left", fill="both")
        tk.Label(right, text="Apply selected list to columns", bg=c["bg"], fg=c["text"]).pack(anchor="w")

        # Scrollable checklist
        sc = tk.Frame(right, bg=c["bg"]); sc.pack(fill="both", expand=True, pady=(4,6))
        self._col_canvas = tk.Canvas(sc, bg=c["bg"], highlightthickness=0)
        self._col_vsb = tk.Scrollbar(sc, orient="vertical", command=self._col_canvas.yview)
        self._col_canvas.configure(yscrollcommand=self._col_vsb.set)
        self._col_vsb.pack(side="right", fill="y")
        self._col_canvas.pack(side="left", fill="both", expand=True)

        self._col_inner = tk.Frame(self._col_canvas, bg=c["bg"])
        self._col_win = self._col_canvas.create_window((0, 0), window=self._col_inner, anchor="nw")

        def _on_cfg(_=None):
            self._col_inner.update_idletasks()
            bbox = self._col_canvas.bbox(self._col_win)
            if bbox:
                self._col_canvas.configure(scrollregion=bbox)
            try:
                self._col_canvas.itemconfigure(self._col_win, width=self._col_canvas.winfo_width())
            except Exception:
                pass
        self._col_inner.bind("<Configure>", _on_cfg)
        self._col_canvas.bind("<Configure>", _on_cfg)

        # Bottom row: select/clear all for the current list
        right_btns = tk.Frame(right, bg=c["bg"]); right_btns.pack(fill="x")
        #self.b_sel_all_cols = tk.Button(right_btns, text="Select all columns", command=self._select_all_for_current_list)
        self.b_clr_all_cols = tk.Button(right_btns, text="Clear all columns", command=self._clear_all_for_current_list)
        #_style_btn(self.b_sel_all_cols); _style_btn(self.b_clr_all_cols)
        #self.b_sel_all_cols.pack(side="left")
        self.b_clr_all_cols.pack(side="right", padx=(6,0))

        # Data caches
        self._cols_sorted = [(col["uid"], col["header"]) for col in self.active_template.get("columns", [])]
        self._col_vars: dict[str, tk.BooleanVar] = {}  # rebuilt per selected list

        # Wire events
        self.lb_lists.bind("<<ListboxSelect>>", lambda e: (self._refresh_terms(), self._render_col_checklist_for_list()))

        # Initial load
        self._refresh_lists()
        # If any list exists, pre-render its columns; terms refresh is already called by _refresh_lists.

        try:
            enable_crisp_dark_mode(self, dark=True, delay_ms=0)
        except Exception:
            pass
        try: _fit_to_screen(self, margin=60)
        except Exception: 
            pass

    # ---------- Utilities ----------
    def _lists_dict(self) -> dict:
        return GLOSSARY_LISTS.get("lists", {})

    def _sel_list_id(self) -> Optional[str]:
        sel = self.lb_lists.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx < 0 or idx >= len(getattr(self, "_lists_sorted", [])):
            return None
        return self._lists_sorted[idx][0]

    # ---------- Left: lists ----------
    def _refresh_lists(self):
        self.lb_lists.delete(0, "end")
        lists = self._lists_dict()
        by_name = sorted(
            [(lid, lists[lid].get("name", "")) for lid in lists],
            key=lambda pair: "".join(ch for ch in unicodedata.normalize("NFKD", pair[1]).casefold() if not unicodedata.combining(ch)).replace("ё", "е"),
        )
        self._lists_sorted = by_name
        for _, name in by_name:
            self.lb_lists.insert("end", name)
        if self.lb_lists.size() > 0:
            self.lb_lists.selection_set(0)
            self.lb_lists.activate(0)
        self._refresh_terms()
        self._render_col_checklist_for_list()

    def _add_list(self):
        name = simpledialog.askstring("New list", "List name:")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        lists = self._lists_dict()
        if any(lists[x]["name"].strip().lower() == name.lower() for x in lists):
            messagebox.showinfo("Info", "A list with that name already exists.")
            return
        lid = _new_uuid()
        lists[lid] = {"name": name, "terms": []}
        save_glossary_lists()
        self._refresh_lists()

    def _rename_list(self):
        lid = self._sel_list_id()
        if not lid:
            return
        cur = self._lists_dict()[lid]["name"]
        new = simpledialog.askstring("Rename list", "New name:", initialvalue=cur)
        if not new:
            return
        new = new.strip()
        if not new or new == cur:
            return
        if any(self._lists_dict()[x]["name"].strip().lower() == new.lower() for x in self._lists_dict()):
            messagebox.showinfo("Info", "A list with that name already exists.")
            return
        self._lists_dict()[lid]["name"] = new
        save_glossary_lists()
        self._refresh_lists()

    def _delete_list(self):
        lid = self._sel_list_id()
        if not lid:
            return
        if not messagebox.askyesno("Confirm", "Delete this list and remove it from all column assignments?"):
            return
        # Remove from lists
        self._lists_dict().pop(lid, None)
        save_glossary_lists()
        # Remove from staged mapping
        for col_uid in list(self._mapping_work.keys()):
            arr = [x for x in self._mapping_work[col_uid] if x != lid]
            if arr:
                self._mapping_work[col_uid] = arr
            else:
                self._mapping_work.pop(col_uid, None)
        # Remove from templates.json immediately to keep parity
        data = _ensure_templates_file([])
        for t in data.get("templates", []):
            if t.get("id") == self.active_template.get("id"):
                mp = t.setdefault("mapping", {})
                for k in list(mp.keys()):
                    if lid in mp[k]:
                        mp[k] = [x for x in mp[k] if x != lid]
                break
        _write_templates_file(data)
        # Reload template in dialog and refresh UI
        self.active_template = _active_template_record() or self.active_template
        self._cols_sorted = [(col["uid"], col["header"]) for col in self.active_template.get("columns", [])]
        self._refresh_lists()

    # ---------- Middle: terms ----------
    def _refresh_terms(self):
        self.lb_terms.delete(0, "end")
        lid = self._sel_list_id()
        if not lid:
            return

        terms = list(self._lists_dict().get(lid, {}).get("terms", []))

        # Sort A→Z, case/diacritics insensitive, and normalize "ё"→"е"
        terms.sort(key=lambda s: "".join(
            ch for ch in unicodedata.normalize("NFKD", str(s or "")).casefold()
            if not unicodedata.combining(ch)
        ).replace("ё", "е"))

        for t in terms:
            self.lb_terms.insert("end", t)

    def _add_term(self, event=None):  # ← event is optional; buttons still call it without args
        lid = self._sel_list_id()
        if not lid:
            return
        term = self.term_entry.get().strip()
        if not term:
            return

        arr = self._lists_dict().setdefault(lid, {}).setdefault("terms", [])

        if any(t.strip().lower() == term.lower() for t in arr):
            messagebox.showinfo("Info", "This term already exists in the list.")
            # keep focus in the entry for quick editing
            self.term_entry.focus_set()
            # prevent default key event propagation when called via Enter
            return "break"

        arr.append(term)

        # keep underlying list sorted (same sorter you already use in _refresh_terms)
        import unicodedata
        arr.sort(key=lambda s: "".join(
            ch for ch in unicodedata.normalize("NFKD", str(s or "")).casefold()
            if not unicodedata.combining(ch)
        ).replace("ё", "е"))

        self.term_entry.delete(0, "end")
        self.term_entry.focus_set()

        save_glossary_lists()
        self._refresh_terms()

        # Select and make visible the just-added term
        try:
            idx = next(i for i, t in enumerate(arr) if t == term)
            self.lb_terms.selection_clear(0, "end")
            self.lb_terms.selection_set(idx)
            self.lb_terms.see(idx)
        except Exception:
            pass

        # stop default handling of the Return key when invoked via binding
        return "break"

    def _remove_term(self):
        lid = self._sel_list_id()
        if not lid:
            return
        sel = self.lb_terms.curselection()
        if not sel:
            return
        term = self.lb_terms.get(sel[0])
        arr = self._lists_dict().get(lid, {}).get("terms", [])
        new_terms = [t for t in arr if t != term]
        new_terms.sort(key=lambda s: "".join(
            ch for ch in unicodedata.normalize("NFKD", str(s or "")).casefold()
            if not unicodedata.combining(ch)
        ).replace("ё", "е"))
        self._lists_dict()[lid]["terms"] = new_terms
        save_glossary_lists()
        self._refresh_terms()

    # ---------- Right: per-list column assignment ----------
    def _render_col_checklist_for_list(self):
        # Clear old checkboxes
        for w in self._col_inner.winfo_children():
            w.destroy()
        self._col_vars.clear()

        lid = self._sel_list_id()
        c = _set_palette(True)

        if not lid:
            tk.Label(self._col_inner, text="Select a list to assign it to columns.",
                     bg=c["bg"], fg=c["muted"]).pack(anchor="w", padx=2, pady=2)
            self._toggle_assign_buttons_state(disabled=True)
            return

        self._toggle_assign_buttons_state(disabled=False)

        # Build checkboxes (sorted by header)
        for uid, header in self._cols_sorted:
            v = tk.BooleanVar()
            # Pre-check if this list is currently mapped to the column (staged mapping)
            mapped_ids = set(self._mapping_work.get(uid, []))
            v.set(lid in mapped_ids)

            def _mk_cb(u=uid, var=v):
                return lambda *_: self._toggle_col_for_list(u, var)
            chk = tk.Checkbutton(self._col_inner, text=header, variable=v,
                                 command=_mk_cb(),
                                 bg=c["bg"], fg=c["text"],
                                 activebackground=c["bg"], activeforeground=c["text"],
                                 selectcolor=c.get("surface", "#151a21"), anchor="w")
            indent = max(6, int(round(10 * UI_SCALE)))  # scales with your UI_SCALE
            chk.pack(fill="x", anchor="w", padx=(indent, 0), pady=(5, 0))
            self._col_vars[uid] = v

        # Update scrollregion
        try:
            self._col_inner.event_generate("<Configure>")
        except Exception:
            pass

    def _toggle_assign_buttons_state(self, disabled: bool):
        state = tk.DISABLED if disabled else tk.NORMAL
        try:
            #self.b_sel_all_cols.configure(state=state)
            self.b_clr_all_cols.configure(state=state)
        except Exception:
            pass

    def _toggle_col_for_list(self, col_uid: str, var: tk.BooleanVar):
        """Update staged mapping when a checkbox is toggled."""
        lid = self._sel_list_id()
        if not lid:
            return
        arr = list(self._mapping_work.get(col_uid, []))
        if var.get():
            if lid not in arr:
                arr.append(lid)
            self._mapping_work[col_uid] = arr
        else:
            arr = [x for x in arr if x != lid]
            if arr:
                self._mapping_work[col_uid] = arr
            else:
                self._mapping_work.pop(col_uid, None)

    def _select_all_for_current_list(self):
        lid = self._sel_list_id()
        if not lid:
            return
        for uid in [u for (u, _) in self._cols_sorted]:
            self._col_vars[uid].set(True)
            arr = list(self._mapping_work.get(uid, []))
            if lid not in arr:
                arr.append(lid)
            self._mapping_work[uid] = arr

    def _clear_all_for_current_list(self):
        lid = self._sel_list_id()
        if not lid:
            return
        for uid in [u for (u, _) in self._cols_sorted]:
            self._col_vars[uid].set(False)
            arr = [x for x in self._mapping_work.get(uid, []) if x != lid]
            if arr:
                self._mapping_work[uid] = arr
            else:
                self._mapping_work.pop(uid, None)

    # ---------- Save / Close ----------
    def _commit_mapping_to_disk(self):
        """Write staged mapping for this active template to templates.json."""
        data = _ensure_templates_file([])
        tid = (self.active_template or {}).get("id")
        if not tid:
            return
        # Clean empty arrays
        clean_map = {}
        for k, v in self._mapping_work.items():
            vv = [str(x) for x in v if str(x).strip()]
            if vv:
                clean_map[str(k)] = vv
        for t in data.get("templates", []):
            if t.get("id") == tid:
                t["mapping"] = clean_map
                break
        _write_templates_file(data)
        # Refresh dialog copy
        self.active_template = _active_template_record() or self.active_template

    def _save_all(self):
        try:
            self._commit_mapping_to_disk()
            self.on_saved()
            messagebox.showinfo("Saved", "Glossary lists and column assignments saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def _on_close(self):
        try:
            self._commit_mapping_to_disk()
            self.on_saved()
        finally:
            self.destroy()



# ===========================
# Template Manager Dialog
# ===========================
class TemplateManagerDialog(tk.Toplevel):
    def __init__(self, master, on_open: Callable[[str], None], before_duplicate: Optional[Callable[[], None]] = None,):
        super().__init__(master)
        self.title("Templates")
        self.geometry("820x520")
        self.minsize(780, 480)
        self.transient(master)
        self.on_open = on_open
        self.before_duplicate = before_duplicate

        c = _set_palette(True)
        self.configure(bg=c["bg"])

        top = tk.Frame(self, bg=c["surface"]); top.pack(side="top", fill="x", padx=10, pady=(10, 6))

        def _style_btn(b):
            try:
                b.configure(bg=c["button_bg"], fg=c["button_fg"],
                            activebackground=c["button_active_bg"], activeforeground=c["button_fg"],
                            relief="flat", highlightthickness=1,
                            highlightbackground=c["button_border"], highlightcolor=c["button_border"])
            except Exception:
                pass

        self.btn_open = tk.Button(top, text="Open", command=self._open_selected)
        self.btn_dup  = tk.Button(top, text="Duplicate", command=self._duplicate)
        self.btn_ren  = tk.Button(top, text="Rename", command=self._rename)
        self.btn_del  = tk.Button(top, text="Delete", command=self._delete)
        self.btn_imp  = tk.Button(top, text="Import CSV/Excel", command=self._import_template)

        for b in (self.btn_open, self.btn_dup, self.btn_ren, self.btn_del, self.btn_imp):
            _style_btn(b); b.pack(side="left", padx=4, pady=6)

        wrap = tk.Frame(self, bg=c["bg"]); wrap.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.lb = tk.Listbox(wrap, exportselection=False, bg=c["surface"], fg=c["text"],
                             selectbackground=c["sel_bg"], selectforeground=c["sel_fg"],
                             highlightthickness=1, highlightbackground=c["border"], relief="flat")
        self.lb.pack(fill="both", expand=True)

        self._reload()
        self.lb.bind("<Double-Button-1>", lambda e: self._open_selected())

        try:
            enable_crisp_dark_mode(self, dark=True, delay_ms=0)
        except Exception:
            pass
        try: _fit_to_screen(self, margin=60)
        except Exception: 
            pass

    def _reload(self):
        self.lb.delete(0, "end")
        self.data = _ensure_templates_file([])
        tid = self.data.get("last_template_id")
        self.rows = []
        for t in self.data.get("templates", []):
            nm = t.get("name", "(unnamed)")
            ident = t.get("id", "")[:8]
            mark = " (current)" if t.get("id") == tid else ""
            self.rows.append(t["id"])
            self.lb.insert("end", f"{nm} [{ident}]{mark}")

        if self.lb.size() > 0:
            sel_idx = 0
            # try to preselect current
            for i, t in enumerate(self.data.get("templates", [])):
                if t.get("id") == tid:
                    sel_idx = i; break
            self.lb.selection_set(sel_idx)
            self.lb.activate(sel_idx)

    def _sel_template_id(self) -> Optional[str]:
        sel = self.lb.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx < 0 or idx >= len(self.rows):
            return None
        return self.rows[idx]

    def _open_selected(self):
        tid = self._sel_template_id()
        if not tid:
            return
        try:
            self.on_open(tid)
            self._reload()
        except Exception as e:
            messagebox.showerror("Open template", str(e))

    def _duplicate(self):
        if callable(self.before_duplicate):
            self.before_duplicate()
        data = _ensure_templates_file([])
        tid = self._sel_template_id()
        if not tid:
            return
        src = None
        for t in data.get("templates", []):
            if t.get("id") == tid:
                src = t; break
        if not src:
            return
        dup = {
            "id": _new_uuid(),
            "name": (src.get("name","") + " (copy)")[:64] or "Copy",
            "columns": [{"uid": _new_uuid(), "header": c["header"]} for c in src.get("columns", [])],
            "mapping": {},  # do not copy mapping by default (easy to switch if desired)
            "column_widths": src.get("column_widths")
        }
        data["templates"].append(dup)
        _write_templates_file(data)
        # Optional: also duplicate data?
        src_path = _data_path_for_template(src["id"])
        dst_path = _data_path_for_template(dup["id"])
        try:
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                pd.read_csv(src_path, encoding="utf-8-sig", keep_default_na=False).to_csv(dst_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            print("dup data warn:", e)
        self._reload()

    def _rename(self):
        data = _ensure_templates_file([])
        tid = self._sel_template_id()
        if not tid:
            return
        for t in data.get("templates", []):
            if t.get("id") == tid:
                cur = t.get("name","")
                new = simpledialog.askstring("Rename template", "New name:", initialvalue=cur)
                if not new:
                    return
                t["name"] = new.strip() or cur
                break
        _write_templates_file(data)
        self._reload()

    def _delete(self):
        data = _ensure_templates_file([])
        tid = self._sel_template_id()
        if not tid:
            return

        # prevent deleting the current template
        if data.get("last_template_id") == tid:
            messagebox.showwarning(
                "Delete",
                "You cannot delete the currently open template. Open a different one first."
            )
            return

        # first confirmation: remove the template entry
        if not messagebox.askyesno(
            "Delete template",
            "Delete the selected template from the list?\n"
            "(You can choose to delete its data file on disk next.)"
        ):
            return

        # remove the template record from templates.json
        data["templates"] = [t for t in data.get("templates", []) if t.get("id") != tid]
        _write_templates_file(data)

        # optional second step: also delete the dataset file
        path = _data_path_for_template(tid)
        if os.path.exists(path):
            if messagebox.askyesno(
                "Delete data file",
                f"Also delete the dataset file on disk?\n\n{path}"
            ):
                try:
                    try:
                        from send2trash import send2trash  # optional, safer on all OSes
                    except Exception:
                        send2trash = None

                    if send2trash:
                        send2trash(path)
                    else:
                        os.remove(path)
                except Exception as e:
                    messagebox.showerror("Delete data file", f"Failed to delete:\n{e}")

        self._reload()

    def _import_template(self):
        path = filedialog.askopenfilename(
            title="Import CSV/Excel",
            filetypes=[("CSV / Excel", "*.csv *.xlsx *.xls")]
        )
        if not path:
            return
        # Read with pandas
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path, header=0, encoding="utf-8-sig", keep_default_na=False)
            else:
                try:
                    df = pd.read_excel(path, header=0)
                except Exception as e:
                    messagebox.showerror("Excel import", f"Excel import requires 'openpyxl'.\n\n{e}")
                    return
        except Exception as e:
            messagebox.showerror("Import", f"Failed to import file:\n{e}")
            return

        # First row are headers -> already used by header=0
        headers = [str(h) for h in df.columns]
        t = _make_default_template(headers)
        t["name"] = simpledialog.askstring("Template name", "Name for the imported template:", initialvalue=os.path.basename(path)) or os.path.basename(path)
        # Reset mapping for new template (as requested)
        t["mapping"] = {}
        # Save template
        data = _ensure_templates_file([])
        data["templates"].append(t)
        _write_templates_file(data)

        # Ask to import data rows into template's dataset
        if messagebox.askyesno("Import data", "Also import the file rows into this template's dataset?"):
            try:
                df = df.fillna("")
                out = _data_path_for_template(t["id"])
                df.to_csv(out, index=False, encoding="utf-8-sig")
            except Exception as e:
                messagebox.showerror("Import data", f"Failed to save dataset:\n{e}")

        self._reload()

# ===========================
# Settings dialog (unchanged, except reset widths now per-template)
# ===========================
class SettingsDialog(tk.Toplevel):
    def __init__(self, master, initial: dict, 
                 on_apply: Callable[[dict], None], 
                 on_reset_widths: Optional[Callable[[], None]] = None,
                 active_template: Optional[dict] = None):
        super().__init__(master)
        self.title("Settings")
        self.geometry("800x800")
        self.minsize(800, 800)
        self.transient(master)
        self.on_apply = on_apply
        self.on_reset_widths = on_reset_widths

        # store active template on the dialog
        self.active_template = active_template or {"columns": [], "date_columns": []}

        # initialize date columns state for this dialog
        cols = (self.active_template.get("columns") or [])
        self._date_cols_columns = [(str(c.get("uid", "")), str(c.get("header", ""))) for c in cols]

        dc = self.active_template.get("date_columns")
        if isinstance(dc, list):
            selected = [str(x) for x in dc]
        else:
            # legacy default if the template has no date_columns: treat "Дата" as date
            selected = [str(c.get("uid", "")) for c in cols
                        if str(c.get("header", "")).strip().lower() == "дата"]

        self._date_cols_pending = set(selected)

        c = _set_palette(True)
        self.configure(bg=c["bg"])

        # --- Scrollable content area (keeps bottom buttons visible) ---
        outer = tk.Frame(self, bg=c["bg"])
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=c["bg"], highlightthickness=0)
        vbar   = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # holder gives visual margins; frm is your existing grid container
        holder = tk.Frame(canvas, bg=c["bg"])
        frm    = tk.Frame(holder, bg=c["bg"])
        frm.pack(fill="both", expand=True, padx=14, pady=12)  # <<< consistent margins

        _win_id = canvas.create_window((0, 0), window=holder, anchor="nw")

        def _on_cfg(_=None):
            try:
                holder.update_idletasks()
                bbox = canvas.bbox(_win_id)
                if bbox:
                    canvas.configure(scrollregion=bbox)
                # keep inner width equal to canvas width (so the scrollbar sits to the side)
                canvas.itemconfigure(_win_id, width=canvas.winfo_width())
            except Exception:
                pass

        holder.bind("<Configure>", _on_cfg)
        canvas.bind("<Configure>", _on_cfg)

        # Smooth mouse wheel scrolling
        def _bind_wheel(w):
            def _on_mousewheel(e):
                try:
                    if getattr(e, "num", None) == 5 or getattr(e, "delta", 0) < 0:
                        canvas.yview_scroll(1, "units")
                    elif getattr(e, "num", None) == 4 or getattr(e, "delta", 0) > 0:
                        canvas.yview_scroll(-1, "units")
                except Exception:
                    pass
                return "break"
            w.bind_all("<MouseWheel>", _on_mousewheel, add="+")  # Win/macOS
            w.bind_all("<Button-4>",  _on_mousewheel, add="+")   # X11
            w.bind_all("<Button-5>",  _on_mousewheel, add="+")
        _bind_wheel(self)

        # Keep a handle if needed elsewhere
        self._settings_canvas = canvas

        def lab(parent, txt):
            return tk.Label(parent, text=txt, bg=c["bg"], fg=c["text"])
        
        def _style_btn_local(b):
            try:
                c = _set_palette(True)
                b.configure(bg=c["button_bg"], fg=c["button_fg"],
                            activebackground=c["button_active_bg"], activeforeground=c["button_fg"],
                            relief="flat", highlightthickness=1,
                            highlightbackground=c["button_border"], highlightcolor=c["button_border"])
            except Exception:
                pass

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
        cur_model_label = _MODELKEY_TO_LABEL.get(cur_model_key, "Base")
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

        # UI scale (0.50 .. 1.50 step 0.25)
        lab(frm, "UI Scale").grid(row=6, column=0, sticky="w", pady=(0,4))
        self.var_scale = tk.DoubleVar(value=float(initial.get("ui_scale", 1.0)))
        self.sld_scale = tk.Scale(
            frm, from_=0.50, to=1.50, resolution=0.25, orient="horizontal",
            variable=self.var_scale, bg=c["bg"], fg=c["text"],
            troughcolor=c["border"], highlightthickness=0, relief="flat",
            showvalue=True, length=240
        )
        self.sld_scale.grid(row=7, column=0, sticky="w", padx=(0,6), pady=(0,10))


        # VAD strictness
        lab(frm, "VAD Strictness (1 = loose, 5 = strict)").grid(row=8, column=0, sticky="w", pady=(0,4))
        self.var_vad = tk.IntVar(value=int(initial.get("vad_strictness", 3)))
        self.sld_vad = tk.Scale(
            frm, from_=1, to=5, orient="horizontal", variable=self.var_vad,
            bg=c["bg"], fg=c["text"], troughcolor=c["border"], highlightthickness=0, relief="flat",
            showvalue=True, length=240
        )
        self.sld_vad.grid(row=9, column=0, sticky="w", padx=(0,6), pady=(0,10))

        # Glossary strictness
        lab(frm, "Glossary correction strictness (1 = cautious, 5 = aggressive)").grid(row=10, column=0, sticky="w", pady=(0,4))
        self.var_gloss = tk.IntVar(value=int(initial.get("glossary_strictness", 3)))
        self.sld_gloss = tk.Scale(
            frm, from_=1, to=5, orient="horizontal", variable=self.var_gloss,
            bg=c["bg"], fg=c["text"], troughcolor=c["border"], highlightthickness=0, relief="flat",
            showvalue=True, length=240
        )
        self.sld_gloss.grid(row=11, column=0, sticky="w", padx=(0,6), pady=(0,10))

        # Speed mode
        self.var_speed = tk.BooleanVar(value=bool(initial.get("speed_mode", False)))
        speed_box = tk.Checkbutton(frm, text="Speed mode (lower latency, may reduce accuracy)",
                                   variable=self.var_speed, bg=c["bg"], fg=c["text"],
                                   activebackground=c["bg"], activeforeground=c["text"],
                                   selectcolor=c.get("surface", "#151a21"))
        speed_box.grid(row=12, column=0, sticky="w", pady=(4,10))

        frm.grid_columnconfigure(0, weight=1)

        # Remove punctuation (global, non-date columns)
        self.var_remove_punct = tk.BooleanVar(value=bool(initial.get("remove_punct", True)))
        rm_box = tk.Checkbutton(
            frm, text="Remove punctuation (keep hyphens) in non-date columns",
            variable=self.var_remove_punct, bg=c["bg"], fg=c["text"],
            activebackground=c["bg"], activeforeground=c["text"],
            selectcolor=c.get("surface", "#151a21")
        )
        rm_box.grid(row=13, column=0, sticky="w", pady=(4,10))


        # Date columns (per current template)
        def _fmt_date_cols_info():
            cols = getattr(self, "_date_cols_columns", [])
            pending = getattr(self, "_date_cols_pending", set())
            names = [h for (u, h) in cols if u in pending]
            if not cols:
                return "No columns available"
            if not names:
                return "None selected"
            preview = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
            return f"{len(names)}/{len(cols)} selected  ({preview})"

        

        lab(frm, "Apply Date formatting to columns").grid(row=14, column=0, sticky="w", pady=(10,4))
        self._date_cols_info = tk.Label(frm, text=_fmt_date_cols_info(), bg=c["bg"], fg=c["muted"])
        self._date_cols_info.grid(row=15, column=0, sticky="w", pady=(0,6))

        def _open_date_columns_dialog():
            def _on_done(uids):
                self._date_cols_pending = set(uids)
                try:
                    self._date_cols_info.config(text=_fmt_date_cols_info())
                except Exception:
                    pass

            DateColumnsDialog(
                self,
                columns=self._date_cols_columns,
                selected_uids=list(self._date_cols_pending),
                on_done=_on_done
            )


        btn_date_cols = tk.Button(frm, text="Select date columns", command=_open_date_columns_dialog)
        _style_btn_local(btn_date_cols)
        btn_date_cols.grid(row=16, column=0, sticky="w", pady=(0,10))

        # --- Append columns (per current template) ---
        def _fmt_append_cols_info():
            cols = getattr(self, "_append_cols_columns", [])
            pending = getattr(self, "_append_cols_pending", set())
            names = [h for (u, h) in cols if u in pending]
            if not cols:
                return "No columns available"
            if not names:
                return "None selected"
            preview = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
            return f"{len(names)}/{len(cols)} selected  ({preview})"

        # Lazy-init from active_template only when needed
        def _ensure_append_state():
            if not hasattr(self, "_append_cols_columns"):
                cols = (self.active_template.get("columns") or [])
                self._append_cols_columns = [(str(c.get("uid", "")), str(c.get("header", ""))) for c in cols]
            if not hasattr(self, "_append_cols_pending"):
                dc = self.active_template.get("append_columns")
                if isinstance(dc, list):
                    selected = [str(x) for x in dc]
                else:
                    selected = []  # default: no append columns
                self._append_cols_pending = set(selected)

        lab(frm, "Append mode columns (append to cell instead of replacing)").grid(row=17, column=0, sticky="w", pady=(10,4))
        _ensure_append_state()
        self._append_cols_info = tk.Label(frm, text=_fmt_append_cols_info(), bg=c["bg"], fg=c["muted"])
        self._append_cols_info.grid(row=18, column=0, sticky="w", pady=(0,6))

        def _open_append_columns_dialog():
            _ensure_append_state()
            def _on_done(uids):
                self._append_cols_pending = set(uids)
                try:
                    self._append_cols_info.config(text=_fmt_append_cols_info())
                except Exception:
                    pass

            AppendColumnsDialog(self,
                                columns=self._append_cols_columns,
                                selected_uids=list(self._append_cols_pending),
                                on_done=_on_done)

        btn_append_cols = tk.Button(frm, text="Select append columns", command=_open_append_columns_dialog)
        _style_btn_local(btn_append_cols)
        btn_append_cols.grid(row=19, column=0, sticky="w", pady=(0,10))


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
            _ensure_append_state()

            try:
                auto_m = max(1, int(self.var_auto.get()))
            except Exception:
                messagebox.showerror("Invalid value", "Autosave must be an integer ≥ 1.")
                return
            
            # UI scale from slider; snap to the 0.25 grid
            scale = float(self.var_scale.get())
            scale = max(0.50, min(1.50, round(scale * 4) / 4.0))

            model_label = self.var_model_label.get()
            model_key = _LABEL_TO_MODELKEY.get(model_label, _pick_default_model_key())

            s = {
                "language": lang_code,
                "autosave_minutes": auto_m,
                "ui_scale": scale,
                "model_key": model_key,
                "speed_mode": bool(self.var_speed.get()),
                "vad_strictness": max(1, min(5, int(self.var_vad.get()))),
                "glossary_strictness": max(1, min(5, int(self.var_gloss.get()))),
                "append_columns": list(self._append_cols_pending),
                "remove_punct": bool(self.var_remove_punct.get()),
            }

            s["date_columns"] = list(getattr(self, "_date_cols_pending", set()))

            try:
                self.on_apply(s)
            finally:
                self.destroy()

        def _reset_widths():
            if callable(self.on_reset_widths):
                self.on_reset_widths()
                messagebox.showinfo("Column widths", "Column widths have been reset to defaults for the current template.")
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
        # Make sure the window is big enough to show the buttons and initial content,
        # but not larger than the screen; center it. Do a second pass after a tick.
        try:
            self.update_idletasks()
            _fit_to_screen(self, margin=60)
            self.after(30, lambda: (_fit_to_screen(self, margin=60)))
        except Exception:
            pass


# ===========================
# Date column dialog
# ===========================
class DateColumnsDialog(tk.Toplevel):
    def __init__(self, master, columns: List[Tuple[str, str]], selected_uids: List[str], on_done: Callable[[List[str]], None]):
        super().__init__(master)
        self.title("Select date columns")
        self.geometry("420x420")
        self.minsize(360, 320)
        self.transient(master)

        self.on_done = on_done
        self._cols = [(str(u or ""), str(h or "")) for (u, h) in columns]
        pre = set(str(x) for x in (selected_uids or []))

        c = _set_palette(True)
        self.configure(bg=c["bg"])

        tk.Label(self, text="Mark the columns that should receive date normalization:",
                 bg=c["bg"], fg=c["text"]).pack(anchor="w", padx=12, pady=(12,6))

        # ---- Scrollable check-list ----
        outer = tk.Frame(self, bg=c["bg"])
        outer.pack(fill="both", expand=True, padx=12, pady=(0,10))

        canvas = tk.Canvas(outer, bg=c["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=c["bg"])
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_cfg(_event=None):
            inner.update_idletasks()
            bbox = canvas.bbox(canvas_window)
            if bbox:
                canvas.configure(scrollregion=bbox)
            # keep inner width == canvas width
            try:
                canvas.itemconfigure(canvas_window, width=canvas.winfo_width())
            except Exception:
                pass
        inner.bind("<Configure>", _on_cfg)
        canvas.bind("<Configure>", _on_cfg)

        # vars per column
        self._vars = {}
        for uid, header in self._cols:
            v = tk.BooleanVar(value=(uid in pre))
            chk = tk.Checkbutton(inner, text=header, variable=v,
                                 bg=c["bg"], fg=c["text"],
                                 activebackground=c["bg"], activeforeground=c["text"],
                                 selectcolor=c.get("surface", "#151a21"), anchor="w")
            chk.pack(fill="x", anchor="w")
            self._vars[uid] = v

        # ---- Buttons ----
        btns = tk.Frame(self, bg=c["bg"]); btns.pack(fill="x", padx=12, pady=(0,12))

        def _style_btn_local(b):
            try:
                b.configure(bg=c["button_bg"], fg=c["button_fg"],
                            activebackground=c["button_active_bg"], activeforeground=c["button_fg"],
                            relief="flat", highlightthickness=1,
                            highlightbackground=c["button_border"], highlightcolor=c["button_border"])
            except Exception:
                pass

        def _select_all():
            for v in self._vars.values(): v.set(True)

        def _clear_all():
            for v in self._vars.values(): v.set(False)

        def _save():
            chosen = [uid for uid, v in self._vars.items() if v.get()]
            try:
                self.on_done(chosen)
            finally:
                self.destroy()

        def _cancel():
            self.destroy()

        b_sel_all = tk.Button(btns, text="Select all", command=_select_all)
        b_clr_all = tk.Button(btns, text="Clear all", command=_clear_all)
        b_ok = tk.Button(btns, text="Save", command=_save)
        b_cancel = tk.Button(btns, text="Cancel", command=_cancel)

        for b in (b_sel_all, b_clr_all, b_ok, b_cancel):
            _style_btn_local(b)

        b_sel_all.pack(side="left")
        b_clr_all.pack(side="left", padx=(6,0))
        b_ok.pack(side="right", padx=(6,0))
        b_cancel.pack(side="right")

        try:
            enable_crisp_dark_mode(self, dark=True, delay_ms=0)
        except Exception:
            pass
        try: _fit_to_screen(self, margin=60)
        except Exception: 
            pass

# ===========================
# Append column dialog
# ===========================

class AppendColumnsDialog(tk.Toplevel):
    def __init__(self, master, columns: List[Tuple[str, str]], selected_uids: List[str],
                 on_done: Callable[[List[str]], None]):
        super().__init__(master)
        self.title("Select columns for Append mode")
        self.geometry("420x420")
        self.minsize(360, 320)
        self.transient(master)

        self.on_done = on_done
        self._cols = [(str(u or ""), str(h or "")) for (u, h) in columns]
        self._sel = set(str(x) for x in (selected_uids or []))

        c = _set_palette(True)
        self.configure(bg=c["bg"])

        tk.Label(self, text="Mark the columns where new text should be APPENDED:",
                 bg=c["bg"], fg=c["text"]).pack(anchor="w", padx=12, pady=(12, 6))

        # --- Scrollable check-list with a little left padding for each checkbox ---
        wrap = tk.Frame(self, bg=c["bg"])
        wrap.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        canvas = tk.Canvas(wrap, bg=c["bg"], highlightthickness=0)
        vs = tk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=c["bg"])

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vs.set)

        canvas.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")

        self._check_vars: list[tuple[str, tk.BooleanVar]] = []
        for uid, header in self._cols:
            row = tk.Frame(inner, bg=c["bg"])
            row.pack(anchor="w", fill="x", padx=12, pady=2)

            # left spacer (visual separation like in the Date dialog)
            tk.Label(row, text=" ", width=1, bg=c["bg"]).pack(side="left")

            var = tk.BooleanVar(value=(uid in self._sel))
            cb = tk.Checkbutton(
                row, text=header, variable=var, anchor="w",
                bg=c["bg"], fg=c["text"],
                activebackground=c["bg"], activeforeground=c["text"],
                selectcolor=c.get("surface", "#151a21")
            )
            cb.pack(side="left", fill="x", expand=True)
            self._check_vars.append((uid, var))

        # --- Buttons (same 4 buttons and layout as the Date columns dialog) ---
        btns = tk.Frame(self, bg=c["bg"])
        btns.pack(fill="x", padx=12, pady=(0, 12))

        def _style_btn_local(b):
            try:
                b.configure(bg=c["button_bg"], fg=c["button_fg"],
                            activebackground=c["button_active_bg"], activeforeground=c["button_fg"],
                            relief="flat", highlightthickness=1,
                            highlightbackground=c["button_border"], highlightcolor=c["button_border"])
            except Exception:
                pass

        def _select_all():
            for _, v in self._check_vars:
                v.set(True)

        def _clear_all():
            for _, v in self._check_vars:
                v.set(False)

        def _cancel():
            self.destroy()

        def _save():
            chosen = [uid for (uid, v) in self._check_vars if v.get()]
            try:
                self.on_done(chosen)
            finally:
                self.destroy()

        b_select_all = tk.Button(btns, text="Select all", command=_select_all)
        b_clear_all  = tk.Button(btns, text="Clear all",  command=_clear_all)
        b_cancel     = tk.Button(btns, text="Cancel",     command=_cancel)
        b_save       = tk.Button(btns, text="Save",       command=_save)

        for b in (b_select_all, b_clear_all, b_cancel, b_save):
            _style_btn_local(b)

        # Layout: left -> Select all ; right -> Clear all, Cancel, Save (exactly like Date dialog)
        b_select_all.pack(side="left")
        b_clear_all.pack(side="left", padx=(6, 0))
        b_save.pack(side="right", padx=(6, 0))
        b_cancel.pack(side="right")


        try:
            enable_crisp_dark_mode(self, dark=True, delay_ms=0)
        except Exception:
            pass
        try: _fit_to_screen(self, margin=60)
        except Exception: 
            pass

# ===========================
# App entry
# ===========================
LAST_ACTIVITY_TS = time.monotonic()
SILENCE_GUARD_GRACE_SEC = 5.0

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
