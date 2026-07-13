import os
import re
import sys
import threading
import time
import torch

import numpy as np
import pandas as pd
import tkinter as tk
import sounddevice as sd

from collections import deque
from contextlib import contextmanager
from tkinter import filedialog, messagebox, simpledialog
from tksheet import Sheet
from typing import List, Optional

from geneograph_voix.Helpers.Audio import (
    AUDIO_QUEUE,
    _preview_is_banned,
    audio_callback,
    clean_person_field,
    is_silence,
    rms_db
)
from geneograph_voix.Helpers.Language import (
    _remove_punct_keep_hyphen,
    looks_like_outro,
    strip_trailing_dot
)
from geneograph_voix.Helpers.Screen import (
    _fit_to_screen,
    _fit_to_screen_bounds,
    _parse_geometry
)
from geneograph_voix.Models.Config import (
    AUDIO_BLOCK_SEC,
    AUTOSAVE_EVERY_MS,
    BLOCK_DURATION,
    COMMIT_MIN_SILENCE_SEC,
    COMMIT_TAIL_SILENCE_SEC,
    DATA_FILE,
    MALE_EXCEPTIONS,
    NAME_COLUMNS,
    PREVIEW_CLEAR_DELAY_MS,
    PREVIEW_MIN_INTERVAL_SEC,
    REMOVE_PUNCT,
    SAMPLERATE,
    SILENCE_GUARD_GRACE_SEC,
    SPEED_MODE,
    TABLE_ZOOM_PCT,
    UI_SCALE,
    _preview_tail_sec
)
from geneograph_voix.Models.Transcription import (
    transcribe_buffer,
    transcribe_buffer_commit
)
from geneograph_voix.Models.Devices import DEVICE
from geneograph_voix.Models.GlossaryLists import (
    GLOSSARY_LISTS,
    _active_template_record,
    _data_path_for_template,
    _ensure_templates_file,
    _make_default_template,
    _new_uuid,
    _set_active_template_id,
    _write_templates_file,
    load_glossary_lists
)
from geneograph_voix.Models.GlossaryCorrection import (
    GLOSSARIES,
    _clear_best_caches,
    correct_text_for_column,
    normalize_date
)
from geneograph_voix.Models.ModelCoefficients import (
    GLOSSARY_STRICTNESS,
    VAD_STRICTNESS,
    _energy_gate_db
)
from geneograph_voix.Models.Settings import (
    LANGUAGE,
    MODEL_KEY,
    _apply_settings_to_globals,
    _read_settings_from_file,
    _save_settings_file
)
from geneograph_voix.Models.SheetConfig import SHEET_BINDS
from geneograph_voix.Models.SileroVad import SILERO_DEVICE
from geneograph_voix.Models.WhisperModel import (
    COMPUTE_TYPE,
    _load_whisper_model,
    _pick_default_model_key,
    _warmup_model
)
from geneograph_voix.Views.GlossaryListsAndMappingDialog import GlossaryListsAndMappingDialog
from geneograph_voix.Views.Palette import _set_palette
from geneograph_voix.Views.SettingsDialog import SettingsDialog
from geneograph_voix.Views.Styles import (
    _apply_dark_ui,
    _try_style_tksheet,
    enable_crisp_dark_mode
)
from geneograph_voix.Views.TemplateManagerDialog import TemplateManagerDialog

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
