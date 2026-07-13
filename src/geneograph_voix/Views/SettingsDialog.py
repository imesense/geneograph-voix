import tkinter as tk

from typing import Optional, Callable
from tkinter import messagebox

from geneograph_voix.Helpers.Screen import _fit_to_screen
from geneograph_voix.Models.Config import (
    _CODE_TO_LABEL,
    _LABEL_TO_CODE,
    _LABEL_TO_MODELKEY,
    _LANG_LABELS,
    _MODEL_LABELS,
    _MODELKEY_TO_LABEL
)
from geneograph_voix.Models.WhisperModel import _pick_default_model_key
from geneograph_voix.Views.AppendColumnsDialog import AppendColumnsDialog
from geneograph_voix.Views.DateColumnsDialog import DateColumnsDialog
from geneograph_voix.Views.Palette import _set_palette
from geneograph_voix.Views.Styles import enable_crisp_dark_mode

try:
    from tkinter import ttk
except Exception:
    ttk = None

# ===========================
# Settings dialog (unchanged, except reset widths now per-template)
# ===========================
class SettingsDialog(tk.Toplevel):
    _append_cols_columns: list[tuple[str, str]]
    _append_cols_pending: set[str]

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
