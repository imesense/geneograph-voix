import tkinter as tk

from typing import List, Tuple, Callable

from geneograph_voix.Helpers.ScreenHelpers import _fit_to_screen
from geneograph_voix.Views.Palette import _set_palette
from geneograph_voix.Views.Styles import enable_crisp_dark_mode

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
