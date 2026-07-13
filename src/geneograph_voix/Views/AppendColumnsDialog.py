import tkinter as tk

from typing import List, Tuple, Callable

from geneograph_voix.Helpers.Screen import _fit_to_screen
from geneograph_voix.Views.Palette import _set_palette
from geneograph_voix.Views.Styles import enable_crisp_dark_mode

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
