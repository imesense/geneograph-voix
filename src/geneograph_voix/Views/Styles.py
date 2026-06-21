import os
import sys

import tkinter as tk

from typing import Union

from geneograph_voix.Helpers.ResourceHelpers import _resource_path
from geneograph_voix.Helpers.ScreenHelpers import _set_windows_dpi_awareness
from geneograph_voix.Models.Config import UI_SCALE
from geneograph_voix.Views.Palette import _set_palette

try:
    from tkinter import ttk
    import tkinter.font as tkfont
except Exception:
    ttk = None
    tkfont = None

# ===========================
# UI theming
# ===========================

# ---------- App icon helpers ----------
APP_ICON_ICO = "app.ico"        # put the same .ico you use with PyInstaller here
APP_ICON_PNG = "app_256.png"    # optional: for non-Windows, a PNG works best

def _set_app_icon(root: Union[tk.Tk, tk.Toplevel]):
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
                root._app_icon_img = img  # type: ignore[attr-defined]
    except Exception as e:
        print("icon set warn:", e)

# Remember the original Tk scaling so re-applying doesn't multiply it
_BASE_TK_SCALING = None

def _auto_scaling(root: Union[tk.Tk, tk.Toplevel]):
    global _BASE_TK_SCALING
    try:
        if _BASE_TK_SCALING is None:
            # Tk's 'scaling' is pixels-per-point (1pt = 1/72"). Default is usually 1.0.
            _BASE_TK_SCALING = float(root.tk.call('tk', 'scaling')) or 1.0
        new_scale = max(0.5, min(2.5, float(_BASE_TK_SCALING) * float(UI_SCALE)))
        root.tk.call('tk', 'scaling', new_scale)
    except Exception:
        pass

def _apply_fonts(root: Union[tk.Tk, tk.Toplevel]):
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

def _recolor_tk_widgets(root: Union[tk.Tk, tk.Toplevel], dark=True):
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

def _try_style_tksheet(root: Union[tk.Tk, tk.Toplevel], dark=True):
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

def _apply_dark_ui(root: Union[tk.Tk, tk.Toplevel], dark=True):
    _set_windows_dpi_awareness()
    _auto_scaling(root)
    _apply_fonts(root)
    _style_ttk(dark=dark)
    _recolor_tk_widgets(root, dark=dark)
    _try_style_tksheet(root, dark=dark)

def enable_crisp_dark_mode(root: Union[tk.Tk, tk.Toplevel], dark=True, delay_ms=350):
    try:
        root.after(delay_ms, lambda: _apply_dark_ui(root, dark=dark))
    except Exception:
        _apply_dark_ui(root, dark=dark)
