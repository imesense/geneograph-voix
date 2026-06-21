import os

import tkinter as tk
import pandas as pd

from typing import Optional, Callable
from tkinter import filedialog, messagebox, simpledialog

from geneograph_voix.Helpers.ScreenHelpers import _fit_to_screen
from geneograph_voix.Models.GlobalGlossaryLists import (
    _data_path_for_template,
    _ensure_templates_file,
    _make_default_template,
    _new_uuid,
    _write_templates_file
)
from geneograph_voix.Views.Palette import _set_palette
from geneograph_voix.Views.Styles import enable_crisp_dark_mode

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
