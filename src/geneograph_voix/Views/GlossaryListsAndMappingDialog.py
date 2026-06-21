import unicodedata

import tkinter as tk

from tkinter import messagebox, simpledialog
from typing import Optional

from geneograph_voix.Helpers.ScreenHelpers import _fit_to_screen
from geneograph_voix.Models.Config import UI_SCALE
from geneograph_voix.Models.GlobalGlossaryLists import (
    GLOSSARY_LISTS,
    _active_template_record,
    _ensure_templates_file,
    _new_uuid,
    _write_templates_file,
    save_glossary_lists
)
from geneograph_voix.Views.Palette import _set_palette
from geneograph_voix.Views.Styles import enable_crisp_dark_mode

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
