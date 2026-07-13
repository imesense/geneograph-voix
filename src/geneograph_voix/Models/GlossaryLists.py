import json
import os
import unicodedata
import uuid

from typing import List, Optional

from geneograph_voix.Models.Config import (
    DATA_DIR,
    GLOSSARIES_FILE,
    TEMPLATES_FILE
)

# =========================================
# Templates & Global Glossary lists
# =========================================

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _make_default_template(headers: List[str]) -> dict:
    return {
        "id": _new_uuid(),
        "name": "Default",
        "columns": [{"uid": _new_uuid(), "header": h} for h in headers],
        "mapping": {},               # column_uid -> [list_id, ...]
        "column_widths": None        # optional list of ints
    }

def _ensure_templates_file(default_headers: List[str]) -> dict:
    data = {}
    if os.path.exists(TEMPLATES_FILE):
        try:
            with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("templates load error:", e)
    if not isinstance(data, dict):
        data = {}
    if "templates" not in data or not isinstance(data.get("templates"), list) or not data["templates"]:
        # create default
        t = _make_default_template(default_headers)
        data = {"last_template_id": t["id"], "templates": [t]}
        _write_templates_file(data)
    return data

def _write_templates_file(obj: dict):
    tmp = os.path.join(os.path.dirname(TEMPLATES_FILE) or ".", "~templates.tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, TEMPLATES_FILE)

def _active_template_record() -> Optional[dict]:
    data = _ensure_templates_file(default_headers=[
        "№","№ М","№ Ж","Дата","Имя","Фамилия","Имя отца","Имя матери","Восприемник","Страница","Комментарий"
    ])
    tid = data.get("last_template_id")
    for t in data.get("templates", []):
        if t.get("id") == tid:
            return t
    # fallback to first
    if data.get("templates"):
        data["last_template_id"] = data["templates"][0]["id"]
        _write_templates_file(data)
        return data["templates"][0]
    return None

def _set_active_template_id(tid: str):
    data = _ensure_templates_file(default_headers=[
        "№","№ М","№ Ж","Дата","Имя","Фамилия","Имя отца","Имя матери","Восприемник","Страница","Комментарий"
    ])
    data["last_template_id"] = tid
    _write_templates_file(data)

def _data_path_for_template(tid: str) -> str:
    return os.path.join(DATA_DIR, f"{tid}.csv")

GLOSSARY_LISTS: dict = {}

def load_glossary_lists():
    global GLOSSARY_LISTS
    obj = {}
    if os.path.exists(GLOSSARIES_FILE):
        try:
            with open(GLOSSARIES_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print("glossaries.json load error:", e)
    if not isinstance(obj, dict):
        obj = {}
    if "lists" not in obj:
        obj["lists"] = {}
    GLOSSARY_LISTS = obj

def save_glossary_lists():
    obj = GLOSSARY_LISTS if isinstance(GLOSSARY_LISTS, dict) else {"lists": {}}
    lists = obj.get("lists", {})
    
    def _key(name: str) -> str:
        t = unicodedata.normalize("NFKD", str(name or "")).casefold()
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        return t.replace("ё", "е")

    items = sorted(lists.items(), key=lambda kv: _key(kv[1].get("name", "")))
    obj = {"lists": {lid: rec for lid, rec in items}}

    tmp = os.path.join(os.path.dirname(GLOSSARIES_FILE) or ".", "~glossaries.tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, GLOSSARIES_FILE)
