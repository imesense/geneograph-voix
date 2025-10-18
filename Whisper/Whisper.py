import tkinter as tk
from tksheet import Sheet
from tkinter import filedialog, messagebox
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

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_SIZE = "large-v3"
LANGUAGE = "ru"
SAMPLERATE = 16000
BLOCK_DURATION = 4  # seconds
COMPUTE_TYPE = "int8_float16"
DATA_FILE = "records.csv"  # persistent storage
COLUMN_WIDTHS_FILE = "column_widths.json"

# ----------------------------
# AUDIO QUEUE
# ----------------------------
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_queue.put(indata.copy())

# ----------------------------
# WHISPER MODEL
# ----------------------------
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper model ({MODEL_SIZE}) on {device}...")
model = WhisperModel(MODEL_SIZE, device=device, compute_type=COMPUTE_TYPE)
print("Model loaded")

# ----------------------------
# NAME GLOSSARY
# ----------------------------
GLOSSARIES = {
    "Дата":["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"],
    "Имя": ["Авксентий", "Агафья", "Феодосий", "Никифор", "Домна", "Домникия"],
    "Фамилия": ["Антоневич", "Атаманчук", "Бажура", "Баран", "Барладян", "Белецкий", "Бергель", "Берёза", "Берко", "Беская", "Беский", "Билецкий", "Блонский", "Богач", "Бойван", "Брык", "Булацел", "Буньковский", "Вакарь", "Василевский", "Ведмедь", "Вербицкий", "Винярская", "Винярский", "Вишневый", "Возиан", "Войцеховская", "Ворник", "Вроблевский", "Галузинский", "Галька", "Ганзюк", "Гнатковский", "Гоголь", "Гулевич", "Гулька", "Гуляйкурка", "Дабиша", "Дамаскин", "Девдера", "Джосан", "Дзецюл", "Дидык", "Добровольский", "Додобец", "Домбровский", "Драгомирецкий", "Жовтяк", "Заведия", "Заверуев", "Заверуха", "Каня", "Катрук", "Кафанатул", "Келеватый", "Кифа", "Комарь", "Кожухарь", "Конофольский", "Коцага", "Красовский", "Креминский", "Крупка", "Крыжанюк", "Малиновский", "Малоглов", "Мамалыга", "Марикуца", "Медведь", "Мельник", "Московчук", "Муляр", "Навроцкий", "Ниткоплут", "Паица", "Паламарь", "Палий", "Пасечник", "Пасинковский", "Пашковский", "Перебийнос", "Перевозник", "Передерко", "Подгорный", "Попиль", "Ренович", "Розгон", "Сахновский", "Синица", "Сиракуца", "Слободяник", "Спринчан", "Ужейко", "Уретий", "Урсол", "Фаринюк", "Федык", "Фибрук", "Фигляр", "Целик", "Чернобыль", "Чолак", "Шарко", "Шахрай", "Шелёнга"],
    "Имя отца": ["Авксентий", "Авксентиев", "Феодосиев"],
    "Имя матери": ["Авксентиева", "Феодосиева"]
}

# ----------------------------
# UTILS
# ----------------------------
def levenshtein(a, b):
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

def correct_text_for_column(text, header):
    glossary = GLOSSARIES.get(header, [])
    if not glossary:
        return text
    words = text.split()
    corrected = []
    for w in words:
        best_match = min(glossary, key=lambda x: levenshtein(x.lower(), w.lower()))
        if levenshtein(best_match.lower(), w.lower()) <= 2:
            corrected.append(best_match)
        else:
            corrected.append(w)
    return " ".join(corrected)

# ----------------------------
# TRANSCRIBE FUNCTION
# ----------------------------
def transcribe_buffer(buffer):
    if buffer.shape[0] == 0:
        return ""
    samples = buffer.flatten()
    all_names = [name for col in GLOSSARIES.values() for name in col]
    initial_prompt = " ".join(all_names)
    segments, _ = model.transcribe(
        samples,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
        no_speech_threshold=0.8,
        initial_prompt=initial_prompt
    )
    text = " ".join([seg.text for seg in segments]).strip()
    return text

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


        # ----------------------------
        # Sheet (headers + 20 initial rows)
        # ----------------------------
        self.headers = ["№","№ М","№ Ж","Дата","Имя","Фамилия","Имя отца",
                        "Имя матери","Восприемник","Страница","Комментарий"]

        self.sheet = Sheet(root, headers=self.headers, height=400, width=1000)
        self.sheet.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.sheet.enable_bindings(("single_select","row_select","column_select","arrowkeys","edit_cell","rc_popup_menu", "drag_select", "column_width_resize", "row_height_resize", "copy", "paste", "delete"))

        # Load saved widths after rendering
        self.root.after(500, self.load_column_widths)
        # Save widths when column is resized
        self.sheet.extra_bindings("column_width_resize", lambda event: self.save_column_widths())

        # Also save on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Expand sheet with window
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Preview label
        self.preview_label = tk.Label(root, text="Preview:", anchor="w", fg="black", font=("Calibri", 12, "bold"))
        self.preview_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Load saved data if exists
        self.load_data()

        # Audio buffer
        self.buffer = np.zeros((0,1), dtype=np.float32)

        # Handle close event
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------------
    # SHEET HELPERS
    # ----------------------------
    def add_text_to_cell(self, text):
        selected = self.sheet.get_selected_cells()
        if selected:
            row, col = list(selected)[0]
            header = self.headers[col]
            text = correct_text_for_column(text, header)
            current = self.sheet.get_cell_data(row, col)
            new_text = f"{current} {text}" if current else text
            self.sheet.set_cell_data(row, col, new_text)
            self.sheet.redraw()
        else:
            messagebox.showwarning("No cell selected", "Please select a cell before adding text.")

    def add_row(self):
        num_cols = self.sheet.get_total_columns()
        self.sheet.insert_rows(rows=[["" for _ in range(num_cols)]], idx="end")

    def clear_all_cells(self):
        total_rows = self.sheet.get_total_rows()
        total_cols = self.sheet.get_total_columns()
        for r in range(total_rows):
            for c in range(total_cols):
                self.sheet.set_cell_data(r, c, "")
        self.sheet.redraw()
        
    def save_column_widths(self):
        """Save column widths safely across tksheet versions"""
        try:
            if hasattr(self.sheet, "get_column_widths"):
                widths = self.sheet.get_column_widths()
            elif hasattr(self.sheet, "column_widths"):
                widths = list(self.sheet.column_widths)
            else:
                print("⚠Could not retrieve column widths — attribute missing.")
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
            print("Loaded column widths:", widths)
            self.root.after(300, lambda: self.apply_column_widths(widths))
        except FileNotFoundError:
            print("No saved column widths found.")
        
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
                    print(f"Could not set width for col {col}: {e}")

            print("Column widths applied successfully")
        except Exception as e:
            print("apply_column_widths error:", e)

            
    # ----------------------------
    # LISTENING
    # ----------------------------
    def start_listening(self):
        self.is_listening = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.audio_capture_thread, daemon=True).start()
        threading.Thread(target=self.transcribe_thread, daemon=True).start()

    def stop_listening(self):
        self.is_listening = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def audio_capture_thread(self):
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=audio_callback):
            while self.is_listening:
                sd.sleep(100)

    def transcribe_thread(self):
        temp_buffer = np.zeros((0,1), dtype=np.float32)
        while self.is_listening:
            while not audio_queue.empty():
                temp_buffer = np.concatenate((temp_buffer, audio_queue.get()))

            # Live preview for 1-second blocks
            if len(temp_buffer) >= SAMPLERATE * 1:
                preview_text = transcribe_buffer(temp_buffer)
                if preview_text:
                    self.root.after(0, lambda t=preview_text: self.preview_label.config(text="🎤 Preview: " + t))

            # Commit only after full block + VAD
            if len(temp_buffer) >= SAMPLERATE * BLOCK_DURATION:
                full_text = transcribe_buffer(temp_buffer)
                if full_text:
                    self.root.after(0, self.add_text_to_cell, full_text)
                temp_buffer = np.zeros((0,1), dtype=np.float32)

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
            df = df.fillna("")  # ensure all NaN are empty strings
            self.sheet.set_sheet_data(df.values.tolist())
        else:
            # Initialize empty rows if no file
            for _ in range(20):
                self.add_row()


    def on_close(self):
        self.save_data()
        self.root.destroy()
        self.save_column_widths()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechSheetApp(root)
    root.mainloop()
