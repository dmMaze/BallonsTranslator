import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import re, os, datetime

def load_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.rstrip() for line in f if line.strip()]

def split_by_page(lines):
    pages = {}
    current_page = None
    for line in lines:
        if line.startswith("###"):
            current_page = line.strip()
            pages[current_page] = []
        elif current_page:
            pages[current_page].append(line.strip())
    return pages

def check_pages(original_pages, translated_pages):
    report = []
    log_missing = []

    for page, jp_lines in original_pages.items():
        en_lines = translated_pages.get(page, [])
        missing = []
        for line in jp_lines:
            if re.match(r"^\d+\.", line):
                num = line.split(".")[0]
                found = any(l.startswith(num + ".") for l in en_lines)
                if not found:
                    missing.append(line)
        if missing:
            report.append(f"⚠️ {page} is missing translations:\n" + "\n".join("   " + m for m in missing))
            log_missing.append(page)
            log_missing.extend(missing)
            log_missing.append("")  # blank line between pages
        else:
            report.append(f"✅ {page} is fully translated.")

    # Save untranslated text into a log file
    if log_missing:
        log_filename = f"untranslated_only_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_path = os.path.join(os.path.dirname(translated_file), log_filename)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_missing))
        messagebox.showinfo("Log Saved", f"Untranslated text saved to:\n{log_path}")

    return "\n\n".join(report)

def merge_log():
    global log_file
    if not translated_file or not original_file:
        messagebox.showerror("Error", "Please load both Original and Translated files first.")
        return

    if not log_file:
        log_path = filedialog.askopenfilename(title="Select Translated Log File", filetypes=[("Text Files", "*.txt")])
        if not log_path:
            return
        log_file = log_path
    else:
        log_path = log_file

    # load files
    log_lines = load_file(log_path)
    translated_lines = load_file(translated_file)
    original_lines = load_file(original_file)

    translated_pages = split_by_page(translated_lines)
    log_pages = split_by_page(log_lines)
    original_pages = split_by_page(original_lines)

    # merge translations with preserved indentation
    for page, log_page_lines in log_pages.items():
        if page not in translated_pages:
            translated_pages[page] = []

        for log_line in log_page_lines:
            if re.match(r"^\d+\.", log_line):  # numbered line
                num = log_line.split(".")[0]

                # find the original line to preserve indentation
                orig_line = next((l for l in original_pages.get(page, []) if l.startswith(num + ".")), None)
                if orig_line:
                    leading_spaces = len(orig_line) - len(orig_line.lstrip(" "))
                else:
                    leading_spaces = 0

                new_line = " " * leading_spaces + log_line

                # replace if exists, else append
                found_idx = next((i for i, l in enumerate(translated_pages[page]) if l.strip().startswith(num + ".")), None)
                if found_idx is not None:
                    translated_pages[page][found_idx] = new_line
                else:
                    translated_pages[page].append(new_line)

    # reconstruct file with correct formatting
    merged_lines = []
    for page, lines in original_pages.items():
        merged_lines.append(page)
        # use original order, but fall back to translated if needed
        for orig_line in lines:
            num_match = re.match(r"^(\d+)\.", orig_line.strip())
            if num_match:
                num = num_match.group(1)
                # check if we already replaced this
                replaced_line = next((l for l in translated_pages.get(page, []) if l.strip().startswith(num + ".")), None)
                if replaced_line:
                    merged_lines.append(replaced_line)
                    continue
            merged_lines.append(orig_line)

    merged_filename = os.path.splitext(os.path.basename(translated_file))[0] + "_merged.txt"
    merged_path = os.path.join(os.path.dirname(translated_file), merged_filename)
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_lines))

    messagebox.showinfo("Merge Completed", f"Merged translations saved to:\n{merged_path}")

# -------------------- Drag & Drop Handlers --------------------

def drop_original(event):
    set_original(event.data.strip("{}"))

def drop_translated(event):
    set_translated(event.data.strip("{}"))

def drop_log(event):
    set_log(event.data.strip("{}"))

def set_original(path):
    global original_file
    if os.path.isfile(path):
        original_file = path
        lbl_original.config(text=f"Original Loaded:\n{os.path.basename(original_file)}", bg="lightgreen")

def set_translated(path):
    global translated_file
    if os.path.isfile(path):
        translated_file = path
        lbl_translated.config(text=f"Translated Loaded:\n{os.path.basename(translated_file)}", bg="lightgreen")

def set_log(path):
    global log_file
    if os.path.isfile(path):
        log_file = path
        lbl_log.config(text=f"Log Loaded:\n{os.path.basename(log_file)}", bg="lightgreen")

def browse_original():
    path = filedialog.askopenfilename(title="Select Original (JP) File", filetypes=[("Text Files", "*.txt")])
    if path:
        set_original(path)

def browse_translated():
    path = filedialog.askopenfilename(title="Select Translated (EN) File", filetypes=[("Text Files", "*.txt")])
    if path:
        set_translated(path)

def run_check():
    if not original_file or not translated_file:
        messagebox.showerror("Error", "Please load both Original and Translated files first.")
        return
    original = load_file(original_file)
    translated = load_file(translated_file)
    original_pages = split_by_page(original)
    translated_pages = split_by_page(translated)
    result = check_pages(original_pages, translated_pages)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, result)

# -------------------- GUI --------------------

root = TkinterDnD.Tk()
root.title("Translation Checker by Page")
root.geometry("850x700")

original_file = None
translated_file = None
log_file = None

frame = tk.Frame(root)
frame.pack(pady=10)

lbl_original = tk.Label(frame, text="Drop Original (JP) File Here", width=40, relief="ridge", bg="lightcoral")
lbl_original.grid(row=0, column=0, padx=10)
lbl_original.drop_target_register(DND_FILES)
lbl_original.dnd_bind('<<Drop>>', drop_original)

btn_load_original = tk.Button(frame, text="Browse Original (JP)", command=browse_original, width=20)
btn_load_original.grid(row=1, column=0, padx=10, pady=5)

lbl_translated = tk.Label(frame, text="Drop Translated (EN) File Here", width=40, relief="ridge", bg="lightcoral")
lbl_translated.grid(row=0, column=1, padx=10)
lbl_translated.drop_target_register(DND_FILES)
lbl_translated.dnd_bind('<<Drop>>', drop_translated)

btn_load_translated = tk.Button(frame, text="Browse Translated (EN)", command=browse_translated, width=20)
btn_load_translated.grid(row=1, column=1, padx=10, pady=5)

# Log file drag & drop
lbl_log = tk.Label(root, text="Drop Log File Here", width=40, relief="ridge", bg="lightcoral")
lbl_log.pack(pady=5)
lbl_log.drop_target_register(DND_FILES)
lbl_log.dnd_bind('<<Drop>>', drop_log)

btn_check = tk.Button(root, text="Check Translation", command=run_check, width=40, bg="lightblue")
btn_check.pack(pady=10)

btn_merge = tk.Button(root, text="Merge Translations from Log", command=merge_log, width=40, bg="lightgreen")
btn_merge.pack(pady=5)

output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=25)
output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()