"""Download oddball EEG (BrainVision) files of Nencki-Symfonia dataset (GigaDB 100990).
Usage: python download.py START END   (subject numbers, inclusive)
"""
import json, os, sys, urllib.request, re
OUT = r"D:\tttk_data\oddball"
fs = json.load(open(r"D:\tttk_data\allfiles.json"))
a, b = int(sys.argv[1]), int(sys.argv[2])
for f in sorted(fs, key=lambda f: f["file_name"]):
    n = f["file_name"]
    m = re.match(r"sub-(\d+)_task-oddball_eeg\.(vhdr|vmrk|eeg)$", n)
    if not m or not (a <= int(m.group(1)) <= b):
        continue
    d = os.path.join(OUT, f"sub-{m.group(1)}"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, n)
    if os.path.exists(p) and os.path.getsize(p) == f["file_size"]:
        continue
    print("get", n, f["file_size"], flush=True)
    urllib.request.urlretrieve(f["url"], p + ".part"); os.replace(p + ".part", p)
print("done", a, b)
