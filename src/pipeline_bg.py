"""Background driver: preprocess every subject as soon as its raw download is complete."""
import os, json, time, subprocess, sys
fs = {f["file_name"]: f["file_size"] for f in json.load(open(r"D:\tttk_data\allfiles.json"))}
for s in range(1, 43):
    out = rf"D:\tttk_data\epochs\sub-{s:02d}.npz"
    if os.path.exists(out): continue
    n = f"sub-{s:02d}_task-oddball_eeg.eeg"; p = rf"D:\tttk_data\oddball\sub-{s:02d}\{n}"
    while not (os.path.exists(p) and os.path.getsize(p) == fs[n]): time.sleep(30)
    subprocess.run([sys.executable, "preprocess.py", str(s)])
print("ALL PREPROCESSED")
