"""Download the Kaggle mirror of the Nencki-Symfonia oddball data (public, kagglehub, no login needed)
and preprocess each subject as soon as its files are complete (download runs ahead in a thread).
Usage: python pipeline_kaggle.py [subjects...]     (default: 1..42)"""
import os, sys, threading, queue, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(HERE, "..", "data", "kaggle"))
import kagglehub
DS = "trungdttik55/nencki-symfonia-eeg-oddball"
EP = os.environ.get("EEG_EPOCHS", os.path.join(HERE, "..", "data", "epochs"))
subs = [int(a) for a in sys.argv[1:]] or list(range(1, 43))
todo = [s for s in subs if not os.path.exists(os.path.join(EP, f"sub-{s:02d}.npz"))]
q = queue.Queue(maxsize=2)          # at most 2 subjects downloaded ahead of preprocessing

def downloader():
    for s in todo:
        for ext in ("eeg.vhdr", "eeg.vmrk", "events.tsv", "eeg.eeg"):
            p = f"data/sub-{s:02d}/eeg/sub-{s:02d}_task-oddball_{ext}"
            for attempt in range(5):
                try: kagglehub.dataset_download(DS, path=p); break
                except Exception as e: print("retry", p, e, flush=True)
            else: raise SystemExit(f"failed {p}")
        print("downloaded", s, flush=True); q.put(s)
    q.put(None)

threading.Thread(target=downloader, daemon=True).start()
while (s := q.get()) is not None:
    r = subprocess.run([sys.executable, os.path.join(HERE, "preprocess.py"), str(s)])
    if r.returncode: print("PREPROCESS FAILED", s, flush=True)
print("ALL PREPROCESSED", flush=True)
