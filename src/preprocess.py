"""Preprocess one subject of the oddball task -> epochs npz.
Pipeline: load BrainVision -> notch 50 Hz -> band-pass 0.1-40 Hz -> resample 250 Hz -> avg reference
-> ICA (eye-blink removal, Fp1/Fp2 as EOG proxy) -> epoch [-0.2, 0.8] s, baseline [-0.2, 0].
Labels: 0=standard (S 5), 1=target (S 6), 2=deviant (S 7). Deviant is stored but discarded in common.load,
so every model is trained/evaluated on standard vs target only.
(Resampling is done on the continuous data, after the 40 Hz low-pass, to keep RAM usage low.)
Usage: python preprocess.py SUBJ [SUBJ ...]
Paths: env EEG_RAW (BIDS root containing sub-XX/eeg/) and EEG_EPOCHS (output folder).
"""
import sys, os, numpy as np, mne
mne.set_log_level("WARNING")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("EEG_RAW", os.path.join(HERE, "..", "data", "kaggle", "datasets", "trungdttik55",
                                             "nencki-symfonia-eeg-oddball", "versions", "2", "data"))
OUT = os.environ.get("EEG_EPOCHS", os.path.join(HERE, "..", "data", "epochs")); os.makedirs(OUT, exist_ok=True)
EVID = {"Stimulus/S  5": 0, "Stimulus/S  6": 1, "Stimulus/S  7": 2}

def vhdr_path(s):
    for p in (os.path.join(RAW, s, "eeg", f"{s}_task-oddball_eeg.vhdr"), os.path.join(RAW, s, f"{s}_task-oddball_eeg.vhdr")):
        if os.path.exists(p): return p
    raise FileNotFoundError(s)

def run(sub, ica=True):
    s = f"sub-{sub:02d}"
    raw = mne.io.read_raw_brainvision(vhdr_path(s), preload=True)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    raw.notch_filter(50.0)
    raw.filter(0.1, 40.0, verbose=False)
    raw.resample(250)
    raw.set_eeg_reference("average")
    n_ica = 0
    if ica:
        f1 = raw.copy().filter(1.0, None)
        m = mne.preprocessing.ICA(n_components=30, method="fastica", random_state=0, max_iter=500)
        m.fit(f1)
        eog = [c for c in ("Fp1", "Fp2") if c in raw.ch_names]
        bad, _ = m.find_bads_eog(f1, ch_name=eog, threshold=3.0)
        m.exclude = bad; n_ica = len(bad); del f1
        raw = m.apply(raw)
    ev, ed = mne.events_from_annotations(raw, event_id=EVID)
    ep = mne.Epochs(raw, ev, ed, tmin=-0.2, tmax=0.8 - 1 / 250, baseline=(-0.2, 0), preload=True, reject=None)
    X = ep.get_data().astype(np.float32); y = ep.events[:, 2].astype(int)
    tmp = os.path.join(OUT, f"tmp_{s}.npz")      # atomic write: readers never see a partial file
    np.savez_compressed(tmp, X=X, y=y, ch=np.array(raw.ch_names), sfreq=250, tmin=-0.2, n_ica=n_ica)
    os.replace(tmp, os.path.join(OUT, f"{s}.npz"))
    print(s, X.shape, np.bincount(y), "ICA excluded:", n_ica, flush=True)

if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))
