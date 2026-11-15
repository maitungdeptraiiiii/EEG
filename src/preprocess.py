"""Preprocess one subject of the oddball task -> epochs npz.
Pipeline: load BrainVision -> notch 50 Hz -> band-pass 0.1-40 Hz -> avg reference
-> ICA (eye-blink removal, Fp1/Fp2 as EOG proxy) -> epoch [-0.2, 0.8] s, baseline [-0.2,0]
-> resample to 250 Hz.  Labels: 0=standard(frequent) 1=target 2=deviant.
Usage: python preprocess.py SUBJ [SUBJ ...]
"""
import sys, os, numpy as np, mne
mne.set_log_level("WARNING")
RAW = r"D:\tttk_data\oddball"; OUT = r"D:\tttk_data\epochs"; os.makedirs(OUT, exist_ok=True)
EVID = {"Stimulus/S  5": 0, "Stimulus/S  6": 1, "Stimulus/S  7": 2}

def run(sub, ica=True):
    s = f"sub-{sub:02d}"
    raw = mne.io.read_raw_brainvision(f"{RAW}\{s}\{s}_task-oddball_eeg.vhdr", preload=True)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    raw.notch_filter(50.0)
    raw.filter(0.1, 40.0, verbose=False)
    raw.set_eeg_reference("average")
    n_ica = 0
    if ica:
        f1 = raw.copy().filter(1.0, None)
        f1.resample(250)
        m = mne.preprocessing.ICA(n_components=30, method="fastica", random_state=0, max_iter=500)
        m.fit(f1)
        eog = [c for c in ("Fp1", "Fp2") if c in raw.ch_names]
        bad, _ = m.find_bads_eog(f1, ch_name=eog, threshold=3.0)
        m.exclude = bad; n_ica = len(bad)
        raw = m.apply(raw)
    ev, ed = mne.events_from_annotations(raw, event_id=EVID)
    ep = mne.Epochs(raw, ev, ed, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True, reject=None)
    ep.resample(250)
    X = ep.get_data().astype(np.float32); y = np.array([{0:0,1:1,2:2}[k] for k in ep.events[:, 2]])
    # map ids back: ed values are already 0/1/2
    np.savez_compressed(f"{OUT}\{s}.npz", X=X, y=y, ch=np.array(raw.ch_names), sfreq=250, tmin=-0.2, n_ica=n_ica)
    print(s, X.shape, np.bincount(y), "ICA excluded:", n_ica, flush=True)

if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))
