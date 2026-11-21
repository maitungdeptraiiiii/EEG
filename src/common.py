import numpy as np, glob, os
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, recall_score,
                             precision_score, f1_score, accuracy_score)
# epochs folder; override with env var EEG_EPOCHS
EP = os.environ.get("EEG_EPOCHS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "epochs"))
CLASSES = {0: "standard", 1: "target"}   # only these two stimuli are used; deviant (2) is discarded
METRICS = ["ROC-AUC", "Balanced Accuracy", "Recall Target", "Precision Target", "Macro F1", "Accuracy"]

def load(sub):
    d = np.load(os.path.join(EP, f"sub-{sub:02d}.npz"))
    X, y = d["X"], d["y"]
    k = np.isin(y, (0, 1))           # keep standard (0) and target (1) only
    X, y = X[k], y[k].astype(int)
    assert set(np.unique(y)) == {0, 1}, f"sub-{sub:02d}: unexpected labels {np.unique(y)}"
    return X, y, list(d["ch"]), float(d["sfreq"]), float(d["tmin"])

def subjects():
    return sorted(int(os.path.basename(p)[4:6]) for p in glob.glob(os.path.join(EP, "sub-*.npz")))

def metrics(y, p, thr=.5):
    """y: true labels (1 = target), p: predicted probability of target."""
    yh = (np.asarray(p) > thr).astype(int)
    return {"ROC-AUC": roc_auc_score(y, p),
            "Balanced Accuracy": balanced_accuracy_score(y, yh),
            "Recall Target": recall_score(y, yh, pos_label=1, zero_division=0),
            "Precision Target": precision_score(y, yh, pos_label=1, zero_division=0),
            "Macro F1": f1_score(y, yh, average="macro", zero_division=0),
            "Accuracy": accuracy_score(y, yh)}

def mean_metrics(rows):
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
