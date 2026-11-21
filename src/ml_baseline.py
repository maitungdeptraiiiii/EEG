"""Standard ML on evoked-response features. Within-subject stratified 5-fold CV.
Usage: python ml_baseline.py [subjects...]"""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from common import *

class WinMean:   # ERP features: mean amplitude in 50 ms bins over 0-800 ms, all channels
    def __init__(s, sf=250., t0=-.2): s.sf, s.t0 = sf, t0
    def fit(s, X, y=None): return s
    def get_params(s, deep=True): return {"sf": s.sf, "t0": s.t0}
    def set_params(s, **p): return s
    def transform(s, X):
        a = int((0 - s.t0) * s.sf); w = int(.05 * s.sf)
        n = (X.shape[2] - a) // w
        return X[:, :, a:a + n * w].reshape(X.shape[0], X.shape[1], n, w).mean(3).reshape(len(X), -1)
    def fit_transform(s, X, y=None): return s.transform(X)

class ChanPCA:   # spatial PCA: removes rank deficiency (avg-ref + ICA) before xDAWN
    def __init__(s, n=60): s.n = n
    def get_params(s, deep=True): return {"n": s.n}
    def set_params(s, **p): return s
    def fit(s, X, y=None):
        C = np.einsum("ect,edt->cd", X, X) / (X.shape[0] * X.shape[2]); w, v = np.linalg.eigh(C); s.W = v[:, ::-1][:, :s.n].T; return s
    def transform(s, X): return np.einsum("kc,ect->ekt", s.W, X)
    def fit_transform(s, X, y=None): return s.fit(X).transform(X)

MODELS = {
    "LDA(shrink)": lambda: make_pipeline(WinMean(), StandardScaler(), LDA(solver="lsqr", shrinkage="auto")),
    "SVM-RBF": lambda: make_pipeline(WinMean(), StandardScaler(), SVC(C=1, class_weight="balanced", probability=True)),
    "LogReg-L2": lambda: make_pipeline(WinMean(), StandardScaler(), LogisticRegression(C=.01, class_weight="balanced", max_iter=2000)),
    "RandomForest": lambda: make_pipeline(WinMean(), RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0)),
    "xDAWN+Riemann+LR": lambda: make_pipeline(ChanPCA(60), XdawnCovariances(nfilter=4, estimator="oas"), TangentSpace(),
                                              LogisticRegression(class_weight="balanced", max_iter=2000)),
}
def evaluate(sub, names=MODELS):
    X, y, *_ = load(sub); X = X.astype(np.float64); out = {}
    for n in names:
        rows = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            m = MODELS[n]().fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
            rows.append(metrics(y[te], p))
        out[n] = mean_metrics(rows)
    return out
if __name__ == "__main__":
    subs = [int(a) for a in sys.argv[1:]] or subjects()
    import os; F = "../results/ml_within.json"; res = json.load(open(F)) if os.path.exists(F) else {}
    for s in subs:
        if str(s) in res: continue
        res[str(s)] = evaluate(s)
        print(s, {k: (round(x["ROC-AUC"], 3), round(x["Balanced Accuracy"], 3)) for k, x in res[str(s)].items()}, flush=True)
        json.dump(res, open("../results/ml_within.json", "w"))
