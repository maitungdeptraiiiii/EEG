"""Deep learning for frequent-vs-target. Modes:
  within : per-subject stratified 5-fold CV          python dl.py within MODEL [subjects...]
  loso   : leave-one-subject-out (pooled training)   python dl.py loso MODEL [test subjects...]
MODEL in EEGNet, Shallow, Deep4, Inception, Conformer, or 'all'."""
import sys, json, copy, numpy as np, torch, torch.nn as nn, warnings; warnings.filterwarnings("ignore")
import braindecode.models as M
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from common import *
dev = "cuda" if torch.cuda.is_available() else "cpu"
def build(n, nch=127, nt=250):
    kw = dict(n_chans=nch, n_outputs=2, n_times=nt)
    return {"EEGNet": lambda: M.EEGNet(**kw), "Shallow": lambda: M.ShallowFBCSPNet(**kw, final_conv_length="auto"),
            "Deep4": lambda: M.Deep4Net(**kw, final_conv_length="auto"), "Inception": lambda: M.EEGInceptionERP(**kw, sfreq=250),
            "Conformer": lambda: M.EEGConformer(**kw)}[n]()
def prob(model, X, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs): out.append(torch.softmax(model(X[i:i+bs].to(dev)), 1)[:, 1].cpu())
    return torch.cat(out).numpy()
def fit_predict(name, Xtr, ytr, Xte, epochs=40, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-8   # per-channel z-score (train stats)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    a, b = train_test_split(np.arange(len(Xtr)), test_size=.15, stratify=ytr, random_state=seed)
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    Xa, ya, Xb, yb = T(Xtr[a]), torch.tensor(ytr[a]), T(Xtr[b]), ytr[b]
    model = build(name, Xtr.shape[1], Xtr.shape[2]).to(dev)
    w = torch.tensor(len(ya) / (2 * np.bincount(ya.numpy())), dtype=torch.float32).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, bst = -1, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xa))
        for i in range(0, len(perm), 64):
            j = perm[i:i+64]
            if len(j) < 2: continue
            opt.zero_grad(); lossf(model(Xa[j].to(dev)), ya[j].to(dev)).backward(); opt.step()
        s = roc_auc_score(yb, prob(model, Xb))            # model selection on validation AUC
        if s > best: best, bst = s, copy.deepcopy(model.state_dict())
    model.load_state_dict(bst)
    return prob(model, T(Xte)), model
def score(y, p): return balanced_accuracy_score(y, p > .5), roc_auc_score(y, p)
def within(name, sub):
    X, y, *_ = load(sub); r = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p, _ = fit_predict(name, X[tr], y[tr], X[te]); r.append(score(y[te], p))
    return tuple(np.mean(r, 0))
def loso(name, test, subs):
    d = {s: load(s) for s in subs}
    Xtr = np.concatenate([d[s][0] for s in subs if s != test]); ytr = np.concatenate([d[s][1] for s in subs if s != test])
    p, _ = fit_predict(name, Xtr, ytr, d[test][0], epochs=20); return score(d[test][1], p)
if __name__ == "__main__":
    mode, mn = sys.argv[1], sys.argv[2]; names = ["EEGNet", "Shallow", "Deep4", "Inception", "Conformer"] if mn == "all" else [mn]
    subs = [int(a) for a in sys.argv[3:]] or subjects()
    import os; F = f"../results/dl_{mode}_{mn}.json"; res = json.load(open(F)) if os.path.exists(F) else {}
    for n in names:
        for s in subs:
            if str(s) in res.get(n, {}): continue
            res.setdefault(n, {})[str(s)] = within(n, s) if mode == "within" else loso(n, s, subjects())
            print(mode, n, s, tuple(round(float(v), 3) for v in res[n][str(s)]), flush=True)
            json.dump(res, open(F, "w"))
