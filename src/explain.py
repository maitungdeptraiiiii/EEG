"""RQ3: is what the DL model learned explainable?  Gradient x input attribution (target class)
for EEGNet, compared with the target-minus-frequent ERP difference.  Usage: python explain.py SUBJ [SUBJ...]"""
import sys, numpy as np, torch, mne, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from dl import fit_predict, dev
from common import *
mne.set_log_level("WARNING")
def attribution(sub, name="EEGNet"):
    X, y, ch, sf, t0 = load(sub)
    # train on 80%, attribute on held-out correctly-classified target trials
    rng = np.random.RandomState(0); idx = rng.permutation(len(X)); k = int(.8 * len(X)); tr, te = idx[:k], idx[k:]
    p, model = fit_predict(name, X[tr], y[tr], X[te])
    mu, sd = X[tr].mean((0, 2), keepdims=True), X[tr].std((0, 2), keepdims=True) + 1e-8
    sel = te[(y[te] == 1)]
    xt = torch.tensor((X[sel] - mu) / sd, dtype=torch.float32, device=dev, requires_grad=True)
    model.eval(); model(xt)[:, 1].sum().backward()
    a = (xt.grad * xt).detach().abs().cpu().numpy().mean(0)       # (ch, time)
    erp = X[y == 1].mean(0) - X[y == 0].mean(0)
    return a, erp, ch, sf, t0
if __name__ == "__main__":
    subs = [int(s) for s in sys.argv[1:]]; A = []; E = []
    for s in subs:
        a, erp, ch, sf, t0 = attribution(s); A.append(a / a.max()); E.append(erp)
    a, erp = np.mean(A, 0), np.mean(E, 0); t = (t0 + np.arange(a.shape[1]) / sf) * 1e3
    info = mne.create_info(ch, sf, "eeg"); info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    ok = [i for i, c in enumerate(info.ch_names) if info["chs"][i]["loc"][:3].any() and c not in ("TP9 LEFT EAR","TP10 RIGHT EAR","O9","O10")]
    info = mne.pick_info(info, ok)
    w = (t >= 300) & (t <= 500)
    fig = plt.figure(figsize=(14, 4)); ax = fig.add_subplot(131)
    ax.plot(t, a.mean(0), label="mean |grad×input|"); ax2 = ax.twinx(); ax2.plot(t, np.abs(erp).mean(0) * 1e6, "r", label="mean |ERP diff| (µV)")
    ax.set_xlabel("ms"); ax.set_title("Time profile: attribution vs ERP effect"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    for k, (v, ttl) in enumerate([(a[:, w].mean(1), "Model attribution 300-500 ms"), (np.abs(erp[:, w]).mean(1), "|ERP difference| 300-500 ms")]):
        mne.viz.plot_topomap(v[ok], info, axes=fig.add_subplot(132 + k), show=False); fig.axes[-1].set_title(ttl)
    plt.tight_layout(); plt.savefig("../results/explain_eegnet.png", dpi=130)
    from scipy.stats import spearmanr
    print("spatial corr (attribution vs |ERP diff|, 300-500ms): r=%.3f" % spearmanr(a[:, w].mean(1), np.abs(erp[:, w]).mean(1))[0])
    print("temporal corr: r=%.3f" % spearmanr(a.mean(0), np.abs(erp).mean(0))[0])
