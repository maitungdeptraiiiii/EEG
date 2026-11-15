import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
subs = subjects(); R = {0: [], 1: []}
for s in subs:
    X, y, ch, sf, t0 = load(s)
    for c in (0, 1): R[c].append(X[y == c].mean(0))
t = t0 + np.arange(X.shape[2]) / sf
g = {c: np.mean(R[c], 0) for c in R}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, name in enumerate(["Fz", "Cz", "Pz"]):
    j = ch.index(name)
    for c, lab in ((0, "frequent"), (1, "target")): ax[i].plot(t * 1e3, g[c][j] * 1e6, label=lab)
    ax[i].axvline(0, c="k", lw=.5); ax[i].set_title(name); ax[i].set_xlabel("ms"); ax[i].set_ylabel("µV")
ax[0].legend(); plt.suptitle(f"Grand-average ERP (n={len(subs)} subjects)"); plt.tight_layout()
plt.savefig("../results/erp.png", dpi=130)
d = (g[1] - g[0]).mean(0) if False else None
i1, i2 = np.searchsorted(t, .3), np.searchsorted(t, .5)
print("P300 window mean diff at Pz (µV):", (g[1][ch.index('Pz')][i1:i2] - g[0][ch.index('Pz')][i1:i2]).mean() * 1e6)
