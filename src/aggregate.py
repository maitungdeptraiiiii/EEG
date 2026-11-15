"""Aggregate results -> results/summary.md + results/comparison.png"""
import json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
R = "../results/"
def ld(f): return json.load(open(R + f)) if os.path.exists(R + f) else {}
ml = ld("ml_within.json"); import glob
def merge(p):
    o = {}
    for f in glob.glob(R + p):
        for n, d in json.load(open(f)).items(): o.setdefault(n, {}).update(d)
    return o
dlw = merge("dl_within_*.json"); dll = merge("dl_loso_*.json")
tab = {}   # name -> (setting) -> {sub: (ba, auc)}
for s, d in ml.items():
    for n, v in d.items(): tab.setdefault(n, {}).setdefault("within", {})[int(s)] = v
for setting, dd in (("within", dlw), ("LOSO", dll)):
    for n, d in dd.items():
        for s, v in d.items(): tab.setdefault(n, {}).setdefault(setting, {})[int(s)] = v
lines = ["| Model | Setting | N | Balanced acc (mean±SD) | AUC (mean±SD) |", "|---|---|---|---|---|"]
for n, d in tab.items():
    for st, r in d.items():
        a = np.array(list(r.values())); lines.append(f"| {n} | {st} | {len(a)} | {a[:,0].mean():.3f} ± {a[:,0].std():.3f} | {a[:,1].mean():.3f} ± {a[:,1].std():.3f} |")
within = {n: d["within"] for n, d in tab.items() if "within" in d}
if within:
    best_ml = max([n for n in within if n in (ml[next(iter(ml))] if ml else {})], key=lambda n: np.mean([v[1] for v in within[n].values()]), default=None)
    best_dl = max([n for n in within if n in dlw], key=lambda n: np.mean([v[1] for v in within[n].values()]), default=None)
    lines += ["", f"Best ML (within, AUC): {best_ml}; best DL (within, AUC): {best_dl}"]
    if best_ml and best_dl:
        s = sorted(set(within[best_ml]) & set(within[best_dl]))
        for k, m in ((0, "balanced acc"), (1, "AUC")):
            a = [within[best_dl][i][k] for i in s]; b = [within[best_ml][i][k] for i in s]
            lines.append(f"Wilcoxon {best_dl} vs {best_ml} ({m}, n={len(s)}): mean diff {np.mean(a)-np.mean(b):+.3f}, p={wilcoxon(a, b).pvalue:.4g}")
open(R + "summary.md", "w").write("\n".join(lines)); print("\n".join(lines))
fig, ax = plt.subplots(figsize=(10, 4)); i = 0; ticks = []
for n, d in tab.items():
    for st, r in d.items():
        v = [x[1] for x in r.values()]; ax.boxplot(v, positions=[i], widths=.6, showfliers=False); ticks.append(f"{n}\n{st}"); i += 1
ax.set_xticks(range(i)); ax.set_xticklabels(ticks, fontsize=6, rotation=45); ax.set_ylabel("AUC per subject"); ax.axhline(.5, c="gray", ls=":")
plt.tight_layout(); plt.savefig(R + "comparison.png", dpi=130)
