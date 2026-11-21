"""Aggregate results (standard vs target) -> results/summary.md + results/comparison.png + results/summary.csv"""
import json, os, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from common import METRICS
R = "../results/"
def ld(f): return json.load(open(R + f)) if os.path.exists(R + f) else {}
def merge(p):
    o = {}
    for f in sorted(glob.glob(R + p)):
        for n, d in json.load(open(f)).items(): o.setdefault(n, {}).update(d)
    return o
ml = ld("ml_within.json"); dlw = merge("dl_within_*.json"); dll = merge("dl_loso_*.json")
ML_NAMES = sorted({n for d in ml.values() for n in d})
tab = {}   # name -> setting -> {sub: {metric: value}}
for s, d in ml.items():
    for n, v in d.items(): tab.setdefault(n, {}).setdefault("within", {})[int(s)] = v
for setting, dd in (("within", dlw), ("LOSO", dll)):
    for n, d in dd.items():
        for s, v in d.items(): tab.setdefault(n, {}).setdefault(setting, {})[int(s)] = v
def arr(r, m): return np.array([v[m] for v in r.values()])

lines = ["| Model | Type | Setting | N | " + " | ".join(METRICS) + " |", "|---|---|---|---|" + "---|" * len(METRICS)]
csv = ["model,type,setting,n," + ",".join(f"{m} mean,{m} sd" for m in METRICS)]
for n, d in tab.items():
    for st, r in d.items():
        typ = "ML" if n in ML_NAMES else "DL"
        lines.append(f"| {n} | {typ} | {st} | {len(r)} | " + " | ".join(f"{arr(r,m).mean():.3f} ± {arr(r,m).std():.3f}" for m in METRICS) + " |")
        csv.append(f"{n},{typ},{st},{len(r)}," + ",".join(f"{arr(r,m).mean():.4f},{arr(r,m).std():.4f}" for m in METRICS))
within = {n: d["within"] for n, d in tab.items() if "within" in d}
auc = lambda n: arr(within[n], "ROC-AUC").mean()
best_ml = max([n for n in within if n in ML_NAMES], key=auc, default=None)
best_dl = max([n for n in within if n not in ML_NAMES], key=auc, default=None)
lines += ["", f"Best ML (within, ROC-AUC): {best_ml}; best DL (within, ROC-AUC): {best_dl}", ""]
if best_ml and best_dl:
    s = sorted(set(within[best_ml]) & set(within[best_dl]))
    lines += [f"Wilcoxon signed-rank, {best_dl} vs {best_ml} (n={len(s)} subjects):", "",
              "| Metric | mean diff (DL - ML) | p |", "|---|---|---|"]
    for m in METRICS:
        a = [within[best_dl][i][m] for i in s]; b = [within[best_ml][i][m] for i in s]
        try: p = f"{wilcoxon(a, b).pvalue:.3g}"
        except ValueError: p = "n/a"
        lines.append(f"| {m} | {np.mean(a)-np.mean(b):+.3f} | {p} |")
open(R + "summary.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
open(R + "summary.csv", "w", encoding="utf-8").write("\n".join(csv) + "\n"); print("\n".join(lines))

fig, axs = plt.subplots(2, 3, figsize=(16, 8)); keys = [(n, st) for n, d in tab.items() for st in d]
for ax, m in zip(axs.ravel(), METRICS):
    ax.boxplot([arr(tab[n][st], m) for n, st in keys], showfliers=False)
    ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels([f"{n}" + ("" if st == "within" else f"\n{st}") for n, st in keys], fontsize=7, rotation=45, ha="right")
    ax.set_title(m); ax.grid(axis="y", alpha=.3)
    if m in ("ROC-AUC", "Balanced Accuracy"): ax.axhline(.5, c="gray", ls=":")
plt.suptitle("Standard vs Target, within-subject 5-fold CV (per-subject distribution)"); plt.tight_layout(); plt.savefig(R + "comparison.png", dpi=130)
