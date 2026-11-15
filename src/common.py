import numpy as np, glob, os
EP = r"D:\tttk_data\epochs"
def load(sub):
    d = np.load(f"{EP}\sub-{sub:02d}.npz")
    X, y = d["X"], d["y"]
    k = y != 2                       # frequent (0) vs target (1); drop deviant
    return X[k], y[k], list(d["ch"]), float(d["sfreq"]), float(d["tmin"])
def subjects():
    return sorted(int(os.path.basename(p)[4:6]) for p in glob.glob(EP + "\sub-*.npz"))
