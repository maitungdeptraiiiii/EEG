"""Run everything (standard vs target only) and keep resuming until all 42 subjects are evaluated.
Can be started while pipeline_kaggle.py is still preprocessing: each pass picks up newly finished subjects.
Usage (from src/): python run_all.py"""
import subprocess, sys, time, os
from common import subjects
N, MODELS = 42, ["EEGNet", "Shallow", "Deep4", "Inception", "Conformer"]
R = os.path.join("..", "results"); os.makedirs(R, exist_ok=True)
def run(args, log):
    with open(os.path.join(R, log), "a", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-u"] + args, stdout=f, stderr=subprocess.STDOUT)
while True:
    n = len(subjects()); print(time.strftime("%H:%M"), "pass with", n, "subjects", flush=True)
    run(["ml_baseline.py"], "ml.log")
    for m in MODELS: run(["dl.py", "within", m], "dl_within.log")
    if n >= N: break
    while len(subjects()) == n: time.sleep(60)
run(["erp_plot.py"], "erp.log"); run(["aggregate.py"], "aggregate.log")
print("ALLDONE", flush=True)
