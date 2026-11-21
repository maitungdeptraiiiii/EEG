#!/usr/bin/env bash
# Full re-run: standard vs target only, 6 metrics. Usage (from src/):  EEG_EPOCHS=/path/to/epochs bash run_seq.sh
cd "$(dirname "$0")"
python erp_plot.py > ../results/erp.log 2>&1
python ml_baseline.py >> ../results/ml.log 2>&1
for m in EEGNet Shallow Deep4 Inception Conformer; do python dl.py within $m >> ../results/dl_within.log 2>&1; done
python aggregate.py > ../results/aggregate.log 2>&1
echo ALLDONE >> ../results/dl_within.log
