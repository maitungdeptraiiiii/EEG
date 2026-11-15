cd /c/Users/Admin/Desktop/tttk/src
python ml_baseline.py >> ../results/ml.log 2>&1
for m in EEGNet Inception Shallow Deep4 Conformer; do python dl.py within $m >> ../results/dl_within.log 2>&1; done
echo ALLDONE >> ../results/dl_within.log
