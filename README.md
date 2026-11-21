# Mini project: EEG single-trial classification, **standard vs target** (Nencki-Symfonia oddball, GigaDB 100990)

Data: Kaggle mirror `trungdttik55/nencki-symfonia-eeg-oddball` (BIDS). 42 subjects, 127 ch, 1000 Hz.
Only standard (S 5) and target (S 6) are used; deviant (S 7) is discarded. Epoch -200..796 ms at 250 Hz -> (n, 127, 250).

Pipeline (src/):
  pipeline_kaggle.py  download (kagglehub, no login) + preprocess -> data/epochs/sub-XX.npz
  preprocess.py       notch, 0.1-40 Hz, 250 Hz, avg-ref, ICA (blinks), epochs
  common.py           loading (standard vs target only) + the 6 metrics
  erp_plot.py         grand-average ERP      ml_baseline.py  LDA/SVM/LogReg/RF/xDAWN+Riemann
  dl.py               EEGNet/Shallow/Deep4/Inception/Conformer (within & LOSO)
  aggregate.py        summary.md / summary.csv / comparison.png
  run_all.py          everything above, resumable
Run:  cd src; python pipeline_kaggle.py; python run_all.py
Metrics: ROC-AUC, Balanced Accuracy, Recall Target, Precision Target, Macro F1, Accuracy (threshold 0.5, target = positive).
Report: report.md
