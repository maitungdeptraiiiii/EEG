# Mini project: EEG single-trial classification, frequent vs target (Nencki-Symfonia oddball, GigaDB 100990)

Data: D:\tttk_data (raw oddball in oddball\, preprocessed epochs in epochs\). 42 subjects, 127 ch, 660 stimuli/subject
(498 standard=frequent, 81 target, 81 deviant -> deviant is dropped). Epoch -200..800 ms, saved at 250 Hz -> (660,127,250).

Pipeline (src/):
  download.py       fetch raw files           preprocess.py   filter+ICA+epoch -> npz
  pipeline_bg.py    preprocess as downloads finish
  erp_plot.py       grand-average ERP         ml_baseline.py  LDA/SVM/LogReg/RF/xDAWN+Riemann (RQ1/RQ2)
  dl.py             EEGNet/Shallow/Deep4/Inception/Conformer, within-subject & LOSO (RQ1/RQ2)
  explain.py        gradient x input vs ERP (RQ3)
Run when all subjects are ready:
  python erp_plot.py; python ml_baseline.py; python dl.py within all; python dl.py loso all; python explain.py 1 2 3 ...
Metrics: balanced accuracy + ROC-AUC (classes are imbalanced 6:1).
