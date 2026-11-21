| Model | Type | Setting | N | ROC-AUC | Balanced Accuracy | Recall Target | Precision Target | Macro F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| LDA(shrink) | ML | within | 42 | 0.959 ± 0.056 | 0.895 ± 0.077 | 0.817 ± 0.133 | 0.839 ± 0.136 | 0.897 ± 0.078 | 0.951 ± 0.037 |
| SVM-RBF | ML | within | 42 | 0.946 ± 0.062 | 0.831 ± 0.097 | 0.680 ± 0.193 | 0.842 ± 0.125 | 0.853 ± 0.096 | 0.940 ± 0.029 |
| LogReg-L2 | ML | within | 42 | 0.952 ± 0.055 | 0.892 ± 0.070 | 0.833 ± 0.108 | 0.755 ± 0.141 | 0.874 ± 0.076 | 0.934 ± 0.044 |
| RandomForest | ML | within | 42 | 0.924 ± 0.065 | 0.767 ± 0.094 | 0.553 ± 0.190 | 0.821 ± 0.104 | 0.797 ± 0.093 | 0.921 ± 0.026 |
| xDAWN+Riemann+LR | ML | within | 42 | 0.964 ± 0.063 | 0.905 ± 0.075 | 0.841 ± 0.125 | 0.827 ± 0.141 | 0.900 ± 0.078 | 0.951 ± 0.040 |
| Conformer | DL | within | 42 | 0.933 ± 0.069 | 0.860 ± 0.081 | 0.823 ± 0.119 | 0.608 ± 0.147 | 0.808 ± 0.086 | 0.887 ± 0.059 |
| Deep4 | DL | within | 42 | 0.932 ± 0.071 | 0.866 ± 0.076 | 0.869 ± 0.098 | 0.558 ± 0.137 | 0.789 ± 0.088 | 0.864 ± 0.072 |
| EEGNet | DL | within | 42 | 0.955 ± 0.053 | 0.896 ± 0.063 | 0.863 ± 0.088 | 0.707 ± 0.131 | 0.859 ± 0.074 | 0.919 ± 0.051 |
| Inception | DL | within | 42 | 0.952 ± 0.058 | 0.894 ± 0.068 | 0.885 ± 0.091 | 0.648 ± 0.141 | 0.837 ± 0.082 | 0.900 ± 0.062 |
| Shallow | DL | within | 42 | 0.920 ± 0.094 | 0.837 ± 0.109 | 0.733 ± 0.193 | 0.689 ± 0.170 | 0.823 ± 0.103 | 0.911 ± 0.052 |

Best ML (within, ROC-AUC): xDAWN+Riemann+LR; best DL (within, ROC-AUC): EEGNet

Wilcoxon signed-rank, EEGNet vs xDAWN+Riemann+LR (n=42 subjects):

| Metric | mean diff (DL - ML) | p |
|---|---|---|
| ROC-AUC | -0.009 | 1.18e-05 |
| Balanced Accuracy | -0.009 | 0.00175 |
| Recall Target | +0.022 | 0.0304 |
| Precision Target | -0.120 | 1.96e-11 |
| Macro F1 | -0.042 | 7.69e-11 |
| Accuracy | -0.031 | 1.5e-11 |
