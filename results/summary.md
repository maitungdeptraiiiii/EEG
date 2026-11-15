| Model | Setting | N | Balanced acc (mean±SD) | AUC (mean±SD) |
|---|---|---|---|---|
| LDA(shrink) | within | 42 | 0.896 ± 0.076 | 0.959 ± 0.055 |
| SVM-RBF | within | 42 | 0.834 ± 0.098 | 0.947 ± 0.062 |
| LogReg-L2 | within | 42 | 0.894 ± 0.069 | 0.953 ± 0.054 |
| RandomForest | within | 42 | 0.651 ± 0.093 | 0.922 ± 0.067 |
| xDAWN+Riemann+LR | within | 42 | 0.906 ± 0.073 | 0.964 ± 0.062 |
| EEGNet | within | 42 | 0.895 ± 0.063 | 0.956 ± 0.052 |
| Inception | within | 42 | 0.891 ± 0.068 | 0.954 ± 0.057 |

Best ML (within, AUC): xDAWN+Riemann+LR; best DL (within, AUC): EEGNet
Wilcoxon EEGNet vs xDAWN+Riemann+LR (balanced acc, n=42): mean diff -0.010, p=0.004498
Wilcoxon EEGNet vs xDAWN+Riemann+LR (AUC, n=42): mean diff -0.008, p=0.0001532