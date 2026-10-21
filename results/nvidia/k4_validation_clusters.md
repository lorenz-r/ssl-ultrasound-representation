# WI-13 — unsupervised clustering, per encoder

k-means over standardised features. `silhouette` is cluster separation;
`ari_*` is agreement with a label the clustering never saw. Depth quartile
is the confound — a high ARI there means the encoder clusters by *where*
the crop came from, which is not a representation of tissue.

## k=2

| encoder | silhouette | ari_cyst | ari_depth_quartile | ari_t0_group |
|---|---|---|---|---|
| vicreg_A | +0.048 | +0.082 | +0.031 | +0.004 |
| vicreg_B | +0.523 | -0.001 | +0.000 | -0.002 |
| mae_norm | +0.065 | +0.000 | +0.054 | -0.001 |
| mae_raw | +0.043 | -0.009 | +0.099 | +0.002 |
| random_resnet18 | +0.452 | +0.018 | +0.072 | +0.011 |
| pca_raw | +0.894 | +0.001 | -0.000 | -0.000 |
| trivial | +0.522 | +0.004 | +0.108 | +0.005 |

## k=4

| encoder | silhouette | ari_cyst | ari_depth_quartile | ari_t0_group |
|---|---|---|---|---|
| vicreg_A | +0.005 | +0.103 | +0.139 | +0.003 |
| vicreg_B | +0.025 | +0.028 | +0.037 | +0.009 |
| mae_norm | +0.046 | +0.004 | +0.038 | -0.004 |
| mae_raw | +0.039 | +0.056 | +0.102 | +0.002 |
| random_resnet18 | +0.327 | +0.013 | +0.185 | +0.014 |
| pca_raw | +0.890 | +0.003 | -0.000 | -0.000 |
| trivial | +0.496 | +0.011 | +0.195 | +0.005 |

## k=8

| encoder | silhouette | ari_cyst | ari_depth_quartile | ari_t0_group |
|---|---|---|---|---|
| vicreg_A | -0.001 | +0.129 | +0.113 | +0.003 |
| vicreg_B | +0.047 | +0.030 | +0.060 | +0.005 |
| mae_norm | +0.024 | +0.002 | +0.033 | -0.000 |
| mae_raw | +0.020 | +0.031 | +0.080 | +0.001 |
| random_resnet18 | +0.207 | +0.011 | +0.157 | +0.017 |
| pca_raw | +0.891 | +0.019 | +0.000 | -0.003 |
| trivial | +0.454 | +0.028 | +0.175 | +0.019 |

## k=16

| encoder | silhouette | ari_cyst | ari_depth_quartile | ari_t0_group |
|---|---|---|---|---|
| vicreg_A | +0.027 | +0.077 | +0.077 | +0.002 |
| vicreg_B | +0.076 | +0.056 | +0.084 | +0.007 |
| mae_norm | +0.021 | +0.003 | +0.023 | +0.000 |
| mae_raw | +0.004 | +0.020 | +0.064 | +0.001 |
| random_resnet18 | +0.139 | +0.013 | +0.132 | +0.013 |
| pca_raw | +0.891 | +0.012 | +0.000 | -0.002 |
| trivial | +0.429 | +0.030 | +0.126 | +0.012 |

## DBSCAN

| encoder | eps | clusters | noise fraction |
|---|---|---|---|
| vicreg_A | 9.754 | 3 | 0.945 |
| vicreg_B | 12.101 | 13 | 0.924 |
| mae_norm | 18.968 | 4 | 0.779 |
| mae_raw | 16.016 | 1 | 0.744 |
| random_resnet18 | 3.572 | 0 | 1.000 |
| pca_raw | 53.388 | 1 | 0.000 |
| trivial | 0.009 | 1 | 0.997 |
