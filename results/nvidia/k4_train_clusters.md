# WI-13 — unsupervised clustering, per encoder

k-means over standardised features. `silhouette` is cluster separation;
`ari_*` is agreement with a label the clustering never saw. Depth quartile
is the confound — a high ARI there means the encoder clusters by *where*
the crop came from, which is not a representation of tissue.

## k=2

| encoder | silhouette | ari_cyst | ari_depth_quartile | ari_t0_group |
|---|---|---|---|---|
| vicreg_A | +0.045 | +0.108 | +0.030 | +0.004 |
| vicreg_B | +0.019 | +0.008 | +0.008 | +0.016 |
| mae_norm | +0.065 | -0.000 | +0.063 | +0.002 |
| mae_raw | +0.043 | -0.009 | +0.097 | +0.009 |
| random_resnet18 | +0.431 | +0.007 | +0.109 | -0.001 |
| pca_raw | +0.749 | +0.001 | +0.000 | -0.000 |
| trivial | +0.525 | -0.003 | +0.143 | -0.000 |

## k=4

| encoder | silhouette | ari_cyst | ari_depth_quartile | ari_t0_group |
|---|---|---|---|---|
| vicreg_A | +0.031 | +0.080 | +0.124 | -0.002 |
| vicreg_B | +0.022 | +0.022 | +0.043 | +0.022 |
| mae_norm | +0.046 | +0.002 | +0.046 | -0.011 |
| mae_raw | +0.041 | +0.064 | +0.093 | -0.003 |
| random_resnet18 | +0.310 | +0.013 | +0.182 | +0.009 |
| pca_raw | +0.658 | +0.029 | +0.002 | -0.009 |
| trivial | +0.475 | +0.019 | +0.193 | +0.003 |

## k=8

| encoder | silhouette | ari_cyst | ari_depth_quartile | ari_t0_group |
|---|---|---|---|---|
| vicreg_A | -0.015 | +0.131 | +0.121 | +0.006 |
| vicreg_B | +0.053 | +0.147 | +0.076 | +0.015 |
| mae_norm | +0.026 | +0.009 | +0.029 | +0.001 |
| mae_raw | +0.016 | +0.032 | +0.072 | -0.002 |
| random_resnet18 | +0.203 | +0.015 | +0.143 | +0.012 |
| pca_raw | +0.645 | +0.022 | +0.001 | -0.005 |
| trivial | +0.452 | +0.034 | +0.158 | +0.014 |

## k=16

| encoder | silhouette | ari_cyst | ari_depth_quartile | ari_t0_group |
|---|---|---|---|---|
| vicreg_A | +0.003 | +0.081 | +0.078 | +0.004 |
| vicreg_B | +0.080 | +0.056 | +0.084 | +0.008 |
| mae_norm | +0.022 | +0.003 | +0.021 | -0.000 |
| mae_raw | -0.001 | +0.020 | +0.058 | -0.001 |
| random_resnet18 | +0.129 | +0.021 | +0.115 | +0.012 |
| pca_raw | +0.627 | +0.030 | +0.007 | -0.010 |
| trivial | +0.414 | +0.036 | +0.105 | +0.010 |

## DBSCAN

| encoder | eps | clusters | noise fraction |
|---|---|---|---|
| vicreg_A | 10.266 | 7 | 0.913 |
| vicreg_B | 11.276 | 7 | 0.975 |
| mae_norm | 18.711 | 1 | 0.740 |
| mae_raw | 14.868 | 1 | 0.800 |
| random_resnet18 | 3.737 | 1 | 0.996 |
| pca_raw | 0.456 | 1 | 0.997 |
| trivial | 0.010 | 0 | 1.000 |
