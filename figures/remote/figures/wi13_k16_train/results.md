# WI-13 results matrix (K=4)

## Policy A vs Policy B

| target | level | metric | policy_a | policy_b | delta_b_minus_a | separated | random_baseline | beats_random |
|---|---|---|---|---|---|---|---|---|
| phase_error | frame | r2 | +0.377 | +0.247 | -0.130 | True | +0.166 | True |
| t0_group | frame | balanced_accuracy | +0.390 | +0.466 | +0.076 | True | +0.405 | True |
| sos_mean_frame | frame | r2 | +0.931 | +0.849 | -0.082 | True | +0.550 | True |
| sos_std_frame | frame | r2 | +0.335 | +0.162 | -0.172 | True | +0.149 | True |
| cyst_fraction | frame | r2 | +0.214 | +0.126 | -0.088 | False | +0.047 | True |
| z_centre | crop | r2 | +0.429 | +0.681 | +0.253 | True | +0.624 | True |
| x_centre | crop | r2 | +0.789 | -0.000 | -0.789 | True | +0.136 | True |

## All cells

| encoder | target | level | metric | score | ci95 | n_eff | n_feat |
|---|---|---|---|---|---|---|---|
| vicreg_A | phase_error | frame | r2 | 0.3772 | [0.320, 0.429] | 830 | 64 |
| vicreg_A | t0_group | frame | balanced_accuracy | 0.3903 | [0.363, 0.417] | 830 | 64 |
| vicreg_A | sos_mean_frame | frame | r2 | 0.9308 | [0.922, 0.938] | 830 | 64 |
| vicreg_A | sos_std_frame | frame | r2 | 0.3347 | [0.277, 0.390] | 830 | 64 |
| vicreg_A | cyst_fraction | frame | r2 | 0.2137 | [0.157, 0.266] | 830 | 64 |
| vicreg_A | z_centre | crop | r2 | 0.4286 | [0.413, 0.444] | 6640 | 64 |
| vicreg_A | x_centre | crop | r2 | 0.7894 | [0.779, 0.800] | 6640 | 64 |
| vicreg_B | phase_error | frame | r2 | 0.2475 | [0.188, 0.300] | 830 | 64 |
| vicreg_B | t0_group | frame | balanced_accuracy | 0.4661 | [0.439, 0.495] | 830 | 64 |
| vicreg_B | sos_mean_frame | frame | r2 | 0.849 | [0.831, 0.866] | 830 | 64 |
| vicreg_B | sos_std_frame | frame | r2 | 0.1624 | [0.114, 0.206] | 830 | 64 |
| vicreg_B | cyst_fraction | frame | r2 | 0.1258 | [0.080, 0.170] | 830 | 64 |
| vicreg_B | z_centre | crop | r2 | 0.6813 | [0.669, 0.694] | 6640 | 64 |
| vicreg_B | x_centre | crop | r2 | -0.0001 | [-0.001, 0.000] | 6640 | 64 |
| random_resnet18 | phase_error | frame | r2 | 0.1656 | [0.120, 0.207] | 830 | 64 |
| random_resnet18 | t0_group | frame | balanced_accuracy | 0.4053 | [0.378, 0.432] | 830 | 64 |
| random_resnet18 | sos_mean_frame | frame | r2 | 0.5505 | [0.507, 0.588] | 830 | 64 |
| random_resnet18 | sos_std_frame | frame | r2 | 0.1493 | [0.110, 0.186] | 830 | 64 |
| random_resnet18 | cyst_fraction | frame | r2 | 0.0471 | [0.010, 0.081] | 830 | 64 |
| random_resnet18 | z_centre | crop | r2 | 0.624 | [0.604, 0.643] | 6640 | 64 |
| random_resnet18 | x_centre | crop | r2 | 0.1362 | [0.118, 0.153] | 6640 | 64 |
| pca_raw | phase_error | frame | r2 | -0.002 | [-0.008, -0.000] | 830 | 64 |
| pca_raw | t0_group | frame | balanced_accuracy | 0.3233 | [0.300, 0.347] | 830 | 64 |
| pca_raw | sos_mean_frame | frame | r2 | -0.004 | [-0.014, 0.001] | 830 | 64 |
| pca_raw | sos_std_frame | frame | r2 | -0.0027 | [-0.009, -0.001] | 830 | 64 |
| pca_raw | cyst_fraction | frame | r2 | -0.0045 | [-0.016, 0.002] | 830 | 64 |
| pca_raw | z_centre | crop | r2 | -0.0004 | [-0.001, -0.000] | 6640 | 64 |
| pca_raw | x_centre | crop | r2 | -0.0002 | [-0.001, 0.000] | 6640 | 64 |
| trivial | phase_error | frame | r2 | 0.1135 | [0.065, 0.158] | 830 | 2 |
| trivial | t0_group | frame | balanced_accuracy | 0.4294 | [0.406, 0.453] | 830 | 2 |
| trivial | sos_mean_frame | frame | r2 | 0.1089 | [0.072, 0.141] | 830 | 2 |
| trivial | sos_std_frame | frame | r2 | 0.099 | [0.059, 0.135] | 830 | 2 |
| trivial | cyst_fraction | frame | r2 | 0.1124 | [0.066, 0.158] | 830 | 2 |
| trivial | z_centre | crop | r2 | 0.6318 | [0.613, 0.649] | 6640 | 2 |
| trivial | x_centre | crop | r2 | -0.0002 | [-0.001, 0.000] | 6640 | 2 |

## Cells not run

| cell | reason |
|---|---|
| contains_cyst | needs per-pixel segmentation map (source NVIDIA frames absent) |
| cyst_fraction_crop | crop-level needs the segmentation map; frame-level cyst_fraction is run |
| sos_mean_crop | crop-level needs the per-pixel SoS map; frame-level sos_mean_frame is run |
| K=16 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
| K=8 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
| K=1 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
