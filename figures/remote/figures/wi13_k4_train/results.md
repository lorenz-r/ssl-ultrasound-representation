# WI-13 results matrix (K=4)

## Policy A vs Policy B

| target | level | metric | policy_a | policy_b | delta_b_minus_a | separated | random_baseline | beats_random | mae | best_vicreg_beats_mae |
|---|---|---|---|---|---|---|---|---|---|---|
| phase_error | frame | r2 | +0.366 | +0.438 | +0.072 | False | +0.393 | True | +0.346 | True |
| t0_group | frame | balanced_accuracy | +0.398 | +0.526 | +0.128 | True | +0.427 | True | +0.428 | True |
| sos_mean_frame | frame | r2 | +0.899 | +0.865 | -0.034 | True | +0.800 | True | +0.800 | True |
| sos_std_frame | frame | r2 | +0.277 | +0.339 | +0.062 | False | +0.273 | True | +0.270 | True |
| cyst_fraction | frame | r2 | +0.331 | +0.312 | -0.019 | False | +0.128 | True | +0.097 | True |
| z_centre | crop | r2 | +0.699 | +0.756 | +0.057 | True | +0.645 | True | +0.388 | True |
| x_centre | crop | r2 | +0.537 | -0.001 | -0.538 | True | +0.177 | True | +0.013 | True |

## All cells

| encoder | target | level | metric | score | ci95 | n_eff | n_feat |
|---|---|---|---|---|---|---|---|
| vicreg_A | phase_error | frame | r2 | 0.3662 | [0.315, 0.412] | 830 | 64 |
| vicreg_A | t0_group | frame | balanced_accuracy | 0.3982 | [0.374, 0.421] | 830 | 64 |
| vicreg_A | sos_mean_frame | frame | r2 | 0.8991 | [0.887, 0.910] | 830 | 64 |
| vicreg_A | sos_std_frame | frame | r2 | 0.2771 | [0.224, 0.323] | 830 | 64 |
| vicreg_A | cyst_fraction | frame | r2 | 0.3313 | [0.276, 0.381] | 830 | 64 |
| vicreg_A | z_centre | crop | r2 | 0.6993 | [0.688, 0.711] | 6640 | 64 |
| vicreg_A | x_centre | crop | r2 | 0.5371 | [0.523, 0.551] | 6640 | 64 |
| vicreg_B | phase_error | frame | r2 | 0.4384 | [0.385, 0.488] | 830 | 64 |
| vicreg_B | t0_group | frame | balanced_accuracy | 0.5262 | [0.495, 0.558] | 830 | 64 |
| vicreg_B | sos_mean_frame | frame | r2 | 0.8648 | [0.850, 0.879] | 830 | 64 |
| vicreg_B | sos_std_frame | frame | r2 | 0.3394 | [0.286, 0.388] | 830 | 64 |
| vicreg_B | cyst_fraction | frame | r2 | 0.3123 | [0.255, 0.367] | 830 | 64 |
| vicreg_B | z_centre | crop | r2 | 0.7561 | [0.746, 0.766] | 6640 | 64 |
| vicreg_B | x_centre | crop | r2 | -0.0005 | [-0.001, -0.000] | 6640 | 64 |
| mae_norm | phase_error | frame | r2 | 0.2562 | [0.200, 0.307] | 830 | 64 |
| mae_norm | t0_group | frame | balanced_accuracy | 0.374 | [0.345, 0.404] | 830 | 64 |
| mae_norm | sos_mean_frame | frame | r2 | 0.8609 | [0.846, 0.875] | 830 | 64 |
| mae_norm | sos_std_frame | frame | r2 | 0.1803 | [0.136, 0.222] | 830 | 64 |
| mae_norm | cyst_fraction | frame | r2 | 0.0248 | [0.000, 0.047] | 830 | 64 |
| mae_norm | z_centre | crop | r2 | 0.3142 | [0.297, 0.332] | 6640 | 64 |
| mae_norm | x_centre | crop | r2 | 0.0022 | [-0.000, 0.004] | 6640 | 64 |
| mae_raw | phase_error | frame | r2 | 0.346 | [0.287, 0.398] | 830 | 64 |
| mae_raw | t0_group | frame | balanced_accuracy | 0.4283 | [0.403, 0.454] | 830 | 64 |
| mae_raw | sos_mean_frame | frame | r2 | 0.7998 | [0.779, 0.820] | 830 | 64 |
| mae_raw | sos_std_frame | frame | r2 | 0.2699 | [0.216, 0.315] | 830 | 64 |
| mae_raw | cyst_fraction | frame | r2 | 0.0966 | [0.059, 0.133] | 830 | 64 |
| mae_raw | z_centre | crop | r2 | 0.3882 | [0.372, 0.405] | 6640 | 64 |
| mae_raw | x_centre | crop | r2 | 0.0127 | [0.007, 0.018] | 6640 | 64 |
| random_resnet18 | phase_error | frame | r2 | 0.393 | [0.337, 0.445] | 830 | 64 |
| random_resnet18 | t0_group | frame | balanced_accuracy | 0.4269 | [0.402, 0.451] | 830 | 64 |
| random_resnet18 | sos_mean_frame | frame | r2 | 0.7998 | [0.778, 0.819] | 830 | 64 |
| random_resnet18 | sos_std_frame | frame | r2 | 0.273 | [0.221, 0.317] | 830 | 64 |
| random_resnet18 | cyst_fraction | frame | r2 | 0.1279 | [0.080, 0.170] | 830 | 64 |
| random_resnet18 | z_centre | crop | r2 | 0.6452 | [0.632, 0.658] | 6640 | 64 |
| random_resnet18 | x_centre | crop | r2 | 0.1773 | [0.159, 0.196] | 6640 | 64 |
| pca_raw | phase_error | frame | r2 | -0.0068 | [-0.038, 0.021] | 830 | 64 |
| pca_raw | t0_group | frame | balanced_accuracy | 0.3179 | [0.302, 0.334] | 830 | 64 |
| pca_raw | sos_mean_frame | frame | r2 | -0.0014 | [-0.007, -0.000] | 830 | 64 |
| pca_raw | sos_std_frame | frame | r2 | -0.0105 | [-0.032, 0.005] | 830 | 64 |
| pca_raw | cyst_fraction | frame | r2 | 0.0012 | [-0.010, 0.010] | 830 | 64 |
| pca_raw | z_centre | crop | r2 | -0.0005 | [-0.001, -0.000] | 6640 | 64 |
| pca_raw | x_centre | crop | r2 | -0.0001 | [-0.001, 0.000] | 6640 | 64 |
| trivial | phase_error | frame | r2 | 0.3997 | [0.340, 0.453] | 830 | 2 |
| trivial | t0_group | frame | balanced_accuracy | 0.397 | [0.372, 0.420] | 830 | 2 |
| trivial | sos_mean_frame | frame | r2 | 0.6708 | [0.637, 0.699] | 830 | 2 |
| trivial | sos_std_frame | frame | r2 | 0.28 | [0.233, 0.322] | 830 | 2 |
| trivial | cyst_fraction | frame | r2 | 0.17 | [0.115, 0.219] | 830 | 2 |
| trivial | z_centre | crop | r2 | 0.5414 | [0.527, 0.555] | 6640 | 2 |
| trivial | x_centre | crop | r2 | -0.0001 | [-0.001, -0.000] | 6640 | 2 |

## Cells not run

| cell | reason |
|---|---|
| contains_cyst | needs per-pixel segmentation map (source NVIDIA frames absent) |
| cyst_fraction_crop | crop-level needs the segmentation map; frame-level cyst_fraction is run |
| sos_mean_crop | crop-level needs the per-pixel SoS map; frame-level sos_mean_frame is run |
| K=16 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
| K=8 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
| K=1 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
