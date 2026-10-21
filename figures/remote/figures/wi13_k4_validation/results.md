# WI-13 results matrix (K=4)

## Policy A vs Policy B

| target | level | metric | policy_a | policy_b | delta_b_minus_a | separated | random_baseline | beats_random | mae | best_vicreg_beats_mae |
|---|---|---|---|---|---|---|---|---|---|---|
| phase_error | frame | r2 | +0.265 | +0.401 | +0.136 | False | +0.476 | False | +0.314 | True |
| t0_group | frame | balanced_accuracy | +0.349 | +0.541 | +0.192 | True | +0.438 | True | +0.533 | True |
| sos_mean_frame | frame | r2 | +0.890 | +0.881 | -0.009 | False | +0.786 | True | +0.751 | True |
| sos_std_frame | frame | r2 | +0.308 | +0.515 | +0.207 | False | +0.397 | True | +0.234 | True |
| cyst_fraction | frame | r2 | +0.464 | +0.318 | -0.146 | False | -0.002 | True | +0.083 | True |
| z_centre | crop | r2 | +0.724 | +0.755 | +0.031 | False | +0.603 | True | +0.379 | True |
| x_centre | crop | r2 | +0.524 | -0.001 | -0.525 | True | +0.117 | True | +0.003 | True |

## All cells

| encoder | target | level | metric | score | ci95 | n_eff | n_feat |
|---|---|---|---|---|---|---|---|
| vicreg_A | phase_error | frame | r2 | 0.265 | [0.095, 0.412] | 93 | 64 |
| vicreg_A | t0_group | frame | balanced_accuracy | 0.3491 | [0.281, 0.418] | 93 | 64 |
| vicreg_A | sos_mean_frame | frame | r2 | 0.8902 | [0.861, 0.914] | 93 | 64 |
| vicreg_A | sos_std_frame | frame | r2 | 0.3078 | [0.124, 0.411] | 93 | 64 |
| vicreg_A | cyst_fraction | frame | r2 | 0.4638 | [0.323, 0.570] | 93 | 64 |
| vicreg_A | z_centre | crop | r2 | 0.7238 | [0.707, 0.739] | 2976 | 64 |
| vicreg_A | x_centre | crop | r2 | 0.5239 | [0.500, 0.547] | 2976 | 64 |
| vicreg_B | phase_error | frame | r2 | 0.4012 | [0.216, 0.530] | 93 | 64 |
| vicreg_B | t0_group | frame | balanced_accuracy | 0.5407 | [0.483, 0.593] | 93 | 64 |
| vicreg_B | sos_mean_frame | frame | r2 | 0.8811 | [0.820, 0.921] | 93 | 64 |
| vicreg_B | sos_std_frame | frame | r2 | 0.5146 | [0.363, 0.616] | 93 | 64 |
| vicreg_B | cyst_fraction | frame | r2 | 0.3178 | [0.170, 0.421] | 93 | 64 |
| vicreg_B | z_centre | crop | r2 | 0.755 | [0.738, 0.769] | 2976 | 64 |
| vicreg_B | x_centre | crop | r2 | -0.0011 | [-0.003, -0.000] | 2976 | 64 |
| mae_norm | phase_error | frame | r2 | 0.2596 | [0.113, 0.363] | 93 | 64 |
| mae_norm | t0_group | frame | balanced_accuracy | 0.2954 | [0.224, 0.366] | 93 | 64 |
| mae_norm | sos_mean_frame | frame | r2 | 0.6607 | [0.547, 0.754] | 93 | 64 |
| mae_norm | sos_std_frame | frame | r2 | 0.1629 | [0.006, 0.275] | 93 | 64 |
| mae_norm | cyst_fraction | frame | r2 | -0.0009 | [-0.077, 0.042] | 93 | 64 |
| mae_norm | z_centre | crop | r2 | 0.3118 | [0.284, 0.337] | 2976 | 64 |
| mae_norm | x_centre | crop | r2 | -0.0014 | [-0.004, -0.000] | 2976 | 64 |
| mae_raw | phase_error | frame | r2 | 0.3141 | [0.138, 0.424] | 93 | 64 |
| mae_raw | t0_group | frame | balanced_accuracy | 0.5333 | [0.475, 0.588] | 93 | 64 |
| mae_raw | sos_mean_frame | frame | r2 | 0.7509 | [0.675, 0.811] | 93 | 64 |
| mae_raw | sos_std_frame | frame | r2 | 0.2335 | [0.066, 0.343] | 93 | 64 |
| mae_raw | cyst_fraction | frame | r2 | 0.0833 | [-0.074, 0.186] | 93 | 64 |
| mae_raw | z_centre | crop | r2 | 0.379 | [0.351, 0.404] | 2976 | 64 |
| mae_raw | x_centre | crop | r2 | 0.0027 | [-0.003, 0.008] | 2976 | 64 |
| random_resnet18 | phase_error | frame | r2 | 0.4759 | [0.315, 0.599] | 93 | 64 |
| random_resnet18 | t0_group | frame | balanced_accuracy | 0.438 | [0.370, 0.503] | 93 | 64 |
| random_resnet18 | sos_mean_frame | frame | r2 | 0.7859 | [0.693, 0.850] | 93 | 64 |
| random_resnet18 | sos_std_frame | frame | r2 | 0.3966 | [0.216, 0.531] | 93 | 64 |
| random_resnet18 | cyst_fraction | frame | r2 | -0.0023 | [-0.259, 0.181] | 93 | 64 |
| random_resnet18 | z_centre | crop | r2 | 0.6027 | [0.578, 0.625] | 2976 | 64 |
| random_resnet18 | x_centre | crop | r2 | 0.1167 | [0.087, 0.144] | 2976 | 64 |
| pca_raw | phase_error | frame | r2 | -0.0346 | [-0.094, -0.012] | 93 | 64 |
| pca_raw | t0_group | frame | balanced_accuracy | 0.2741 | [0.203, 0.347] | 93 | 64 |
| pca_raw | sos_mean_frame | frame | r2 | -0.0131 | [-0.076, -0.002] | 93 | 64 |
| pca_raw | sos_std_frame | frame | r2 | -0.0085 | [-0.071, 0.002] | 93 | 64 |
| pca_raw | cyst_fraction | frame | r2 | -0.0226 | [-0.082, -0.008] | 93 | 64 |
| pca_raw | z_centre | crop | r2 | -0.0007 | [-0.003, 0.000] | 2976 | 64 |
| pca_raw | x_centre | crop | r2 | -0.0037 | [-0.012, 0.001] | 2976 | 64 |
| trivial | phase_error | frame | r2 | 0.5028 | [0.336, 0.614] | 93 | 2 |
| trivial | t0_group | frame | balanced_accuracy | 0.3917 | [0.321, 0.463] | 93 | 2 |
| trivial | sos_mean_frame | frame | r2 | 0.6786 | [0.582, 0.752] | 93 | 2 |
| trivial | sos_std_frame | frame | r2 | 0.3049 | [0.162, 0.395] | 93 | 2 |
| trivial | cyst_fraction | frame | r2 | 0.2879 | [0.038, 0.444] | 93 | 2 |
| trivial | z_centre | crop | r2 | 0.515 | [0.493, 0.535] | 2976 | 2 |
| trivial | x_centre | crop | r2 | -0.001 | [-0.003, -0.000] | 2976 | 2 |

## Cells not run

| cell | reason |
|---|---|
| contains_cyst | needs per-pixel segmentation map (source NVIDIA frames absent) |
| cyst_fraction_crop | crop-level needs the segmentation map; frame-level cyst_fraction is run |
| sos_mean_crop | crop-level needs the per-pixel SoS map; frame-level sos_mean_frame is run |
| K=16 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
| K=8 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
| K=1 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
