# WI-13 results matrix (K=4)

## Policy A vs Policy B

| target | level | metric | policy_a | policy_b | delta_b_minus_a | separated | random_baseline | beats_random | mae | best_vicreg_beats_mae |
|---|---|---|---|---|---|---|---|---|---|---|
| phase_error | frame | r2 | +0.258 | +0.386 | +0.129 | False | +0.204 | True | None | None |
| t0_group | frame | balanced_accuracy | +0.317 | +0.465 | +0.148 | True | +0.339 | True | None | None |
| sos_mean_frame | frame | r2 | +0.911 | +0.872 | -0.039 | False | +0.701 | True | None | None |
| sos_std_frame | frame | r2 | +0.308 | +0.469 | +0.161 | False | +0.212 | True | None | None |
| cyst_fraction | frame | r2 | +0.539 | +0.581 | +0.041 | False | -0.056 | True | None | None |
| z_centre | crop | r2 | +0.793 | +0.538 | -0.255 | True | +0.440 | True | None | None |
| x_centre | crop | r2 | +0.530 | -0.001 | -0.531 | True | +0.021 | True | None | None |
| contains_cyst | crop | balanced_accuracy | +0.927 | +0.745 | -0.182 | True | +0.779 | True | None | None |
| cyst_fraction_crop | crop | r2 | +0.843 | +0.726 | -0.117 | True | +0.329 | True | None | None |
| sos_mean_crop | crop | r2 | +0.760 | +0.562 | -0.198 | True | +0.442 | True | None | None |

## All cells

| encoder | target | level | metric | score | ci95 | n_eff | n_feat |
|---|---|---|---|---|---|---|---|
| vicreg_A | phase_error | frame | r2 | 0.2576 | [0.076, 0.396] | 93 | 64 |
| vicreg_A | t0_group | frame | balanced_accuracy | 0.3167 | [0.248, 0.386] | 93 | 64 |
| vicreg_A | sos_mean_frame | frame | r2 | 0.911 | [0.871, 0.939] | 93 | 64 |
| vicreg_A | sos_std_frame | frame | r2 | 0.3085 | [0.141, 0.421] | 93 | 64 |
| vicreg_A | cyst_fraction | frame | r2 | 0.5391 | [0.366, 0.636] | 93 | 64 |
| vicreg_A | z_centre | crop | r2 | 0.7928 | [0.778, 0.808] | 2976 | 64 |
| vicreg_A | x_centre | crop | r2 | 0.5302 | [0.507, 0.553] | 2976 | 64 |
| vicreg_A | contains_cyst | crop | balanced_accuracy | 0.9269 | [0.915, 0.937] | 2592 | 64 |
| vicreg_A | cyst_fraction_crop | crop | r2 | 0.8434 | [0.818, 0.864] | 2592 | 64 |
| vicreg_A | sos_mean_crop | crop | r2 | 0.7602 | [0.741, 0.778] | 2976 | 64 |
| vicreg_B | phase_error | frame | r2 | 0.3861 | [0.224, 0.501] | 93 | 64 |
| vicreg_B | t0_group | frame | balanced_accuracy | 0.4648 | [0.401, 0.529] | 93 | 64 |
| vicreg_B | sos_mean_frame | frame | r2 | 0.8721 | [0.816, 0.911] | 93 | 64 |
| vicreg_B | sos_std_frame | frame | r2 | 0.4694 | [0.252, 0.605] | 93 | 64 |
| vicreg_B | cyst_fraction | frame | r2 | 0.5805 | [0.420, 0.673] | 93 | 64 |
| vicreg_B | z_centre | crop | r2 | 0.5378 | [0.513, 0.561] | 2976 | 64 |
| vicreg_B | x_centre | crop | r2 | -0.0011 | [-0.003, -0.000] | 2976 | 64 |
| vicreg_B | contains_cyst | crop | balanced_accuracy | 0.7446 | [0.727, 0.762] | 2592 | 64 |
| vicreg_B | cyst_fraction_crop | crop | r2 | 0.7263 | [0.696, 0.753] | 2592 | 64 |
| vicreg_B | sos_mean_crop | crop | r2 | 0.5625 | [0.536, 0.589] | 2976 | 64 |
| random_resnet18 | phase_error | frame | r2 | 0.2039 | [0.019, 0.351] | 93 | 64 |
| random_resnet18 | t0_group | frame | balanced_accuracy | 0.3389 | [0.272, 0.407] | 93 | 64 |
| random_resnet18 | sos_mean_frame | frame | r2 | 0.7008 | [0.568, 0.790] | 93 | 64 |
| random_resnet18 | sos_std_frame | frame | r2 | 0.212 | [0.054, 0.315] | 93 | 64 |
| random_resnet18 | cyst_fraction | frame | r2 | -0.056 | [-0.155, 0.013] | 93 | 64 |
| random_resnet18 | z_centre | crop | r2 | 0.44 | [0.417, 0.463] | 2976 | 64 |
| random_resnet18 | x_centre | crop | r2 | 0.0214 | [0.007, 0.035] | 2976 | 64 |
| random_resnet18 | contains_cyst | crop | balanced_accuracy | 0.7794 | [0.762, 0.796] | 2592 | 64 |
| random_resnet18 | cyst_fraction_crop | crop | r2 | 0.329 | [0.277, 0.369] | 2592 | 64 |
| random_resnet18 | sos_mean_crop | crop | r2 | 0.4416 | [0.417, 0.464] | 2976 | 64 |
| pca_raw | phase_error | frame | r2 | -0.0402 | [-0.104, -0.016] | 93 | 64 |
| pca_raw | t0_group | frame | balanced_accuracy | 0.3176 | [0.249, 0.389] | 93 | 64 |
| pca_raw | sos_mean_frame | frame | r2 | -0.0119 | [-0.075, -0.001] | 93 | 64 |
| pca_raw | sos_std_frame | frame | r2 | -0.0084 | [-0.069, 0.004] | 93 | 64 |
| pca_raw | cyst_fraction | frame | r2 | -0.0295 | [-0.092, -0.008] | 93 | 64 |
| pca_raw | z_centre | crop | r2 | -0.0007 | [-0.003, 0.000] | 2976 | 64 |
| pca_raw | x_centre | crop | r2 | -0.0032 | [-0.010, 0.001] | 2976 | 64 |
| pca_raw | contains_cyst | crop | balanced_accuracy | 0.509 | [0.501, 0.517] | 2592 | 64 |
| pca_raw | cyst_fraction_crop | crop | r2 | -0.0042 | [-0.008, -0.001] | 2592 | 64 |
| pca_raw | sos_mean_crop | crop | r2 | -0.0113 | [-0.014, -0.009] | 2976 | 64 |
| trivial | phase_error | frame | r2 | 0.3727 | [0.165, 0.506] | 93 | 2 |
| trivial | t0_group | frame | balanced_accuracy | 0.3417 | [0.271, 0.411] | 93 | 2 |
| trivial | sos_mean_frame | frame | r2 | 0.7692 | [0.697, 0.820] | 93 | 2 |
| trivial | sos_std_frame | frame | r2 | 0.198 | [0.026, 0.300] | 93 | 2 |
| trivial | cyst_fraction | frame | r2 | 0.2241 | [-0.042, 0.380] | 93 | 2 |
| trivial | z_centre | crop | r2 | 0.4038 | [0.380, 0.426] | 2976 | 2 |
| trivial | x_centre | crop | r2 | -0.001 | [-0.003, -0.000] | 2976 | 2 |
| trivial | contains_cyst | crop | balanced_accuracy | 0.7853 | [0.769, 0.801] | 2592 | 2 |
| trivial | cyst_fraction_crop | crop | r2 | 0.2261 | [0.180, 0.267] | 2592 | 2 |
| trivial | sos_mean_crop | crop | r2 | 0.4183 | [0.395, 0.440] | 2976 | 2 |

## Cells not run

| cell | reason |
|---|---|
