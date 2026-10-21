# WI-13 results matrix (K=4)

## Policy A vs Policy B

| target | level | metric | policy_a | policy_b | delta_b_minus_a | separated | random_baseline | beats_random |
|---|---|---|---|---|---|---|---|---|
| phase_error | frame | r2 | +0.258 | +0.386 | +0.129 | False | +0.204 | True |
| t0_group | frame | balanced_accuracy | +0.317 | +0.465 | +0.148 | True | +0.339 | True |
| sos_mean_frame | frame | r2 | +0.911 | +0.872 | -0.039 | False | +0.701 | True |
| sos_std_frame | frame | r2 | +0.308 | +0.469 | +0.161 | False | +0.212 | True |
| cyst_fraction | frame | r2 | +0.539 | +0.581 | +0.041 | False | -0.056 | True |
| z_centre | crop | r2 | +0.793 | +0.538 | -0.255 | True | +0.440 | True |
| x_centre | crop | r2 | +0.530 | -0.001 | -0.531 | True | +0.021 | True |

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
| vicreg_B | phase_error | frame | r2 | 0.3861 | [0.224, 0.501] | 93 | 64 |
| vicreg_B | t0_group | frame | balanced_accuracy | 0.4648 | [0.401, 0.529] | 93 | 64 |
| vicreg_B | sos_mean_frame | frame | r2 | 0.8721 | [0.816, 0.911] | 93 | 64 |
| vicreg_B | sos_std_frame | frame | r2 | 0.4694 | [0.252, 0.605] | 93 | 64 |
| vicreg_B | cyst_fraction | frame | r2 | 0.5805 | [0.420, 0.673] | 93 | 64 |
| vicreg_B | z_centre | crop | r2 | 0.5378 | [0.513, 0.561] | 2976 | 64 |
| vicreg_B | x_centre | crop | r2 | -0.0011 | [-0.003, -0.000] | 2976 | 64 |
| random_resnet18 | phase_error | frame | r2 | 0.2039 | [0.019, 0.351] | 93 | 64 |
| random_resnet18 | t0_group | frame | balanced_accuracy | 0.3389 | [0.272, 0.407] | 93 | 64 |
| random_resnet18 | sos_mean_frame | frame | r2 | 0.7008 | [0.568, 0.790] | 93 | 64 |
| random_resnet18 | sos_std_frame | frame | r2 | 0.212 | [0.054, 0.315] | 93 | 64 |
| random_resnet18 | cyst_fraction | frame | r2 | -0.056 | [-0.155, 0.013] | 93 | 64 |
| random_resnet18 | z_centre | crop | r2 | 0.44 | [0.417, 0.463] | 2976 | 64 |
| random_resnet18 | x_centre | crop | r2 | 0.0214 | [0.007, 0.035] | 2976 | 64 |
| pca_raw | phase_error | frame | r2 | -0.0402 | [-0.104, -0.016] | 93 | 64 |
| pca_raw | t0_group | frame | balanced_accuracy | 0.3176 | [0.249, 0.389] | 93 | 64 |
| pca_raw | sos_mean_frame | frame | r2 | -0.0119 | [-0.075, -0.001] | 93 | 64 |
| pca_raw | sos_std_frame | frame | r2 | -0.0084 | [-0.069, 0.004] | 93 | 64 |
| pca_raw | cyst_fraction | frame | r2 | -0.0295 | [-0.092, -0.008] | 93 | 64 |
| pca_raw | z_centre | crop | r2 | -0.0007 | [-0.003, 0.000] | 2976 | 64 |
| pca_raw | x_centre | crop | r2 | -0.0032 | [-0.010, 0.001] | 2976 | 64 |
| trivial | phase_error | frame | r2 | 0.3727 | [0.165, 0.506] | 93 | 2 |
| trivial | t0_group | frame | balanced_accuracy | 0.3417 | [0.271, 0.411] | 93 | 2 |
| trivial | sos_mean_frame | frame | r2 | 0.7692 | [0.697, 0.820] | 93 | 2 |
| trivial | sos_std_frame | frame | r2 | 0.198 | [0.026, 0.300] | 93 | 2 |
| trivial | cyst_fraction | frame | r2 | 0.2241 | [-0.042, 0.380] | 93 | 2 |
| trivial | z_centre | crop | r2 | 0.4038 | [0.380, 0.426] | 2976 | 2 |
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
