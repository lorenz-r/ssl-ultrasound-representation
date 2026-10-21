# WI-13 results matrix (K=4)

## Policy A vs Policy B

| target | level | metric | policy_a | policy_b | delta_b_minus_a | separated | random_baseline | beats_random |
|---|---|---|---|---|---|---|---|---|
| phase_error | frame | r2 | +0.370 | +0.305 | -0.065 | False | +0.204 | True |
| t0_group | frame | balanced_accuracy | +0.406 | +0.526 | +0.120 | False | +0.442 | True |
| sos_mean_frame | frame | r2 | +0.948 | +0.865 | -0.084 | True | +0.783 | True |
| sos_std_frame | frame | r2 | +0.503 | +0.440 | -0.063 | False | +0.256 | True |
| cyst_fraction | frame | r2 | +0.355 | +0.397 | +0.042 | False | +0.172 | True |
| z_centre | crop | r2 | +0.441 | +0.687 | +0.246 | True | +0.614 | True |
| x_centre | crop | r2 | +0.787 | -0.001 | -0.788 | True | +0.121 | True |

## All cells

| encoder | target | level | metric | score | ci95 | n_eff | n_feat |
|---|---|---|---|---|---|---|---|
| vicreg_A | phase_error | frame | r2 | 0.3696 | [0.213, 0.485] | 93 | 64 |
| vicreg_A | t0_group | frame | balanced_accuracy | 0.4056 | [0.335, 0.473] | 93 | 64 |
| vicreg_A | sos_mean_frame | frame | r2 | 0.9482 | [0.930, 0.961] | 93 | 64 |
| vicreg_A | sos_std_frame | frame | r2 | 0.5026 | [0.309, 0.616] | 93 | 64 |
| vicreg_A | cyst_fraction | frame | r2 | 0.3551 | [0.227, 0.442] | 93 | 64 |
| vicreg_A | z_centre | crop | r2 | 0.4409 | [0.417, 0.465] | 2976 | 64 |
| vicreg_A | x_centre | crop | r2 | 0.7871 | [0.770, 0.803] | 2976 | 64 |
| vicreg_B | phase_error | frame | r2 | 0.3048 | [0.131, 0.431] | 93 | 64 |
| vicreg_B | t0_group | frame | balanced_accuracy | 0.5259 | [0.465, 0.579] | 93 | 64 |
| vicreg_B | sos_mean_frame | frame | r2 | 0.8647 | [0.809, 0.906] | 93 | 64 |
| vicreg_B | sos_std_frame | frame | r2 | 0.4396 | [0.273, 0.559] | 93 | 64 |
| vicreg_B | cyst_fraction | frame | r2 | 0.397 | [0.226, 0.512] | 93 | 64 |
| vicreg_B | z_centre | crop | r2 | 0.687 | [0.667, 0.705] | 2976 | 64 |
| vicreg_B | x_centre | crop | r2 | -0.001 | [-0.003, -0.000] | 2976 | 64 |
| random_resnet18 | phase_error | frame | r2 | 0.2045 | [0.042, 0.332] | 93 | 64 |
| random_resnet18 | t0_group | frame | balanced_accuracy | 0.4417 | [0.374, 0.507] | 93 | 64 |
| random_resnet18 | sos_mean_frame | frame | r2 | 0.7831 | [0.709, 0.842] | 93 | 64 |
| random_resnet18 | sos_std_frame | frame | r2 | 0.2556 | [0.069, 0.410] | 93 | 64 |
| random_resnet18 | cyst_fraction | frame | r2 | 0.1724 | [0.045, 0.277] | 93 | 64 |
| random_resnet18 | z_centre | crop | r2 | 0.6143 | [0.587, 0.638] | 2976 | 64 |
| random_resnet18 | x_centre | crop | r2 | 0.1206 | [0.099, 0.140] | 2976 | 64 |
| pca_raw | phase_error | frame | r2 | -0.0419 | [-0.109, -0.013] | 93 | 64 |
| pca_raw | t0_group | frame | balanced_accuracy | 0.4352 | [0.339, 0.552] | 93 | 64 |
| pca_raw | sos_mean_frame | frame | r2 | -0.016 | [-0.079, -0.001] | 93 | 64 |
| pca_raw | sos_std_frame | frame | r2 | 0.0014 | [-0.061, 0.021] | 93 | 64 |
| pca_raw | cyst_fraction | frame | r2 | -0.0432 | [-0.112, -0.022] | 93 | 64 |
| pca_raw | z_centre | crop | r2 | -0.0013 | [-0.003, -0.001] | 2976 | 64 |
| pca_raw | x_centre | crop | r2 | -0.001 | [-0.003, -0.000] | 2976 | 64 |
| trivial | phase_error | frame | r2 | 0.0729 | [-0.159, 0.212] | 93 | 2 |
| trivial | t0_group | frame | balanced_accuracy | 0.5028 | [0.439, 0.563] | 93 | 2 |
| trivial | sos_mean_frame | frame | r2 | 0.0226 | [-0.073, 0.091] | 93 | 2 |
| trivial | sos_std_frame | frame | r2 | -0.0163 | [-0.291, 0.114] | 93 | 2 |
| trivial | cyst_fraction | frame | r2 | 0.2721 | [-0.019, 0.428] | 93 | 2 |
| trivial | z_centre | crop | r2 | 0.6114 | [0.580, 0.639] | 2976 | 2 |
| trivial | x_centre | crop | r2 | -0.0009 | [-0.003, 0.000] | 2976 | 2 |

## Cells not run

| cell | reason |
|---|---|
| contains_cyst | needs per-pixel segmentation map (source NVIDIA frames absent) |
| cyst_fraction_crop | crop-level needs the segmentation map; frame-level cyst_fraction is run |
| sos_mean_crop | crop-level needs the per-pixel SoS map; frame-level sos_mean_frame is run |
| K=16 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
| K=8 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
| K=1 column | K sets the stem's input channel count, so this cell needs its own VICReg run (~1.5 h/arm); only K=4 was trained in WI-12 |
