# WI-13 results matrix (K=4)

## Policy A vs Policy B

| target | level | metric | policy_a | policy_b | delta_b_minus_a | separated | random_baseline | beats_random | mae | best_vicreg_beats_mae |
|---|---|---|---|---|---|---|---|---|---|---|
| phase_error | frame | r2 | +0.378 | +0.399 | +0.021 | False | +0.314 | True | None | None |
| t0_group | frame | balanced_accuracy | +0.407 | +0.495 | +0.089 | True | +0.368 | True | None | None |
| sos_mean_frame | frame | r2 | +0.918 | +0.876 | -0.042 | True | +0.780 | True | None | None |
| sos_std_frame | frame | r2 | +0.294 | +0.322 | +0.028 | False | +0.209 | True | None | None |
| cyst_fraction | frame | r2 | +0.365 | +0.223 | -0.142 | True | +0.091 | True | None | None |
| z_centre | crop | r2 | +0.768 | +0.561 | -0.207 | True | +0.491 | True | None | None |
| x_centre | crop | r2 | +0.528 | -0.000 | -0.528 | True | +0.035 | True | None | None |
| contains_cyst | crop | balanced_accuracy | +0.948 | +0.771 | -0.177 | True | +0.783 | True | None | None |
| cyst_fraction_crop | crop | r2 | +0.772 | +0.703 | -0.070 | True | +0.396 | True | None | None |
| sos_mean_crop | crop | r2 | +0.731 | +0.582 | -0.150 | True | +0.473 | True | None | None |

## All cells

| encoder | target | level | metric | score | ci95 | n_eff | n_feat |
|---|---|---|---|---|---|---|---|
| vicreg_A | phase_error | frame | r2 | 0.3782 | [0.324, 0.426] | 830 | 64 |
| vicreg_A | t0_group | frame | balanced_accuracy | 0.4067 | [0.383, 0.432] | 830 | 64 |
| vicreg_A | sos_mean_frame | frame | r2 | 0.9182 | [0.907, 0.928] | 830 | 64 |
| vicreg_A | sos_std_frame | frame | r2 | 0.2943 | [0.240, 0.340] | 830 | 64 |
| vicreg_A | cyst_fraction | frame | r2 | 0.3655 | [0.305, 0.417] | 830 | 64 |
| vicreg_A | z_centre | crop | r2 | 0.7682 | [0.758, 0.778] | 6640 | 64 |
| vicreg_A | x_centre | crop | r2 | 0.5282 | [0.513, 0.542] | 6640 | 64 |
| vicreg_A | contains_cyst | crop | balanced_accuracy | 0.9479 | [0.942, 0.954] | 6136 | 64 |
| vicreg_A | cyst_fraction_crop | crop | r2 | 0.7722 | [0.748, 0.794] | 6136 | 64 |
| vicreg_A | sos_mean_crop | crop | r2 | 0.7314 | [0.719, 0.744] | 6640 | 64 |
| vicreg_B | phase_error | frame | r2 | 0.3989 | [0.345, 0.447] | 830 | 64 |
| vicreg_B | t0_group | frame | balanced_accuracy | 0.4954 | [0.468, 0.524] | 830 | 64 |
| vicreg_B | sos_mean_frame | frame | r2 | 0.876 | [0.859, 0.890] | 830 | 64 |
| vicreg_B | sos_std_frame | frame | r2 | 0.3223 | [0.267, 0.371] | 830 | 64 |
| vicreg_B | cyst_fraction | frame | r2 | 0.2232 | [0.173, 0.268] | 830 | 64 |
| vicreg_B | z_centre | crop | r2 | 0.5613 | [0.545, 0.576] | 6640 | 64 |
| vicreg_B | x_centre | crop | r2 | -0.0002 | [-0.001, 0.000] | 6640 | 64 |
| vicreg_B | contains_cyst | crop | balanced_accuracy | 0.771 | [0.760, 0.782] | 6136 | 64 |
| vicreg_B | cyst_fraction_crop | crop | r2 | 0.7027 | [0.676, 0.728] | 6136 | 64 |
| vicreg_B | sos_mean_crop | crop | r2 | 0.5817 | [0.565, 0.598] | 6640 | 64 |
| random_resnet18 | phase_error | frame | r2 | 0.3144 | [0.260, 0.364] | 830 | 64 |
| random_resnet18 | t0_group | frame | balanced_accuracy | 0.3685 | [0.337, 0.400] | 830 | 64 |
| random_resnet18 | sos_mean_frame | frame | r2 | 0.78 | [0.756, 0.801] | 830 | 64 |
| random_resnet18 | sos_std_frame | frame | r2 | 0.2092 | [0.165, 0.249] | 830 | 64 |
| random_resnet18 | cyst_fraction | frame | r2 | 0.0908 | [0.040, 0.138] | 830 | 64 |
| random_resnet18 | z_centre | crop | r2 | 0.4912 | [0.476, 0.506] | 6640 | 64 |
| random_resnet18 | x_centre | crop | r2 | 0.0346 | [0.025, 0.044] | 6640 | 64 |
| random_resnet18 | contains_cyst | crop | balanced_accuracy | 0.7835 | [0.773, 0.794] | 6136 | 64 |
| random_resnet18 | cyst_fraction_crop | crop | r2 | 0.3958 | [0.365, 0.424] | 6136 | 64 |
| random_resnet18 | sos_mean_crop | crop | r2 | 0.4727 | [0.457, 0.488] | 6640 | 64 |
| pca_raw | phase_error | frame | r2 | -0.0162 | [-0.036, -0.002] | 830 | 64 |
| pca_raw | t0_group | frame | balanced_accuracy | 0.3132 | [0.292, 0.333] | 830 | 64 |
| pca_raw | sos_mean_frame | frame | r2 | -0.0014 | [-0.007, -0.000] | 830 | 64 |
| pca_raw | sos_std_frame | frame | r2 | -0.0043 | [-0.015, 0.005] | 830 | 64 |
| pca_raw | cyst_fraction | frame | r2 | -0.0039 | [-0.010, -0.001] | 830 | 64 |
| pca_raw | z_centre | crop | r2 | -0.0004 | [-0.001, -0.000] | 6640 | 64 |
| pca_raw | x_centre | crop | r2 | -0.0 | [-0.001, 0.000] | 6640 | 64 |
| pca_raw | contains_cyst | crop | balanced_accuracy | 0.5073 | [0.504, 0.511] | 6136 | 64 |
| pca_raw | cyst_fraction_crop | crop | r2 | -0.0008 | [-0.002, -0.000] | 6136 | 64 |
| pca_raw | sos_mean_crop | crop | r2 | -0.0014 | [-0.002, -0.001] | 6640 | 64 |
| trivial | phase_error | frame | r2 | 0.3612 | [0.301, 0.413] | 830 | 2 |
| trivial | t0_group | frame | balanced_accuracy | 0.3761 | [0.352, 0.399] | 830 | 2 |
| trivial | sos_mean_frame | frame | r2 | 0.7606 | [0.739, 0.780] | 830 | 2 |
| trivial | sos_std_frame | frame | r2 | 0.2429 | [0.199, 0.282] | 830 | 2 |
| trivial | cyst_fraction | frame | r2 | 0.1398 | [0.089, 0.186] | 830 | 2 |
| trivial | z_centre | crop | r2 | 0.4435 | [0.429, 0.458] | 6640 | 2 |
| trivial | x_centre | crop | r2 | -0.0001 | [-0.001, -0.000] | 6640 | 2 |
| trivial | contains_cyst | crop | balanced_accuracy | 0.7787 | [0.768, 0.789] | 6136 | 2 |
| trivial | cyst_fraction_crop | crop | r2 | 0.3235 | [0.289, 0.356] | 6136 | 2 |
| trivial | sos_mean_crop | crop | r2 | 0.4356 | [0.421, 0.449] | 6640 | 2 |

## Cells not run

| cell | reason |
|---|---|
