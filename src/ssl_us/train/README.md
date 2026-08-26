# Training instrumentation — failure signatures

Verbatim from PLAN.md §4.7. **VICReg loss value alone does not indicate
success.** It can fall while the representation collapses into a
low-dimensional subspace. All instrumentation below ships with the harness
in WI-12, before any run is launched — not added after a run looks wrong.

## Tier 1 — every epoch (cheap, no labels or crop-level only)

| Instrument | Detail | Cost |
|---|---|---|
| Three VICReg terms logged separately | invariance, variance, covariance — **never log only the sum**. Plus the variance term's hinge margin. | free |
| Per-dimension std of projector output | 2048-d, logged as a histogram | free |
| Singular-value spectrum of the BACKBONE feature covariance | 512-d, **pre-projector, not the projector** | one 512x512 SVD/epoch |
| RankMe | `exp(-sum(p_k log p_k))`, `p_k = sigma_k / sum(sigma) + eps`, over that **backbone** spectrum. **Primary collapse diagnostic; needs no labels.** | free given the SVD |
| Online linear probe — cyst/background | crop-level, balanced accuracy, on **detached** features | small |
| kNN evaluation | held-out subset, no training required | small |
| Random-init floor | identical architecture, plotted as a visible line **on the same axes** as every chart above | one-off |

## Tier 2 — every N epochs (needs crop-to-frame aggregation)

| Instrument | Detail |
|---|---|
| Frame-level sound-speed regression | ridge R^2, crops mean-pooled to frames (§6.12 — SoS is a frame-level target) |
| Frame-level phase-error regression | ridge R^2, same aggregation, plotted against the **0.585 oracle line** (§4.1) |

Backbone (512-d) and projector (2048-d) are kept deliberately distinct:
**collapse appears in the backbone spectrum well before the projector's
variance hinge reacts**, so RankMe is computed on the backbone and the
per-dimension histogram on the projector. Reporting RankMe on the projector
would be the standard mistake here.

**Stop-gradient on the online probe is a correctness requirement, not a
detail.** The probe head trains on `features.detach()`; a test asserts that
a probe-loss backward pass leaves every encoder parameter's `.grad`
unchanged. Without it the "linear probe" silently becomes supervised
fine-tuning and every frozen-feature number in the results matrix is
invalid. See `instrument.assert_no_grad_leak` and
`tests/test_instrument.py`.

## Failure signatures

| Symptom | Diagnosis | Action |
|---|---|---|
| loss down **and** RankMe down | **dimensional collapse** | raise covariance weight; check projector width; check augmentation strength |
| loss down, probe flat, RankMe healthy | **augmentation shortcut** — the model solved the pretext task without learning content | inspect which augmentation makes views trivially matchable |
| probe **degrading** over training | augmentations are destroying the probed property | **see below — this is not automatically a failure** |

**The third row is the experiment, not a bug.** For **Policy A**,
degradation of the *physics* probes (`z_centre`, `x_centre`, `t0_group`)
over training is the **expected and desired** outcome — Policy A randomizes
acquisition factors precisely to make the encoder invariant to them, and
whatever the encoder is invariant to is deleted from the latent and can no
longer be probed. The instrumentation therefore tracks **probe trajectories
per label axis**, never a single aggregate, and the harness must not flag
per-axis degradation as a failure. The Policy A down-geometry/up-tissue
versus Policy B up-geometry/down-tissue contrast (PLAN.md §4.1) is the
deliverable; seeing it *develop over training* is the strongest form of that
evidence.
