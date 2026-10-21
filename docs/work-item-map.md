# Work-item map

The repo used to be organised by **work item** (`WI-N`), and every file carried
its WI number in its name. On 2026-08-25 it was reorganised around **pipeline
stage** and **architecture** instead. The WI numbering itself is not gone — it
is still the vocabulary of [MEMORY.md](MEMORY.md) Part II, which is unchanged
and remains the design record.

This file is the translation table. Use it when a docstring, a commit message,
or a MEMORY.md section names a WI number or an old path.

> `MEMORY.md` was **deliberately not rewritten**. It is the historical record:
> it says what was done, when, and under which name. Rewriting its paths would
> make it describe a repo that never existed at the time it was written.
> Everything it cites resolves through the tables below.

---

## Work item → where the code lives now

| WI | What it is | Code |
|----|------------|------|
| WI-0  | Verify the raw NVIDIA frame format against its documentation | `scripts/data/probe_nvidia_frames.py`, `tests/test_nvidia_frame.py` |
| WI-2  | Eval harness, runnable before any beamformer exists | `src/ssl_us/eval/matrix.py`, `tests/test_eval_integration.py` |
| WI-3  | Sector occupancy vs depth | `scripts/figures/occupancy.py`, `src/ssl_us/beamform/sectors.py` |
| WI-4  | NVIDIA reconstruction gate | `scripts/beamform/validate_nvidia_reconstruction.py`, `tests/test_nvidia_reconstruction_gate.py` |
| WI-6  | Resumable build pipeline | `src/ssl_us/beamform/build.py`, `tests/test_build.py` |
| WI-8  | Build + verify the K-sector store | `scripts/beamform/{build,verify,profile}_nvidia_store.py`, `tests/test_nvidia_store_build.py` |
| WI-10 | Train/validation splits and their audit | `scripts/data/build_splits.py`, `src/ssl_us/data/splits.py`, `tests/test_splits.py` |
| WI-11 | Augmentation policies A and B | `src/ssl_us/data/augment.py`, `scripts/figures/augmentations.py` |
| WI-12 | VICReg (ResNet-18), Policy A and B | `scripts/train/vicreg.py`, `scripts/cluster/train_vicreg_policies.sh` |
| WI-13 | NVIDIA results matrix, K-ablation, qualitative pass, geometry control | `scripts/eval/nvidia_*.py`, `scripts/eval/geometry_control.py` |
| WI-14 | Spatial MAE baseline | `scripts/train/mae.py`, `scripts/eval/mae_mask_sweep.py` |
| WI-15 | Cross-aperture arm (optional, not run) | `src/ssl_us/models/xaperture.py` |
| WI-16 | Stanford FSA transfer corpus + transfer table | `scripts/data/fetch_stanford_fsa.py`, `scripts/beamform/build_stanford_store.py`, `scripts/eval/stanford_*.py`, `scripts/eval/rankme.py` |
| WI-17 | I-JEPA arm | `scripts/train/ijepa.py`, `src/ssl_us/{models,train}/ijepa.py` |
| WI-18 | VICReg on a ViT-S backbone | `scripts/cluster/train_vicreg_vit.sh` (runs `scripts/train/vicreg.py --backbone vit_s`) |
| WI-19 | Per-pixel segmentation probe + mask figure | `scripts/eval/nvidia_segmentation.py`, `src/ssl_us/eval/dense_{features,probe}.py`, `docs/cluster-segmentation-probe.md` |

---

## Scripts: old path → new path

| Old | New |
|-----|-----|
| `scripts/wi0_probe.py` | `scripts/data/probe_nvidia_frames.py` |
| `scripts/wi10_splits.py` | `scripts/data/build_splits.py` |
| `scripts/wi13_fetch_label_maps.py` | `scripts/data/fetch_nvidia_label_maps.py` |
| `scripts/wi16_fetch_stanford_fsa.py` | `scripts/data/fetch_stanford_fsa.py` |
| `scripts/wi8_build.py` | `scripts/beamform/build_nvidia_store.py` |
| `scripts/wi8_verify.py` | `scripts/beamform/verify_nvidia_store.py` |
| `scripts/wi8_profile.py` | `scripts/beamform/profile_nvidia_store.py` |
| `scripts/wi16_beamform.py` | `scripts/beamform/build_stanford_store.py` |
| `scripts/wi4_reference_validation.py` | `scripts/beamform/validate_nvidia_reconstruction.py` |
| `scripts/wi16_reference_validation.py` | `scripts/beamform/validate_stanford_reconstruction.py` |
| `scripts/wi12_train.py` | `scripts/train/vicreg.py` |
| `scripts/wi14_train.py` | `scripts/train/mae.py` |
| `scripts/wi17_train.py` | `scripts/train/ijepa.py` |
| `scripts/wi13_matrix.py` | `scripts/eval/nvidia_matrix.py` |
| `scripts/wi16_matrix.py` | `scripts/eval/stanford_matrix.py` |
| `scripts/wi13_qualitative.py` | `scripts/eval/nvidia_qualitative.py` |
| `scripts/wi16_qualitative.py` | `scripts/eval/stanford_qualitative.py` |
| `scripts/wi16_rankme.py` | `scripts/eval/rankme.py` |
| `scripts/wi13_geometry_control.py` | `scripts/eval/geometry_control.py` |
| `scripts/wi14_mask_sweep.py` | `scripts/eval/mae_mask_sweep.py` |
| `scripts/wi13_anatomy_verify.py` | `scripts/eval/verify_anatomy_reproduction.py` |
| `scripts/wi3_occupancy_figure.py` | `scripts/figures/occupancy.py` |
| `scripts/wi11_augmentation_figure.py` | `scripts/figures/augmentations.py` |
| `scripts/wi12_run_both.sh` | `scripts/cluster/train_vicreg_policies.sh` |
| `scripts/wi13_kablation.sh` | `scripts/cluster/train_vicreg_k_ablation.sh` |
| `scripts/wi14_run.sh` | `scripts/cluster/train_mae.sh` |
| `scripts/wi17_run.sh` | `scripts/cluster/train_ijepa.sh` |
| `scripts/wi18_run.sh` | `scripts/cluster/train_vicreg_vit.sh` |
| `scripts/wi17_eval_chain.sh`, `scripts/wi18_eval_chain.sh` | `scripts/cluster/eval_matrices.sh` (they were identical; consolidated) |
| `scripts/wi18_diag_chain.sh` | `scripts/cluster/eval_diagnostics.sh` |
| `scripts/wi13_anatomy_columns.sh` | `scripts/cluster/eval_anatomy_columns.sh` |
| `scripts/wi13_qualitative_columns.sh` | `scripts/cluster/eval_qualitative_columns.sh` |
| `scripts/wi17_geom_chain.sh` | `scripts/cluster/eval_geometry_control.sh` |
| `scripts/wi12_ballast.py` | `scripts/cluster/ballast.py` |
| `scripts/wi17_ballast_watch.sh` | `scripts/cluster/ballast_watch.sh` |
| `scripts/wi12_status.sh` | `scripts/cluster/status.sh` |

## Tests: old name → new name

| Old | New |
|-----|-----|
| `test_nv_frame.py` | `test_nvidia_frame.py` |
| `test_wi4_reference_validation.py` | `test_nvidia_reconstruction_gate.py` |
| `test_wi16_reference_validation.py` | `test_stanford_reconstruction_gate.py` |
| `test_wi8_build.py` | `test_nvidia_store_build.py` |
| `test_wi10_splits.py` | `test_splits.py` |
| `test_wi12_eval_integration.py` | `test_eval_integration.py` |
| `test_wi13_matrix.py` | `test_nvidia_matrix.py` |
| `test_wi16_matrix.py` | `test_stanford_matrix.py` |
| `test_wi16_results.py` | `test_stanford_results.py` |
| `test_wi16_stanford_fsa.py` | `test_stanford_fsa_corpus.py` |
| `test_wi16_beamform.py` | `test_stanford_beamform.py` |

## Committed results: `tests/data/` → `results/`

`tests/data/` is gone. The numbers it held were never really test fixtures —
they are the project's measured results, and they now live in `results/`, which
is tracked in git. The tests still read them (that is what makes a silent
regression visible), they are simply no longer *named* as test data.

| Old | New |
|-----|-----|
| `tests/data/wi0_results.json` | `results/gates/nvidia_frame_probe.json` |
| `tests/data/wi4_results.json` | `results/gates/nvidia_reconstruction.json` |
| `tests/data/wi4_reference_selfconsistency.json` | `results/gates/nvidia_reference_selfconsistency.json` |
| `tests/data/wi16_results.json` | `results/gates/stanford_reconstruction.json` |
| `tests/data/wi8_depth_profile.json` | `results/gates/nvidia_store_depth_profile.json` |
| `tests/data/wi8_store_verification.json` | `results/gates/nvidia_store_verification.json` |
| `tests/data/wi10_splits.json` | `results/splits/nvidia_splits.json` |
| `tests/data/wi13_k{K}_{split}_results.json` | `results/nvidia/k{K}_{split}.json` |
| `tests/data/wi13_k4_{split}_qual_results.json` | `results/nvidia/k4_{split}_qualitative.json` |
| `tests/data/wi13_k4_{split}_clusters.md` | `results/nvidia/k4_{split}_clusters.md` |
| `tests/data/wi13_geom_k4_{split}_results.json` | `results/nvidia/geometry_k4_{split}.json` |
| `tests/data/wi14_mask_sweep.json` | `results/mae/mask_sweep.json` |
| `tests/data/wi14_mae_k4{_rawtarget}_training.json` | `results/mae/k4{_rawtarget}_training.json` |
| `tests/data/wi16_{variant}_k{K}{suffix}_results.json` | `results/stanford/{variant}_k{K}{suffix}.json` |

## Generated outputs: `figures/` → `outputs/`

`outputs/` is gitignored (as `figures/` was). Existing runs were **moved, not
regenerated** — the numbers are the same ones, at new paths.

| Old | New |
|-----|-----|
| `figures/wi13_k{K}_{split}[_anatomy]` | `outputs/nvidia/k{K}_{split}[_anatomy]` |
| `figures/wi13_k4_{split}_qual` | `outputs/nvidia/k4_{split}_qualitative` |
| `figures/wi13_geom_k4_{split}` | `outputs/nvidia/geometry_k4_{split}` |
| `figures/wi16_{variant}_k{K}{suffix}` | `outputs/stanford/{variant}_k{K}{suffix}` |
| `figures/wi14_sweep[_b64]` | `outputs/mae/mask_sweep[_b64]` |
| `figures/wi4/` | `outputs/gates/nvidia_reconstruction/` |
| `figures/wi16/` | `outputs/gates/stanford_reconstruction/` |
| `figures/wi8/` | `outputs/gates/nvidia_store/` |
| `figures/wi10/` | `outputs/splits/` |
| `figures/wi11_augmentations.png` | `outputs/diagnostics/augmentations.png` |

---

## What deliberately kept its WI name

These are **paths to real data on the cluster filesystem**, outside the repo.
Renaming the strings without moving multi-gigabyte stores and trained weights
would break every run, so they were left alone:

| Name | What it is |
|------|------------|
| `stores/wi8_fsa_k16.zarr` | the 23 GB NVIDIA K-sector store |
| `stores/wi8_label_maps/` | fetched per-pixel segmentation + sound-speed maps |
| `stores/wi16_stanford_fsa.zarr` | the Stanford FSA source corpus |
| `stores/wi16_{mm,lambda}_matched_k16.zarr` | the beamformed Stanford stores |
| `checkpoints/wi12_vicreg_policy{A,B}` | VICReg ResNet-18, K=4 |
| `checkpoints/wi13_vicreg_k{K}_policy{A,B}` | VICReg ResNet-18, K-ablation cells |
| `checkpoints/wi14_mae_k{K}[_rawtarget]` | MAE arms |
| `checkpoints/wi17_ijepa_k{K}` | I-JEPA arm |
| `checkpoints/wi18_vicreg_vit_k{K}_policyA` | VICReg ViT-S arm |

`src/ssl_us/eval/encoders.py` matches these names via `CKPT_TEMPLATES` and the
per-arm paths beside it. If the on-disk directories are ever renamed, that
module is the single place to update.
