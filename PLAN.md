# PLAN — beamformed-domain SSL pretraining on NV-Raw2Insights-US

Branch: `feature/beamformed-vicreg`

Status of this document: written **before any code**, from (a) a full read of the
current repo and (b) a live inspection of the NVIDIA dataset's metadata columns —
all 923 samples were actually read over HTTP range requests, so every number in
§2 below is measured, not quoted from the card. The one thing **not** yet read is
the `iq_real` / `iq_imag` payload itself (265 MB/frame); that is Work Item 0 and
nothing downstream is committed until it returns real numbers.

Sections marked **⚠** are places where the dataset or the repo makes part of the
brief impossible, unwise, or measurably different from assumption. Read §6 first if
you read nothing else.

**Revision 4** follows the reciprocity observation to its consequences and closes a
label leak:

- **§4.5a reciprocity has three consequences.** Transmit-side cross-aperture masking is shortcut-solvable by transposing receive data — mitigated **unconditionally** by never exposing both aperture axes in one tensor. Transmit and receive diversity are near-redundant, which combined with ±10° vs ±26.6° span **re-ranks the PW stack down**: build it only if the measured residual is materially non-zero. And **reciprocity will not hold on Stanford** (separate electronics, element variation, `Refocus` residuals) — a new named transfer failure mode that sim-only evaluation cannot see.
- **§4.4b DBUA is a label leak, enforced in code.** `calculate_delays_heterogeneous_medium` consumes the sound-speed map, so `gt_sos`-beamformed data cannot probe sound speed or phase error. Two strictly separated stages with `assert_no_label_leak` in the guard path. `bmode_variant_2` is tagged `gt_sos` conservatively until WI-0 resolves it. The **difference** between the two stages is close to a pure aberration signal — noted as a bonus target.
- **§3.2d crop size matches within an arm.** Rev-3 said train at 128, transfer at 64 for K ≥ 8; that reintroduces the statistics mismatch it was trying to avoid. Now two internally-consistent configurations: `K ≤ 4 @ 128` and `K ≥ 8 @ 64`.
- **§4.4 external ultrasound baselines are a ceiling, not the answer.** HVAE is cardiac, Speckle2Self in-vivo; a weak result may be anatomy mismatch. The clean control for B-mode vs sub-apertures is our own VICReg across signal stages.

**§10 reports the WI-0 measurements**, including the axis order and the reciprocity
residual.

**Revision 3** adds the **zea toolbox investigation (§0 — read it first, it
restructures WI-3/WI-4/WI-16)** and the depth decisions:

- **§0 zea wins on nearly everything.** `tof_correction` returns the **pre-summation** `(n_tx, n_pix, n_el, n_ch)` tensor, so angular sectors are a ~20-line segment-sum replacing their 6-line `DelayAndSum`. Baseband phase rotation, delays, interpolation, f-number masking, pixel chunking and grid construction all come from zea. **OpenH-RF already ships a working converter and reference reconstruction for our exact dataset.** WI-3 and WI-4 shrink substantially.
- **§0.5 two expectations were wrong:** zea's augmentation library is a *single class*, and its TAESD is the **stock Stable Diffusion XL autoencoder — a natural-image model, not ultrasound**. HVAE and Speckle2Self are the real ultrasound-domain baselines.
- **§0.2 two silent-error traps found:** the IQ axis order is ambiguous between NVIDIA's card and OpenH-RF's converter and is invisible on square, reciprocal data; and `bmode_focused` is documented differently by the two sources.
- **§3.2c the 8 mm depth floor was MINE, not NVIDIA's** — the simulation spans 0–40 mm. Stanford is entirely in-distribution, and the transfer depth problem dissolves at K ≤ 4.
- **§3.2d crop size does not fix the stored grid** — pixel spacing does. At the real 0.079 mm/px the transfer band holds 146 crop positions, not one.
- **§4.5c depth-agnosticism**, four measures in priority order, with zea's `make_tgc_curve` supplying the highest-value one.

**Revision 2** applied six earlier changes. Three were corrections to me, and one
would have silently broken transfer:

- **§3.2 sub-apertures are now ANGULAR SECTORS, not element groups.** This is the important one. Element-indexed sub-apertures are not comparable across probes, so a model consuming them as tensor channels binds its weights to a quantity that changes meaning between datasets — it would have run, produced numbers, and the numbers would have been meaningless on transfer. The 176-element / 11-per-group decision is withdrawn; all 180 elements are used.
- **§3.2b transmit-angle diversity is reinstated.** My "no angles to subset" claim was true as-stored and wrong in principle. Measured: synthesizing plane waves from the Tx–Rx matrix costs **+2.8 % compute**, and the storage is costed in a table. Decision: FSA × K=16 + PW(5) × K=4, **61.5 GB**.
- **§3.6 Stanford is reinstated as a first-class transfer corpus**, which resolves §6.8 and §6.9 rather than deferring them.
- **§4.7 full collapse instrumentation** ships before the first run.
- **§4.5b cross-aperture masking axis** is now an argued choice (transmit), not an assumption.
- **§4.1 evaluation** is regression-first, and two checks the review asked for were **run during planning rather than deferred** — with consequences: the crop-level sound-speed probe turns out to be a frame-level probe in disguise (§6.12), and `phase_error` carries a measured oracle ceiling of R² = 0.585.

---

## 0. zea investigation — findings first, because they restructure the plan

Investigated `zea` 0.1.4 (sdist read directly, not just docs), the `tue-bmd/zea`
repo tree, and `open-h/OpenH-RF`. Headline: **zea wins on almost everything it was
asked about, and OpenH-RF has already converted our exact dataset.** Two of the
user's expectations about zea turn out to be wrong, both in ways that matter.

### 0.1 CRITICAL — yes, the pre-summation tensor is exposed

`zea.beamform.beamformer.tof_correction(...)` returns
**`(n_tx, n_pix, n_el, n_ch)`** — delayed, f-number-masked, and IQ **phase-rotated**
per-element data, *before any summation*. The K=16 design is safe.

The pipeline is `Beamform(PatchedGrid(TOFCorrection → DelayAndSum) → ReshapeGrid)`,
and `DelayAndSum` (`zea/ops/pipeline.py:1411`) is six lines:

```python
beamformed_data = ops.sum(data, -2)   # over n_el  (DAS)
beamformed_data = ops.sum(beamformed_data, -3)   # over n_tx  (compounding)
```

**Our angular-sector beamformer replaces exactly that op** — a segment-sum over the
element axis into 16 bins, then the transmit sum. ~20 lines, as a registered
`zea.ops.Operation`, dropped into their pipeline in place of `DelayAndSum`.

What we get for free, all of which revision 2 planned to write:

| Needed | zea provides |
|---|---|
| Delay computation (tx + rx, focused/PW/FSA) | `calculate_delays`, `transmit_delays`, `compute_receive_distances` |
| Interpolation at fractional delays | `apply_delays` |
| **Baseband phase rotation** `exp(j2πf_d τ)` | `complex_rotate` — **trap #1 of §3.5 is solved by their code** |
| Receive f-number mask + window functions | `fnumber_mask`, `fnum_window_fn_{rect,hann,tukey}` |
| **Pixel chunking** | `PatchedGrid(num_patches=…)` |
| Grid construction | `zea.beamform.pixelgrid.cartesian_pixel_grid` |
| Heterogeneous sound-speed beamforming | `calculate_delays_heterogeneous_medium` |
| Lens correction, pfield weighting, apodization stages | present, unused by us |

**Pixel chunking is not optional for us:** the full pre-sum tensor for one frame is
180 × 231 192 × 180 × 2 × 4 B = **60 TB**. `PatchedGrid` is the reason this is
tractable at all, and it already exists.

⇒ **WI-3 and WI-4 shrink to: our sector-id map, our segment-sum op, and the
validation gate.** Delay maths, interpolation, phase rotation, f-number masking,
chunking and grid construction are all deleted from our scope.

### 0.2 CRITICAL — OpenH-RF has already converted NV-Raw2Insights-US

`open-h/OpenH-RF` (Apache-2.0) ships `examples/nv-raw2insights-us/` containing a
working `convert.py`, `reconstruct.py`, `pipeline.yaml` and a data card **for our
exact dataset**. This is most of WI-0 and a reference reconstruction, already written
and openly licensed.

The authoritative FSA scan-parameter mapping, which was the genuinely hard part of
WI-3, is given outright:

```python
scan = {
    "t0_delays":       np.zeros((n_tx, n_el)),   # FSA: no transmit delay profile
    "tx_apodizations": np.eye(n_tx, n_el),       # FSA: one element per transmit  ← the key line
    "focus_distances": np.zeros(n_tx),
    "transmit_origins": probe_geometry,          # each transmit originates at its element
    "polar_angles":    np.zeros(n_tx),
    "initial_times":   np.full(n_tx, t0),
    "sampling_frequency": fs, "center_frequency": fc,
    "demodulation_frequency": fd, "sound_speed": c0,
}
```

`reconstruct.py` additionally beamforms with the **ground-truth SoS map** via zea's
heterogeneous mode — giving us a *SoS-corrected* signal stage for free.

**⚠ But it does NOT contain Stanford.** OpenH-RF has exactly three real
sub-datasets: `nv-raw2insights-us`, `pala-ulm-ratbrain`, `tracked-cirs-phantom`.
There is no Stanford RF set and no unified two-corpus release. **WI-16's cross-probe
work is not removed.** What *is* removed is the file-reading half of it, by §0.4.

**⚠ And it contradicts NVIDIA's own card on two points.**

1. **IQ axis order is genuinely ambiguous.** NVIDIA's card says `iq_real` is `[n_tx, n_el, n_ax]`. OpenH-RF's converter unpacks `n_tx, n_el, _ = iq_real.shape` and then does `np.transpose(raw_iq, (1, 2, 0, 3))`, which yields `(n_el, n_ax, n_tx, 2)` — while its own inline comment claims `(n_tx, n_ax, n_el, 2)`. **These disagree, and because `n_tx == n_el == 180` the array shape is identical either way, and because the Tx–Rx matrix is near-symmetric by acoustic reciprocity the reconstructed image looks correct either way.** This is exactly the class of silent error WI-0 exists to catch. **The diagonal-onset test in WI-0 is now more important, not less** — and it must be run before any transmit-axis operation (PW synthesis, transmit masking), which are the two places a tx/rx swap would actually bite.
2. **`bmode_focused` means something different than the NVIDIA card says.** NVIDIA calls it "Focused transmit b-mode". OpenH-RF's data card calls it "B-mode (DBUA aberration-corrected)" and labels it that way in the converter. Those are not the same thing. As a *signal stage* in our results matrix this matters: aberration-corrected is not a transmit-scheme variant, it is a different reconstruction of the same transmits. Resolve in WI-0; until then, label it `bmode_variant_2` and do not claim which it is.

### 0.3 Probe / Parameters: adopt as the shared coordinate frame

`zea.Probe` (element geometry, `n_el` and `pitch` derived from `probe_geometry`,
linear and curved constructors, `from_name`) and `zea.Parameters` (`sound_speed`,
`sampling_frequency`, `center_frequency`, `probe_geometry`, `f_number`, `grid`,
`flatgrid`, `xlims`/`zlims`, `wavelength`, `extent_imshow`, `attenuation_coef`,
`tgc_gain_curve`, `selected_transmits`) is exactly the probe-agnostic abstraction
that revision 2 was going to hand-roll as `geometry.py`. **Adopt it.** Our
`geometry.py` shrinks to the sector-id and occupancy maps.

### 0.4 REFoCUS points the other way — and that is better for us

`zea.ops.Refocus` decodes focused/plane-wave transmits **into** multistatic
(full-matrix) data, `(n_tx, n_ax, n_el, n_ch) → (n_el, n_ax, n_el, n_ch)`, with
adjoint / Tikhonov / rSVD / tSVD inversion.

It does **not** give us the PW stack — NVIDIA is already multistatic, and the
forward direction (multistatic → plane wave) is a delay-and-sum on the transmit
axis, ~15 lines, still ours.

**But it solves a harder problem than the one asked about.** Stanford is *focused*
multi-focal transmits. REFoCUS decodes it into multistatic — putting **both corpora
into the same transmit representation**, so the identical PW synthesis, identical
sector binning, and identical pipeline run on both. That is a substantially stronger
transfer story than grid-matching alone, and it was not in revision 2. Added to
§3.6 and WI-16.

### 0.5 ⚠ Two expectations that turned out wrong

**Augmentations: there is essentially nothing to reuse.**
`zea/data/augmentations.py` contains exactly **one** class, `RandomCircleInclusion`.
Policies A and B remain entirely ours. (Some useful primitives live elsewhere —
`zea.func.ultrasound.make_tgc_curve`, `zea.ops` normalize/log-compress — but there
is no augmentation library.)

**TAESD is not an ultrasound model.** zea's `taesdxl` preset is the stock
**Stable Diffusion XL Tiny AutoEncoder** re-implemented in Keras (its own docstring
links `madebyollin/taesd`). It is a *natural-image* latent space — the same category
as DINOv3, not a domain baseline. It is also TensorFlow/JAX-only.

Genuinely ultrasound-domain pretrained models in `zea.models.presets`:

| Preset | Domain | Use as baseline? |
|---|---|---|
| `hvae` | **EchoNet-LVH cardiac B-mode**, 256×256 | **Yes** — real ultrasound latent space |
| `speckle2self-invivo` | in-vivo ultrasound speckle reduction | **Yes** — encoder features; speckle-aware |
| `carotid-segmenter`, `augmented_camus_seg`, `echonet*`, `mobilenetv2_regional_quality` | cardiac/carotid B-mode | task-specific, lower value |
| `taesdxl` | **natural images (SD-XL)** | yes, but as a *second natural-image control*, not a domain baseline |
| `lpips` | natural images | metric only |

⇒ Baseline set becomes: DINOv3 (natural image), TAESD (natural image), **HVAE**
(ultrasound), **Speckle2Self** (ultrasound), plus our from-scratch models. Note all
four are B-mode-only, which is the point (§4.4).

### 0.6 Backend split: confirmed, and OpenH-RF already does it

zea is Keras 3; our training is PyTorch. **Use zea with `KERAS_BACKEND=jax` for the
offline beamforming pipeline only, keep PyTorch for training.** OpenH-RF's own
scripts do exactly this (`os.environ.setdefault("KERAS_BACKEND", "jax")`).

One refinement on the proposal: install zea in a **separate virtualenv** rather than
alongside torch. keras ≥ 3.12.1 + JAX-CUDA + torch-CUDA in one environment is a
known source of CUDA-version conflicts, and there is no reason to share — the two
stages are already decoupled by the on-disk store, which is the interface. Two envs,
one data contract.

### 0.7 Version risk

`zea` 0.1.4, 24 releases, arXiv 2512.01433 (Dec 2025), JOSS. Young but active.

- **Pin `zea==0.1.4` exactly.** Not a range.
- **Known-unstable surface:** `File.create()`'s signature changed recently — OpenH-RF's converter carries the comment *"In this zea version the probe (incl. probe_geometry) is passed here, not in `scan`; the old `probe_name=` argument is no longer accepted."* That is an API break between adjacent versions. Wrap `File.create` behind one adapter function of ours.
- Docs still reference `0.1.0a3` URLs in places; the beamformer internals (`tof_correction`, `DelayAndSum`, `PatchedGrid`) look stable and are the parts we depend on most.
- Mitigation: our dependency is *narrow* — `tof_correction`, `Parameters`, `Probe`, `pixelgrid`, `File`, `Refocus`. If zea breaks, the fallback is our own summation over their delay maths, which is what revision 2 planned anyway.

### 0.8 Net effect on the plan

**Adopted:** zea for delays, TOF correction, phase rotation, f-number masking, pixel
chunking, grid construction, probe/parameter abstraction, file format, Verasonics
reading (`zea/data/convert/verasonics.py` handles `TX.Delay`/`Apod`/`focus`/`Origin`/
`Steer` — replacing most of `utils/load_hdf5_data.py`), REFoCUS for Stanford, and two
ultrasound-domain frozen baselines.

**Still ours:** the angular sector-id map and segment-sum op; PW forward synthesis;
the reference-validation gate; the entire store, split, augmentation, training and
evaluation stack; and (to verify) Stanford's HVMux receive-aperture remap, which I
did not find handled in zea's Verasonics converter.

**Made harder by zea:** nothing structural. The costs are a second virtualenv, a
0.1.x pin, and one adapter around `File.create`.

---

## 1. Repo investigation

### 1.1 What exists

The repo is a single Poetry package, ~5.6k lines of Python, no tests.

```
src/ssl_ultrasound_representation/
  train.py                       645  MAE training loop + argparse CLI
  dataset/dataset.py             499  HF/zarr loader, splits, torch Dataset
  eval/latent.py                 465  latent extraction, PCA/t-SNE, plots
  mae/model/us_mae.py            329  SignalMAE (ViT enc/dec over channel data)
  mae/model/us_mae_transformer.py 194  dead alternative model, unused
  mae/model/pos_encoding.py       56  sinusoidal + N-D grid PE
  mae/model/loss.py               27  patch MSE
  mae/model/linear_projection.py  15  two functions that raise NotImplementedError
  utils/beamform.py              131  PyMUST DAS wrapper for monitoring snapshots
  utils/monitoring.py            237  W&B: scalars, image panels, checkpoint artifacts
  utils/load_hdf5_data.py        236  Verasonics .mat readers
scripts/dataset_scripts/         ~1.3k  build / push / inspect / visualize
src/dataset.py                    24  back-compat shim re-exporting the package module
```

Git: 13 local branches, currently on `feature/test-sso`; `main` exists on origin.
Recent history is MAE debugging (`overfit_single_sample`, decoder width, patch
embed aliasing) — consistent with the stated "it does not work".

### 1.2 Reusable independent of data source

| Component | File | Verdict |
|---|---|---|
| W&B monitor: init, scalar logging, checkpoint artifact + pruning, `WANDB_CACHE_DIR` handling | `utils/monitoring.py:43-237` | **Keep almost as-is.** The only data-coupled part is `log_reconstructions` / `_comparison_figure` / `set_vis_frames` (lines 97-159), which assume `(N,transducer,scanline,2,sample)` and a `ZoneBeamformer`. Everything else is source-agnostic and genuinely good (the artifact-prune-after-`wait()` and the RAM-overlay cache fix are hard-won). |
| Device / AMP selection | `train.py:58-78` `default_device`, `_amp_beneficial` | **Keep verbatim.** `_amp_beneficial` gating on sm_70+ is correct and worth carrying. |
| Worker-count clamp via `sched_getaffinity` | `train.py:116-121` | **Keep.** Correct cgroup-aware behaviour. |
| LR schedule (linear warmup → cosine, per step) | `train.py:348-367` | **Keep the shape**, drop the `overfit_single_sample` special case. |
| HF chunk-selective download (`allow_patterns` built from per-frame chunk indices) | `dataset/dataset.py:148-183` | **Port the mechanism, not the file.** This is the only genuinely non-obvious piece of the loader and it will be needed verbatim for the published K=16 store. |
| Split-disjointness assertion | `dataset/dataset.py:216-225` `_assert_disjoint` | **Port.** Generalise the key from `subject` to `frame_id` (see §6.2). |
| CLI-flag-to-config pattern incl. the `is not None and is not False` identity check | `train.py:629-641` | **Keep the idiom.** The comment explaining why `not in (None, False)` drops `--weight-decay 0` is correct and easy to re-break. |
| Projection + plotting helpers | `eval/latent.py:233-315` `project`, `_scatter`, `plot_embedding` | **Keep the plotting**, replace the label plumbing. `_scatter`'s categorical/continuous auto-switch is reusable as-is. |
| Position-regression residual analysis | `eval/latent.py:250-274` `regress_out` | **Keep and promote.** This is exactly the tool needed for the depth-confound problem in §4.3 — it is currently the most valuable thing in the eval module. |
| Poetry/pyproject, `.tool-versions`, `requirements.txt` | root | Keep, but see §1.4 (torch pin must change). |
| Tests | — | **Do not exist.** Zero test files, no `conftest.py`, no CI. This is written from scratch. |

### 1.3 Every place the 160-frame format is assumed

The corpus is actually **168 frames / 84 `.mat` files / 20 subjects**
(`benbarkow/us-ssl-mf`, verified live); the local `data/processed/` copy is a
14-frame / 7-file / 2-subject subset. Format assumptions, exhaustively:

**Hardcoded tensor shapes and constants**

- `train.py:175-177` — `n_transducers = 192`, `n_scanlines = 192`, `n_samples = 215`. Not read from metadata; typed as literals.
- `train.py:178-181` — `depth_window=43`, `transducer_window=64`, and the derived `n_depth_windows = 215//43 = 5`, `n_transducer_groups = 192//64 = 3`. 43 is chosen solely because it divides 215.
- `mae/model/us_mae.py:24-29` — the same 192/192/215/43/64 as constructor defaults, plus `assert n_samples % depth_window == 0` and `assert n_transducers % transducer_window == 0` (lines 52-60).
- `mae/model/us_mae.py:70` — `patch_dim = 2 * transducer_window * depth_window`, i.e. the "2" for IQ is baked into the patch dimension.
- `mae/model/us_mae.py:281-304` — `patchify` / `unpatchify` hardwire the 5-D `(B, T, Sc, IQ, D)` layout and the `(scanline, transducer_group, depth_window)` token order.
- `eval/latent.py:112` — **`n_transducers, n_scanlines, n_samples = 192, 192, 215` re-hardcoded a second time**, in the legacy-checkpoint path, with the comment "Data geometry is fixed by the pipeline (hardcoded in train.py)".
- `eval/latent.py:117-126` — token-count consistency check built from the same three literals; raises if they disagree.
- `eval/latent.py:152-163` — `_token_grid_labels` decodes token index → (depth_window, aperture_group, scanline) assuming that exact grid.
- `utils/beamform.py:33` — `nx: int = 192` default output width; `nz: int = 256`; `strip_halfwidth_m: float = 1.5e-3` and `fnumber: float = 1.5` are magic numbers tuned for this probe.
- `scripts/dataset_scripts/build_mae_dataset.py:42-44` — `DEPTH_CUTOFF = 860`, `DOWNSAMPLE_FACTOR = 4`, `N_SAMPLES_OUT = 215`.
- `scripts/dataset_scripts/rechunk_zarr.py:6` — chunk shape `(1,32,32,1,2,215)` in the docstring/logic.
- `scripts/dataset_scripts/build_mae_dataset.py:57` — `frame_chunks()` returns a 6-tuple; the arity itself encodes the layout.

**Normalization constants — the important one**

- `train.py:22-34` `normalize_frame` performs **per-depth RMS normalization**: it divides every fast-time sample by its RMS over `(transducer, scanline, iq)`. The docstring says this explicitly — "Raw IQ decays ~2 orders of magnitude from shallow to deep […] we divide each depth sample by its RMS". **This is precisely the operation the new brief forbids** (§3 of the task: dataset-level scaling, not per-patch, because attenuation is a label axis). It is applied in both `train.py:259` and `eval/latent.py:344`, so it silently contaminates the existing latent analysis too. Any depth/attenuation result produced by the current eval is invalid.

**Axis-order / semantic assumptions**

- `dataset/dataset.py:41-46` — filenames `ultrasonic_dataset.zarr(.zip)`, array name `rf_iq`.
- `dataset/dataset.py:334-353,399-404` — `UltrasoundMAEDataset` slices `arr[idx, :, :, focal_zone]`, i.e. axis 3 *is* the focal zone. `focal_zone` is threaded through `build_split_datasets` (line 430), `train.py:260`, `eval/latent.py:345`, `signal_decay.py:74,136`, `utils/beamform.py:32,45`, `monitoring.py:51,75`.
- `dataset/dataset.py:155-163` — the `max_frames` chunk-glob trick depends on the frame axis being chunked 1-per-chunk *and leading*.
- `train.py:398`, `eval/latent.py:198-202` — model called as `model(x, pos_embed)` with `x` 5-D.

**Split logic keyed to Stanford provenance**

- `dataset/dataset.py:227-281` `split_by_subject` — requires `metadata["frames"][i]["subject"]`. There is no subject in the NVIDIA data.
- `dataset/dataset.py:284-330` `split_by_file` — requires `relative_path`, and its whole rationale ("both frames of the same `.mat` cine stay together", "each Verasonics file holds `numFrames=2`") is Stanford-specific.
- `train.py:257-258`, `eval/latent.py:341-342` — `split_strategy="subject"`, `ratios=(0.85,0.15,0.0)` defaults.
- `eval/latent.py:179-181,193` — `idx_to_subject` map; subject is the only frame-level label the current eval has.
- `scripts/dataset_scripts/push_to_hf.py:117-118,127-128,190-196,277-287` — dataset-card generation counts by `subject` and `subject × subfolder`.

**Beamforming geometry frozen into a config**

- `configs/beamform_params.json` (1.7 MB) — `c=1540`, `fc=7.813e6`, `fs=7.8125e6`, `pitch=200 µm`, `n_elements=192`, `n_samples=215`, `t0=5.12e-7`, 192 `scanline_origins_m`, 3 `focal_depths_m`, and a full `tx_delays_s[3][192][192]` table. Consumed by `utils/beamform.py:38-58` and `monitoring.py:74-75`. Entirely Stanford-specific and entirely useless for FSA data.
- `scripts/dataset_scripts/export_beamform_params.py` exists only to regenerate that file from a `.mat`.

**Verasonics-specific I/O (whole files, no reusable core)**

- `utils/load_hdf5_data.py` (236 lines) — `Resource.Parameters.speedOfSound`, `Trans.frequency`, `Receive.ADCRate`, `TX.Apod`, HVMux aperture handling.
- `scripts/dataset_scripts/visualize_beamformed_pymust.py` (457), `build_mae_dataset.py` (340), `inspect_metadata.py` (156), `focal_zone_comparison.py` (218), `load_data.py` (220), `signal_decay.py` (188), `rechunk_zarr.py` (90).
- `build_mae_dataset.py:106-125` — the cyclic HVMux channel→element remap (`phys = (rx_ap-1)+i`, `src = ((rx_ap-1)+i) % n_ch`). Hard-won and correct for Stanford; meaningless for NVIDIA.

**Test fixtures**: none exist, so nothing to migrate — but also no safety net for any of the above.

### 1.4 What must be written from scratch

1. NVIDIA parquet frame reader (streaming, one frame at a time).
2. GPU sub-aperture delay-and-sum for **baseband IQ** on an FSA transmit sequence, with the phase-rotation term (§3.2). PyMUST cannot do this — `pymust.dasmtx` builds a per-scanline sparse matrix for a focused, single-transmit, scanline-oriented acquisition; FSA needs a (tx × rx) double sum on a Cartesian grid and a K-way partitioned accumulator.
3. The whole offline build pipeline: shard iteration, resume, manifest, validation, normalization statistics.
4. The K=16 store + reader + the K-reduction runtime transform.
5. Splits + leakage assertion keyed to `frame_id`.
6. Augmentation policies A and B.
7. ResNet-18 with an adapted stem, VICReg projector + loss, training loop (the MAE loop's *scaffolding* survives; its body does not).
8. The entire evaluation harness: probes, clustering, baselines, results matrix, confound-first plotting.
9. Tests + CI. From zero.

**Rough accounting:** of ~5.6k lines, ~700 survive as-is (monitoring, device/AMP, LR schedule, plotting primitives, `regress_out`, `_assert_disjoint`, the HF chunk-glob trick), ~1.2k are ported-with-rewrite (train loop scaffolding, dataset wrapper, eval orchestration), and ~3.7k are deleted.

---

## 2. Dataset investigation — `nvidia/NV-Raw2Insights-US`

All figures below were **measured** by reading the small columns of all 95 parquet
shards (923 rows) over HTTP range requests. The `iq_*` payload was deliberately
not fetched.

### 2.1 On-disk layout and format

- HuggingFace dataset repo `nvidia/NV-Raw2Insights-US`, public, **not gated**, no access request.
- Apache Parquet, HF `datasets`-style Arrow extension arrays.
- `data/train-{00000..00044}-of-00045.parquet` — ~5.43 GB each, 18–19 rows each, **830 rows**.
- `data/validation-{00000..00049}-of-00050.parquet` — 0.44–0.71 GB each, 1–2 rows each, **93 rows**.
- **Total 923 rows, 271.36 GB download / 247.16 GB decoded.** One row group per shard.
- Created 2025-12-01, last modified 2026-04-21.

### 2.2 Per-frame tensor shapes and dtype (verified against the Arrow schema)

| Field | Shape | Dtype | Notes |
|---|---|---|---|
| `iq_real` | `[180, 180, 1024]` | float32 | axes `[n_tx, n_rx, n_ax]`, **265.4 MB/frame with `iq_imag`** |
| `iq_imag` | `[180, 180, 1024]` | float32 | |
| `bmode` | `[507, 456]` | float32 | synthetic-aperture DAS envelope, **not** log-compressed |
| `bmode_focused` | `[507, 456]` | float32 | focused-transmit envelope |
| `bmode_extent` | `[4]` | float32 | **mm**, `[x_min, x_max, z_max, z_min]` |
| `sound_speed_map` | `[32, 32]` | float32 | m/s |
| `sound_speed_extent` | `[4]` | float32 | **m** (unit differs from `bmode_extent`) |
| `segmentation_map` | `[507, 456]` | uint8 | 0 = background tissue, 1 = cyst |
| `phase_error` | scalar | float32 | radians |
| `fs`,`fc`,`fd`,`t0`,`c0` | scalar | float64 | |
| `elpos` | `[3, 180]` | float32 | m, `[x,y,z]` per element |
| `frame_idx` | scalar | int32 | |

Storage per frame: IQ 265.4 MB, everything else 2.1 MB.

### 2.3 Acquisition geometry (measured)

| Quantity | Value | How obtained |
|---|---|---|
| Elements | 180, linear | `elpos.shape` |
| Pitch | **201.0 µm exactly** (σ = 2.3e-10 m) | `mean(diff(elpos[0]))` |
| Aperture | 35.979 mm | `elpos[0,-1] - elpos[0,0]` |
| Element y, z | all 0.0 | `elpos[1]`, `elpos[2]` |
| Sampling rate `fs` | 13.330 MHz | constant over all 923 |
| Centre freq `fc` | 6.5 MHz | constant |
| Demod freq `fd` | 6.5 MHz | constant — **data is baseband IQ** |
| Sound speed `c0` | 1540 m/s | constant |
| `t0` | **3 discrete values**: −1.76201e-7, −1.66412e-7, −1.56623e-7 s | see §2.4 |
| Fast-time samples | 1024 → 76.8 µs | |
| Transmit sequence | **Full synthetic aperture: 180 transmits, one per element** | card + `iq` axis 0 |
| Wavelength λ | 236.9 µm → pitch = **0.85 λ** | derived |
| Reference grid | 507 (z) × 456 (x), x ∈ [−17.9895, 17.9895] mm, z ∈ [0, 40] mm | `bmode_extent`, constant over all 923 |
| Grid spacing | dz = 79.05 µm, dx = 79.07 µm → **isotropic, ≈ λ/3** | derived |

Two sanity checks I did on paper: (i) the worst-case two-way path in the FOV
(corner pixel, farthest tx and rx) is 107.6 mm → 69.9 µs → sample index 934 < 1024,
so **1024 samples covers the whole FOV** for every tx/rx pair; (ii) pitch = 0.85 λ
> λ/2, so **grating lobes exist for steered synthesis** — relevant if anyone later
synthesises steered plane waves from the FSA data (§6.3).

### 2.4 ALL available metadata and labels — enumerated

Per-frame fields, and what each is actually worth as a probe axis:

| # | Field | Type | Measured range / distribution | Probe value |
|---|---|---|---|---|
| 1 | `sound_speed_map` | 32×32 continuous, m/s | global 1283.3 – 1629.5; per-frame mean 1380.8 – 1530.8 (σ_between = 47.1); per-frame σ_within 3.7 – 28.6 | **Primary.** Both a frame-level scalar (`sos_mean`) and a spatially-resolved field that can be resampled onto crops. |
| 2 | `segmentation_map` | 507×456 binary | 0 = tissue, 1 = cyst | **Primary.** Pixel-aligned with our output grid → per-crop `cyst_fraction` and `contains_cyst` for free. Has known label noise (§2.6). |
| 3 | `phase_error` | scalar, rad | 923 distinct values; 1.2407 – 2.8059, mean 1.8265, σ 0.2773 | **Secondary — partly redundant.** corr(`phase_error`, `sos_std`) = **+0.756**, corr with `sos_mean` = −0.383. 57 % of its variance is shared with SoS heterogeneity. |
| 4 | `t0` | scalar, s — **3 discrete levels** | n = 435 / 116 / 372; in cycles of fc: −1.1453 / −1.0817 / −1.0181 | **The only categorical acquisition-parameter axis in the dataset.** Present in both splits (train 390/108/332, val 45/8/40). See the caveat in §6.4 — this is a beamformer *input*, so probing it post-beamforming may be at chance by construction, and that is the correct answer, not a bug. |
| 5 | `fs`, `fc`, `fd`, `c0` | scalar | **exactly one distinct value each** across all 923 | Zero probe value. Constants. |
| 6 | `bmode_extent`, `sound_speed_extent` | 4-vector | **exactly one distinct value each** across all 923 | Zero probe value. Fixed FOV. |
| 7 | `elpos` | 3×180 | identical geometry every frame | Zero probe value. |
| 8 | `frame_idx` | int32 | **always 0**, all 923 rows | Zero probe value — one frame per simulation. |
| 9 | `bmode`, `bmode_focused` | 507×456 | reference reconstructions | Not labels, but **beamforming ground truth** and a second "signal stage". |
| 10 | split membership | train / validation | 830 / 93 | Negative control — should probe at chance. |
| 11 | Derived (ours) | crop `z_centre`, `x_centre`, `sos_mean/std/grad` over the crop, `cyst_fraction`, `n_cysts` | | Geometry axes 6–7 of §4.1. |

**There is no session ID, no acquisition ID, no probe preset, no depth setting, no
timestamp, no phantom ID, no tissue-type annotation beyond binary cyst/background,
and no subject.** `frame_idx` is 0 for every row and is the *only* identifier field.
Provenance is recoverable only as `(shard_filename, row_index)`, which is a storage
artefact, not semantics.

**⚠ Loudly, as requested: the metadata is thin in exactly the dimension the brief
assumed, and rich in a dimension the brief did not anticipate.** There are no
acquisition/session/preset axes at all (§6.1). There are, however, dense
spatially-resolved physics labels — a continuous sound-speed field and a
pixel-aligned binary tissue mask — which is a *better* substrate for linear probing
than the categorical labels the brief was designed around. The evaluation design
survives; the label axes have to be substituted (§4.1).

### 2.5 Provenance, licence, and redistribution

- **Source:** synthetic, generated with **k-Wave** acoustic simulation. Ground truth comes from the simulation parameters, not annotation. Owner: NVIDIA Corporation. Authors: Walter Simson, Sean Huver.
- **Licence: CC BY 4.0** (`https://creativecommons.org/licenses/by/4.0/legalcode.en`). Not gated, marked "ready for commercial use".
- **Redistribution of a derivative is permitted.** CC BY 4.0 explicitly allows remixing and redistribution in any medium or format, including commercially, subject to (i) attribution, (ii) a link to the licence, and (iii) **an indication that changes were made**. Beamforming is a change; the card must say so.
- ⇒ **No blocker to publishing the K=16 derivative on HuggingFace.** Private hosting is not needed. Requirements for our card are in Work Item 9.
- Note the licence split in the companion GitHub repo `NVIDIA-Medtech/NV-Raw2insights-US`: **code is Apache-2.0, model weights are CC BY-NC 4.0**. We use neither, but do not accidentally vendor the weights — CC BY-NC would contaminate a commercially-usable release.
- Reference code worth reading (Apache-2.0, so freely reusable): `inference.py` documents the intended IQ normalization (`iq /= IQ_RMS` with `IQ_RMS = 0.6616`) and confirms the `[n_tx, n_rx, n_ax]` axis order by how it flattens.

### 2.6 Known issues declared by NVIDIA

Quoted from the card, all of which bound what our probes can achieve:

1. `bmode_extent` is in **millimetres** while `sound_speed_extent` and `elpos` are in **metres**. A unit slip here silently mis-registers labels to pixels.
2. `bmode_extent` z-ordering is `[x_min, x_max, z_max, z_min]` — **z_max before z_min**, unlike matplotlib's `extent`.
3. **Cysts touching the frame edge are not in `segmentation_map`.** So `cyst_fraction = 0` does not mean "no cyst" near the borders — a systematic false-negative band.
4. **Regions of gross reverberation artefact are incorrectly segmented as cyst.** So the cyst probe has irreducible label noise, and a perfect encoder cannot reach 100 %.

Additionally, measured by me and *not* on the card:

5. `sound_speed_map` values (1283–1630 m/s, frame means 1381–1531) sit systematically **below** the declared `c0 = 1540 m/s`, and NVIDIA's own `inference.py` multiplies its network output by 1000 before comparing to this field. Absolute calibration of this map should be treated as unverified; **relative** structure is safe to probe. Flagged as an open question (§7).
6. **75 of 923 frames (8.1 %) have an entirely empty `segmentation_map`** — 63 train, 12 validation, measured. Combined with NVIDIA's known issue #3 (edge-touching cysts unlabelled), an empty mask does *not* reliably mean "no cyst in this frame". These 75 frames are excluded from the cyst probes by default and reported as a separate row; including them would inject an unknown mix of true negatives and unlabelled positives directly into the primary tissue axis. Cyst area fraction over the remaining 848 frames: median 3.7 %, IQR 2.2–5.7 %, max 14.4 %.

---

## 3. Design decisions where latitude was left

### 3.1 Migration: option (b) — cut the old path

**Decision: (b).** No dataset abstraction over the two sources.

Reasoning, concretely: the two formats share no operation. Stanford is
*pre*-beamforming multi-focal channel data on a (192 element × 192 scanline × 3
focal zone × 215 sample) grid with focused transmits and an HVMux channel remap.
NVIDIA is FSA channel data that we turn into *post*-beamforming Cartesian images on
a (507 × 456 × 16 sub-aperture) grid. A common `Dataset` interface over those two
would be a union type whose only shared method is `__len__`. The abstraction would
cost real complexity and buy nothing.

**But there is a seam, and it is not the loader.** If the 168-frame Stanford set is
ever used as a transfer-eval corpus (§3.6), it must be beamformed into the *same*
`(2, Z, X, 16)` output format first. So the shared contract is the **output store
format**, with two entirely independent producers writing into it. That is a
one-way data format, not a loader interface, and it costs nothing today. Building
the store reader against a documented format rather than against "the NVIDIA
pipeline's output" is the whole of the accommodation.

Old code is deleted from the working tree in Work Item 1 and lives in git history
on `main`. The specific mechanisms called out as reusable in §1.2 are ported
deliberately, into new files, with their explanatory comments intact.

### 3.2 Sub-apertures are ANGULAR SECTORS, not element groups

**Decision (revised): for each output pixel, compute each receive element's view
angle to that pixel, bin elements into K = 16 uniform angular sectors spanning
±θ_max, and sum within each sector. All 180 elements are used.**

Rationale (adopted from review, and correct): element-indexed sub-apertures are not
comparable across probes. "Sub-aperture 3 of 16" subtends a different angular range
on a 180-element 0.201 mm-pitch array than on Stanford's 192-element array at a
different depth range, so a model that consumes sub-apertures as tensor channels
learns weights bound to a physical quantity that silently changes meaning between
datasets. **This would have broken §3.6 transfer without any visible symptom** — the
model would run, produce numbers, and the numbers would be meaningless. Angular
sectors mean the same thing on any array.

The previous element-group decision (central 176 elements, 11 per group) and its
2.2 % aperture sacrifice are **withdrawn**. Angular binning has no divisibility
constraint; use all 180.

**Nesting / the ladder.** Uniform binning in angle over a fixed span is *automatically*
nested: 16 uniform sectors merge pairwise into 8 uniform sectors, and so on to 1.
The ladder K = 16 → 8 → 4 → 2 → 1 holds by construction. ✔

**⚠ Per-sector normalization conflicts with the ladder unless the store is raw.**
Sectors contain different element counts, so raw amplitude differences are a trivial
shortcut and must be normalized away — but if normalized sector means `N_k = S_k/n_k`
are what is *stored*, then `N_{2j} + N_{2j+1} ≠ (S_{2j}+S_{2j+1})/(n_{2j}+n_{2j+1})`
and the ladder breaks.

**Resolution: store raw sector sums `S_k`; store the occupancy map separately; apply
normalization as a runtime transform.** Then the ladder is plain pairwise addition
(exact, cheapest), and normalization at any K is a divide by the correspondingly
merged occupancy. The occupancy map `n[z, x, k]` depends only on geometry, **not on
the frame**, so it is computed once and stored once: 507 × 456 × 16 × 2 B = 7.4 MB
for the entire dataset. Default normalization is `S_k / n_k` (sector *mean*, which
equalizes signal level); `S_k / √n_k` (equalizes incoherent-noise variance) is
available as a flag. Empty sectors (`n_k = 0`) are written as 0 and flagged in the
mask.

**Implementation note that makes this free.** For a linear array, view angle is
*monotonic in element index* at fixed pixel, so each angular sector is a **contiguous
element index range**. Sector membership is therefore a precomputed, frame-invariant
`sector_id[z, x, e]` int8 map (507 × 456 × 180 = 41.6 MB, resident on GPU), and the
inner loop is an ordinary DAS accumulation with a scatter-add into 16 buffers.
**Angular binning costs essentially nothing over plain DAS** — no K-fold overhead.

**⚠ Angular sectors are sparse at shallow depth, and this is a hard physical limit.**
The review's secondary argument — that fixed element groups at shallow depth sit
outside the active aperture and contribute noise — is right, but angular binning does
not solve the shallow-depth problem so much as *transform* it: element groups at
shallow depth contain elements that shouldn't be there; angular sectors at shallow
depth contain **too few elements to fill 16 bins at all**. Measured, elements per
sector:

| z (mm) | f# = 1.5 (±18.4°) | f# = 1.0 (±26.6°) | f# = 0.75 (±33.7°) |
|---:|---:|---:|---:|
| 2 | 0.4 | 0.6 | 0.9 |
| 5 | 1.1 | 1.5 | 2.1 |
| 8 | 1.7 | 2.5 | 3.4 |
| 10 | 2.1 | 3.1 | 4.2 |
| 15 | 3.1 | 4.7 | 6.2 |
| 20 | 4.2 | 6.2 | 8.3 |
| 30 | 6.2 | 9.4 | 11.2 |
| 40 | 8.3 | 11.2 | 11.2 |

≥ 2 elements/sector requires z ≥ 9.5 mm at f# = 1.5, z ≥ 6.2 mm at f# = 1.0,
z ≥ 4.8 mm at f# = 0.75. This is a property of a 180-element array, not of the
binning scheme; no choice of sub-aperture definition escapes it.

**Decision: θ_max from f# = 1.0 (half-span 26.57°).** f# = 1.0 keeps ≥ 3
elements/sector from z = 9.5 mm down; f# = 0.75 buys little because element
directivity (element width ≈ 0.85 λ, −6 dB near ±35°) is attenuating those views
anyway.

**⚠ Revised in rev-3: the 8 mm bound is a SAMPLING rule, not a storage rule, and it
applies only at high K.** Sparsity is a property of *K*, not of the data — the raw
sector sums at shallow depth are sparse but perfectly valid, and merging them down
the ladder repopulates them. Measured, elements per sector at f# = 1.0:

| z (mm) | active | K=16 | K=8 | K=4 | K=1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.3 | 0.6 | 1.2 | 5 |
| 2 | 10 | 0.6 | 1.2 | 2.5 | 10 |
| 4 | 20 | 1.2 | 2.5 | 5.0 | 20 |
| 8 | 40 | 2.5 | 5.0 | 10.0 | 40 |
| 15 | 75 | 4.7 | 9.4 | 18.8 | 75 |
| 21.6 | 108 | 6.8 | 13.5 | 27.0 | 108 |
| 40 | 180 | 11.2 | 22.5 | 45.0 | 180 |

**K = 4 and K = 1 are fine everywhere**, including z = 1–2 mm. Only K = 16 (and
marginally K = 8) needs the restriction.

⇒ **Store the full 0–40 mm depth range.** Apply `z_min = 8 mm` as a crop-sampling
constraint **only on K ≥ 8 experiments**; K ≤ 4 samples the full range. This is a
config value per experiment, not a property of the store.

### 3.2c ⚠ Answering the rev-2 ambiguity: whose 8 mm bound is it?

**It is entirely mine.** Measured, and constant across all 923 frames:

```
bmode_extent = [-17.9895, 17.9895, 40.0, 0.0] mm   →  z spans 0.0 to 40.0 mm
```

NVIDIA's simulation has data from **z = 0**. The 8 mm floor introduced in revision 2
was my own f#-driven sector-occupancy exclusion, not a property of the source. Rev-2
read ambiguously on this; it should not have.

**Consequences, and they are good:**

- **Stanford's 0–21.6 mm is entirely inside NVIDIA's 0–40 mm extent.** Nothing about shallow Stanford is out of distribution.
- At **K ≤ 4** the transfer depth problem **dissolves completely** — full-range training, full-range transfer, no restriction on either side.
- At **K = 16** the restriction is ours and applies to *both* corpora identically, so it is a matched constraint rather than a domain gap.
- The rev-2 statement that "the headline transfer comparison must be restricted to 8–21.6 mm" is therefore **withdrawn for K ≤ 4** and retained only for K ≥ 8.

### 3.2d Crop size and pixel spacing — decided, and one premise corrected

**⚠ Crop size does not fix the stored grid.** Pixel *spacing* does. Crop size is a
sampling parameter and can be changed at any time without rebeamforming, so it is
not a decision that must precede the beamformer. Only option (c) is irreversible.

Second, the arithmetic improves once the real spacing is used. Our grid is
**0.079 mm/px** (the dataset's own), not 0.1 mm/px:

| Band | extent | rows | 128-crop positions | 64-crop positions |
|---|---:|---:|---:|---:|
| 0–21.6 mm (Stanford full, K ≤ 4) | 21.6 mm | 273 | **146** | 210 |
| 8–21.6 mm (K ≥ 8 restricted) | 13.6 mm | 172 | **45** | 109 |

A 128-px crop is 10.12 mm, not 12.8 mm. So the K ≤ 4 transfer arm has 146 distinct
depth positions — ample, not "effectively one crop position". The concern was real
but it was an artefact of the 0.1 mm/px estimate compounding with my own 8 mm floor.

**Decisions:**

- **(c) — keep 0.079 mm/px (the dataset's own grid). Not negotiable.** It is what buys the pixel-aligned `segmentation_map`, the direct `bmode` validation gate, and zero-resampling label attachment (§3.3). Changing it forfeits all three.
- **128 × 128 as the primary crop size** for every arm that transfers at 128.
- **⚠ Crop size matches within an arm, never across the train/transfer boundary.** The K ≥ 8 arm transfers at 64 × 64 (only 45 positions at 128, §3.2d), so **that arm is also trained at 64 × 64.** Training at 128 and transferring at 64 would reintroduce precisely the feature-statistics mismatch that motivated choosing a single primary size — receptive-field coverage, GAP pooling extent, and effective batch statistics all shift with crop size. This corrects rev-3, which specified 64 for the transfer side only and was wrong.
- ⇒ **Two training configurations, not one:** `K ≤ 4 @ 128×128` (full depth range) and `K ≥ 8 @ 64×64` (z ≥ 8 mm). Each is internally consistent from pretraining through probe through transfer. Crop size joins K as an arm-defining parameter rather than a global constant.
- Option (b) — 64 everywhere — remains **rejected**: it would halve spatial context in the K ≤ 4 arm, which has no positional shortage at all (146 positions).
- Cross-arm comparisons (K = 16 vs K = 1) therefore differ in crop size as well as K. **That confound is stated in the results matrix**, and the clean K-ablation is run *within* a crop size: K ∈ {4, 2, 1} at 128, K ∈ {16, 8} at 64, with K = 4 built at both sizes as the bridging cell.

**⚠ This supersedes §3.4.** Angular sectors *are* a receive f-number by construction —
there is no longer a "no receive apodisation" option. K = 1 is now an f#-limited
full-aperture DAS, which may or may not match NVIDIA's `bmode`. WI-4 therefore tests
**three** variants against the reference (unlimited aperture, f# = 1.5, f# = 1.0) and
the sector span is set to whichever matches. This turns §3.4 from a standing decision
into an empirical question with a gate.

**Storage (angular, transmit-compounded):** unchanged at 2 × 507 × 456 × 16 × 4 B =
**29.59 MB/frame → 27.3 GB**, plus 7.4 MB occupancy and 41.6 MB sector map, both
once. See §3.2b for the transmit-angle addition.

### 3.2b Transmit diversity: synthesize plane waves, costed

The earlier claim that "there are no angles to subset" was **true as-stored and wrong
in principle**, and the review is right to push back. The full 180 × 180 Tx–Rx matrix
is the universal transmit dataset: a plane wave at angle α is
`p_α,rx(t) = Σ_tx s_tx,rx(t − x_tx sin α / c)`, so any transmit sequence is
synthesizable post hoc. Transmit-angle subsetting is the one *physically exact*
positive-pair generator available — two angles observe identical scatterers, so the
pair is correct by construction rather than by augmentation heuristic.

**Compute cost is negligible.** Synthesis is a delay-and-sum along the transmit axis
(180 × 1024 per angle per rx). Beamforming a synthesized plane wave is then a single
sum over rx (180 terms/pixel) instead of the FSA double sum (32 400 terms/pixel), so
each angle costs **1/180 of an FSA beamform**. Five angles ≈ **+2.8 % compute.**
The entire cost is storage.

**⚠ Two hard constraints, both measured.**

1. **Grating lobes cap steering at ±10.3°.** pitch/λ = 0.201/0.2369 = 0.848 > 0.5, so λ/pitch = 1.1787 and the first grating lobe enters the visible region at |α| = arcsin(λ/pitch − 1) = **10.30°**. Verified: α = 10° → grating lobe at sin θ = −1.005 (evanescent, OK); α = 12° → −0.971 (**visible**). Usable angle sets are therefore A = 5: {−10, −5, 0, +5, +10}° and A = 7: {−9, −6, −3, 0, +3, +6, +9}°. Note that ±10° of transmit diversity is **narrower than the ±26.6° receive sector span** — the transmit axis is exact but its angular leverage is modest. That tempers, without refuting, the claim that losing it weakens Policy A more than anything else.
2. **A-angle compounding does not reproduce the FSA image.** Coherently summing 5 synthesized plane waves is not the same operation as full synthetic-aperture focusing over 180 transmits. So the PW stack cannot replace the FSA-compounded output — it is stored *alongside* it (which is what the review asked for), and the FSA-compounded K = 16 remains the reference-validation target and the primary training input.

**Storage, costed in full** (float32, 923 frames, 507 × 456 grid):

| Configuration | MB/frame | GB total | × base |
|---|---:|---:|---:|
| FSA-compound × K=16 (base) | 29.6 | 27.3 | 1.00 |
| FSA × K=16 + PW(5) × K=1 | 38.8 | 35.8 | 1.31 |
| **FSA × K=16 + PW(5) × K=4** | **66.6** | **61.5** | **2.25** |
| FSA × K=16 + PW(7) × K=4 | 81.4 | 75.1 | 2.75 |
| FSA × K=16 + PW(5) × K=8 | 103.6 | 95.6 | 3.50 |
| FSA × K=16 + PW(5) × K=16 | 177.6 | 163.9 | 6.00 |
| FSA × K=16 + PW(7) × K=16 | 236.7 | 218.5 | 8.00 |

**Decision: FSA × K=16 + PW(5) × K=4 — 61.5 GB, 2.25× base.** It is not prohibitive:
61.5 GB is a quarter of the 271 GB we already have to ingest, and well within what
HuggingFace hosts (NVIDIA's own source set is 271 GB). It keeps a full-resolution
receive-sector axis on the compounded image, gives Policy A a genuine 5-angle
transmit axis, and retains a 4-sector receive ladder *within* each angle so the two
aperture axes can be crossed. PW(5) × K=16 at 164 GB is where I would stop — it is
6× base for a joint resolution nothing in the current experiment plan consumes.

Marginals both recoverable: summing the PW stack's sector axis gives per-angle
full-aperture images; summing its angle axis gives the 5-angle-compounded K = 4
image (which is *not* the FSA image — see constraint 2).

**If storage must shrink**, drop to PW(5) × K=1 (35.8 GB, 1.31× base): Policy A keeps
its exact positive-pair generator, and only the joint (angle × sector) structure is
lost. Dropping the PW stack entirely is a deliberate choice with a named cost, not a
default.

### 3.3 Output grid: adopt the dataset's own grid verbatim

507 (z) × 456 (x), z ∈ [0, 40] mm, x ∈ [−17.9895, 17.9895] mm, spacing 79.05 µm ≈ λ/3.

Three reasons, all load-bearing:
1. K = 1 becomes pixel-for-pixel comparable to `bmode` — the validation gate is quantitative, not "looks about right".
2. `segmentation_map` is *already* on this grid → every 128×128 crop carries its own pixel-aligned cyst mask with no resampling. This is the single biggest simplification available.
3. `sound_speed_map` (32×32) resamples onto it once, via `sound_speed_extent`, giving per-crop SoS statistics.

A 128×128 crop is 10.1 × 10.1 mm — a physically sensible patch (≈ 43 λ).

### 3.4 Receive apodisation — SUPERSEDED by §3.2

Angular sector binning imposes a receive angular limit by construction, so the
earlier "no receive apodisation" decision no longer exists as an option. The
f-number that defines the sector span is now set empirically in WI-4 from three
candidates (unlimited, f# = 1.5, f# = 1.0) by which one matches NVIDIA's `bmode`,
with f# = 1.0 as the prior (§3.2). The transmit-side angular mask is unchanged.

The per-(z, x, k) occupancy map that the old §3.4 named as a contingency is now a
**required, always-built artefact** (7.4 MB, frame-invariant), because the
normalization in §3.2 depends on it.

### 3.5 The three beamforming traps

Written down now because the brief's warning — "a subtly wrong delay table produces
plausible-looking garbage" — has three specific realisations here:

1. **Baseband phase rotation.** The data is demodulated at `fd` = 6.5 MHz. After interpolating a trace at delay τ, the sample *must* be multiplied by `exp(+j·2π·fd·τ)` before summing. Omit it and the coherent sum destroys itself; the envelope still looks like speckle, so the image looks plausible and is wrong. This is the single most likely failure.
2. **`t0` sign and per-frame variation.** `t0` is negative (−1.6e-7 s) and takes **three different values across the dataset**. Sample index for delay τ is `(τ − t0)·fs`. Using a single hardcoded `t0` mis-registers depth by up to 0.13 samples for the wrong group — small, but it is exactly the kind of thing that silently degrades coherence. Read `t0` per frame.
3. **Unit mixing.** `bmode_extent` is mm; `sound_speed_extent` and `elpos` are m. NVIDIA flags this themselves. Every geometry constant in the code carries its unit in the variable name.

### 3.6 Stanford reinstated: sim-pretrain / real-probe

**Decision (revised): Stanford is a first-class transfer-evaluation corpus, not a
cut candidate.**

The reframing in review is correct and I had this backwards. Because NVIDIA is
k-Wave simulation, adding Stanford is not a corpus switch — it is the standard
**sim-pretrain / real-probe** structure:

- **pretrain** on NVIDIA (923 simulated frames, beamformed)
- **probe** on NVIDIA (dense physics ground truth: SoS field, cyst mask, aberration)
- **probe** on Stanford (168 frames, real in-vivo rat tissue — the transfer test)

This retires both §6.8 (tissue diversity is one phantom family — Stanford supplies
real tissue) and §6.9 (the project description names Stanford — it is still in the
project, as the evaluation corpus). Migration option (b) still holds for the
*training* path; the output store simply has two producers, which §3.1 already
identified as the real seam.

**Angular sectors are what make this possible at all.** Under the withdrawn
element-group scheme, sub-aperture channel k meant a different angular range on each
probe, so transferring a trained encoder would have silently mismatched its input
semantics. Angular sectors make channel k mean the same thing on both arrays. This
is the single change that turns transfer from unsound into sound.

**Shared grid — two variants, both cheap, run as an ablation:**

| Variant | NVIDIA spacing | Stanford spacing | 128-px crop |
|---|---|---|---|
| **mm-matched** | 79.05 µm | 79.05 µm | 10.12 mm on both; 42.7 λ vs 51.4 λ |
| **λ-matched** | λ/3 = 79.0 µm | λ/3 = 65.7 µm | 42.7 λ on both; 10.12 mm vs 8.41 mm |

Stanford's λ = 1540/7.813 MHz = 197.1 µm vs NVIDIA's 236.9 µm. Speckle grain scales
with λ, so λ-matching keeps texture statistics comparable while mm-matching keeps
anatomy scale comparable. The review's hypothesis that λ-matching transfers better is
plausible and untested; it is a resample parameter, so both are cheap.

**REFoCUS unifies the transmit representation too (new in rev-3).** `zea.ops.Refocus`
decodes Stanford's *focused* multi-focal transmits into **multistatic** form (§0.4).
That puts both corpora in the same transmit representation as NVIDIA's native FSA, so
the identical PW synthesis, identical sector binning and identical pipeline run on
both. Transfer is then matched on **three** axes — grid spacing, sector definition,
and transmit scheme — rather than one. This is a materially stronger position than
rev-2's grid-matching alone.

> **⚠ rev-6 correction (2026-08-16): Stanford is not only multi-focal, and
> `Refocus` is not needed.** Every statement above about decoding focused
> transmits was written from the phase-1 repo, which only ever handled the
> MultiFocal acquisitions. The published source
> ([figshare 28291985](https://doi.org/10.25452/figshare.plus.28291985),
> *Pulse-Echo Ultrasound Murine In Vivo Acquisitions*, Apache-2.0) ships the
> **same animals acquired three ways**: 117 FSA files (76.3 GB), 117 Hadamard,
> and 81 MultiFocal + 33 MF_Fast. On the FSA files `TX.Apod` has exactly **one
> non-zero element per transmit**, 192 transmits for 192 elements — natively
> multistatic, exactly like NVIDIA's 180-transmit corpus.
>
> So the transmit axis is matched *by construction*, with no inversion step,
> no choice of adjoint/Tikhonov/rSVD/tSVD, and no `Refocus` residual to
> account for. §4.5a's "reciprocity will not hold on Stanford" loses one of its
> three named causes. The MultiFocal path is not deleted — it stays available
> as an ablation, and the Hadamard set is a third encoding of the same scenes —
> but the headline transfer arm should be FSA-on-FSA.
>
> Two other numbers above need revising with it: Stanford's depth range is
> **0.39–21.2 mm** (the publishers truncated every file at ~846 fast-time
> samples; the "0–21.6 mm" figure came from the same truncation read through
> the old pipeline), and the corpus is now **117 acquisitions over 30 inferred
> sessions**, not 168 frames. See `scripts/wi16_fetch_stanford_fsa.py`.

**⚠ Depth: mostly resolved by §3.2c.** NVIDIA spans **0–40 mm** and Stanford
0–21.6 mm, so Stanford is entirely inside NVIDIA's extent. At **K ≤ 4 there is no
restriction on either side** and the headline transfer number uses the full range
(146 distinct 128-px crop positions on Stanford, §3.2d). At **K ≥ 8** our own 8 mm
sector-occupancy floor applies — identically to both corpora, so it is a matched
constraint, not a domain gap — leaving 8–21.6 mm and 45 crop positions, which is where
the 64-px secondary crop size earns its place. The depth-restricted control row is
reported at every K regardless, as the falsification instrument for §4.5c.

**⚠ Stanford's receive aperture is transmit-dependent, and zea may not cover it.**
It is a focused multi-focal sequence with HVMux mapping 128 channels onto 192
elements, so the element set per (scanline, focal zone) is a moving sub-aperture.
`zea/data/convert/verasonics.py` handles `TX.Delay` / `Apod` / `focus` / `Origin` /
`Steer` and plane-wave detection, which replaces most of the repo's
`utils/load_hdf5_data.py` — but **I found no explicit HVMux / receive-aperture mux
handling in it**. The repo's own cyclic channel→element remap
(`build_mae_dataset.py:106-125`, hard-won and correct) may therefore still be needed
on top. Verify early in WI-16; if zea does handle it, that code stays deleted.
Angular sectors remain well-defined either way, but the occupancy map becomes
scanline-dependent — 3-D `n[z,x,k]` becomes 4-D.

### 3.7 Store format

Zarr v2 (matching the repo's existing pin), **two arrays, one chunk per frame each**:

| Array | Shape | Chunk | Size |
|---|---|---|---|
| `fsa` | `(N, 2, Z, X, 16)` float32 | one frame | 29.6 MB/frame → 27.3 GB |
| `pw` | `(N, 5, 2, Z, X, 4)` float32 | one frame | 37.0 MB/frame → 34.1 GB |

1846 chunk files total, well under HF's 10k-files-per-directory limit. The repo
already learned this lesson the hard way (`build_mae_dataset.py:48-57` documents the
chunk-count blowup).

**Two arrays rather than one on purpose:** Policy B and every K-ablation cell never
touch the PW stack, so keeping it in a separate array halves their I/O (§5.4). It
also lets the PW stack be dropped or rebuilt independently, which is cut candidate #1
(§5.3).

Frame-invariant geometry — `sector_id[z,x,e]` int8 (41.6 MB) and occupancy
`n[z,x,k]` uint16 (7.4 MB) — is stored **once**, not per frame. Labels go in a
sidecar `manifest.json` plus a small zarr group for `segmentation` and `sos`.

Rejected: parquet with 29.6 MB rows — NVIDIA's own choice, and the reason their
shards are 5.4 GB and random access is impossible.

---

## 4. Evaluation design (built first, validated on random features)

### 4.1 Label axes, probe types, and the two measured caveats

**Probes are regression-first**: ridge with R² and MAE for every continuous axis;
logistic (balanced accuracy + AUC) only for `contains_cyst` and `t0_group`.

**Protocol, fixed for every frame-level cell in the matrix:**

- **Ridge penalty selected by inner cross-validation**, never a fixed λ. A fixed penalty silently rewards whichever feature space happens to suit it.
- **k-fold CV over the 830 train frames**, *not* a single fit evaluated on the 93-frame validation split. 93 frames gives an R² standard error wide enough to swallow most effects we care about; the validation split is reserved as a final untouched check, reported once.
- **Bootstrap confidence intervals on every frame-level cell.** With n = 923 effective (§6.12) these are the only honest error bars, and they must appear in the table, not in a footnote.
- **PCA-to-64 dimensions as a fixed capacity control**, applied identically to every feature space. Without it, a 2048-d representation and a 512-d one are not being compared on equal terms, and the wider one wins on ridge by capacity alone.
- Folds grouped by `frame_id` throughout (§4.2).

| Axis | Level | Probe | Source | Policy A | Policy B |
|---|---|---|---|---|---|
| `contains_cyst` (>1 %) | crop | logistic | `segmentation_map` crop | ↑ | ↓ |
| `cyst_fraction` | crop | ridge | `segmentation_map` crop | ↑ | ↓ |
| `sos_mean` | **frame** | ridge | `sound_speed_map` | ↑ | ↓ |
| `sos_std` | **frame** | ridge | `sound_speed_map` | ↑ | ↓ |
| `sos_mean` | crop | ridge | resampled field | reported, **caveated** — see below |
| `phase_error` | frame | ridge | `phase_error` | ↑ | ↓ |
| `z_centre` | crop | ridge | crop coords | ↓ (destroyed) | ↑ (preserved) |
| `x_centre` | crop | ridge | crop coords | ↓ | ↑ |
| `t0_group` | frame | logistic | `t0` | ↓ | ↑ (see §6.4) |
| `split` | frame | logistic | shard name | chance (negative control) | chance |

**⚠ MEASURED: the crop-level sound-speed probe is a frame-level probe in disguise.**
The review asked for this check before trusting any sound-speed number. I ran it on
all 923 frames (143 988 crops, 8 × 9 SoS cells per 128 × 128 crop — the review's
"roughly 8×8" is right):

| Quantity | Value |
|---|---:|
| Between-frame std of frame-mean SoS | **46.93 m/s** |
| Within-frame, between-crop std of crop-mean SoS | **6.32 m/s** |
| Mean within-crop std of SoS | **7.58 m/s** |
| Within-frame share of total crop-mean variation | 13.3 % of std → **1.8 % of variance** |

Two conclusions:

1. **98.2 % of the variance in crop-mean sound speed is between frames.** A crop-level SoS probe has an effective sample size of **923, not 143 988**. Crop-level error bars would overstate confidence by more than an order of magnitude. ⇒ **`sos_mean` and `sos_std` are frame-level targets**, probed on frame-mean-pooled crop features. The crop-level version is reported only as a diagnostic.
2. The spatially-resolved part of the target is **buried in its own blur**: within-crop SoS spread (7.58 m/s) *exceeds* the between-crop signal within a frame (6.32 m/s), an SNR of 0.83. Spatially-resolved SoS probing at 128 × 128 is not viable on this data at this crop size. A larger crop would not help (less spatial resolution); a smaller one makes it worse.

**The review's specific fear is disconfirmed, however.** corr(crop `sos_mean`,
crop depth) = **−0.023 pooled, −0.177 within-frame**. The SoS field is *not*
depth-organised, so the SoS probe is **not** covertly measuring depth position. The
real risk is different and needs a different control: because SoS is near-constant
within a frame, *any* frame-level cue (global gain, mean speckle brightness,
aberration blur) suffices. ⇒ **Control:** report, in the same row, the R² of a
2-feature probe using only the crop envelope's mean and std. If the encoder does not
clear that, it has learned nothing about sound speed.

**⚠ MEASURED: partial correlations for `phase_error`.** Per review:

| | value |
|---|---:|
| corr(`phase_error`, `sos_std`) | +0.756 |
| corr(`phase_error`, `sos_mean`) | −0.383 |
| corr(`sos_std`, `sos_mean`) | −0.361 |
| **partial** corr(`phase_error`, `sos_std` \| `sos_mean`) | **+0.717** |
| **partial** corr(`phase_error`, `sos_mean` \| `sos_std`) | **−0.180** |
| **R²(`phase_error` ~ `sos_mean` + `sos_std`)** | **0.585** |

The `phase_error`↔`sos_std` link is **direct, not mediated** by `sos_mean` (0.756 →
0.717 under partialling). Conversely most of `phase_error`'s relation to `sos_mean`
*is* mediated (−0.383 → −0.180). And **58.5 % of `phase_error` is a linear function
of the two SoS summary statistics**, so only 41.5 % is independent information.

⇒ **Reporting rule:** the `phase_error` row carries R² = 0.585 as a printed
**oracle-baseline column** — that is what a probe achieves from ground-truth SoS
statistics alone. A `phase_error` R² at or below 0.585 is not evidence the encoder
represents aberration; it is consistent with representing sound speed only.

### 4.2 Leakage

**⚠ There is no acquisition ID to split on** (§6.1). The assertion the brief asks
for is still built, with the key redefined:

- The unit is the **frame** (`frame_id = "<split>/<shard>/<row>"`, assigned at build time and written into the manifest). Every simulation contributes exactly one frame (`frame_idx == 0` for all 923 rows).
- All crops from a frame share that frame's `frame_id`.
- `assert_no_frame_leakage(splits)` **hard-fails** if any `frame_id` appears on both sides of any split, and is called unconditionally in the probe harness, not just in tests.
- The realistic leakage mode here is **patch-level**: overlapping crops from one frame landing on both sides of a probe's train/test split. That is the assertion's actual job.
- Empirical check performed (§4.6): no duplicate or near-duplicate phantoms across NVIDIA's own train/validation boundary.

### 4.3 Confounding

The brief says "the first UMAP must be coloured by acquisition ID". There is no
acquisition ID. The equivalent — and, I think, the more dangerous — confound here is
**crop depth**: attenuation, diffraction spreading, and per-depth aperture coverage
all vary strongly with z, and will dominate patch statistics.

**Decision: the plotting code's default colour is `z_centre`, second `sos_mean`
(frame), third `phase_error`, and only fourth `cyst_fraction`.** The first figure
produced by any run is depth-coloured. `regress_out` (ported from
`eval/latent.py:250-274`) is run on `z_centre` and `x_centre` and the residual
embedding is plotted alongside, with the explained-variance fraction printed — that
tells us how much of the latent is pure geometry before we claim anything about
tissue.

### 4.4 Methods and baselines

Probes: ridge regression (continuous axes, report R²) and logistic regression
(categorical, report balanced accuracy + AUC), on **frozen** features, 5-fold
grouped by `frame_id`.

Clustering: k-means (sweep k), DBSCAN (sweep eps on the k-NN-distance elbow).
Metrics: ARI / NMI against `contains_cyst` and against `t0_group`, plus silhouette.

Visualisation: UMAP and t-SNE, both coloured per §4.3.

Baselines in the same table, identical crops, identical probe protocol:
1. **Random-init ResNet-18**, identical architecture and stem. Also plotted as a visible floor on every training chart (§4.7).
2. **PCA on raw crops** (flattened `(2,128,128,16)`), matched output dimension.
3. **Handcrafted speckle statistics**: Nakagami m and Ω fitted on the envelope, envelope SNR (μ/σ), skewness, kurtosis, and GLCM texture features (contrast, homogeneity, energy, correlation) at 4 offsets — computed per angular sector and pooled.
4. **Trivial**: mean and std of the crop envelope, 2 features. If a learned encoder does not beat this on a given axis, say so. Doubles as the §4.1 frame-level-cue control for the sound-speed probes.
5. **Frozen DINOv3** — natural-image control. Log-compressed B-mode, single channel replicated to 3. ⚠ **`facebook/dinov3-*` is `gated: manual`** on HF — submit the request in WI-1, not when needed. Ungated fallback `facebook/dinov2-small`; report which was used.
6. **Frozen zea HVAE** (`hf://zeahub/hvae`) — **ultrasound-domain control**, trained on EchoNet-LVH cardiac B-mode at 256×256. Encoder features.
7. **Frozen zea Speckle2Self** (`hf://zeahub/speckle2self-invivo`) — **ultrasound-domain, speckle-aware** control, trained on in-vivo ultrasound.
8. **Frozen TAESD** (`hf://zeahub/taesdxl_encoder`) — ⚠ **a second natural-image control, NOT an ultrasound model** (§0.5): zea's `taesdxl` is the stock Stable Diffusion XL tiny autoencoder re-implemented in Keras. Useful, but do not report it as domain-matched.
9. **Oracle**: for `phase_error` only, the R² = 0.585 achievable from ground-truth SoS statistics (§4.1).

Baselines 5–8 split into **natural-image** (DINOv3, TAESD) and **ultrasound-domain**
(HVAE, Speckle2Self). None can consume 2-channel complex IQ or a sector stack — they
see only log-compressed B-mode. All run on the K = 1 log-envelope stage, so they need
no extra data.

**⚠ Read these as a CEILING, not as the answer to "B-mode vs sub-apertures".** The
temptation is to conclude from a weak HVAE result that sub-apertures beat B-mode.
That inference is unsound: **HVAE is cardiac (EchoNet-LVH) and Speckle2Self is
in-vivo**, while our data is a simulated cyst phantom. A poor result from either may
reflect **anatomy and domain mismatch** — cardiac chambers vs anechoic cysts,
in-vivo speckle statistics vs k-Wave speckle — rather than anything about the signal
stage. The confound is not removable by any amount of careful probing, because these
models cannot be retrained on our data without ceasing to be external baselines.

**The clean control for that question is internal: our own VICReg trained on B-mode
versus the same VICReg trained on sub-apertures, in the signal-stage ablation
(§4.5).** Identical architecture, identical corpus, identical augmentation policy,
identical probe protocol — the only difference is the input stage. That comparison
answers "what is the beamformed sub-aperture domain buying?", and it is the number to
cite.

**Reporting rule.** External frozen models are presented as *ceilings* — "a frozen
general-purpose / ultrasound-domain model reaches X on this axis" — and every claim
about signal stage cites the internal B-mode-vs-sub-aperture row instead. The two are
kept in visually separate blocks of the results table so they are not read as a
single ranking.

⚠ HVAE and TAESD are Keras models (TF/JAX backends). Run them in the **zea
virtualenv** (§0.6) as a feature-extraction pass writing features to disk, then probe
them in the PyTorch env alongside everything else. No runtime coupling.

### 4.4b ⚠ DBUA / SoS-corrected beamforming is a LABEL LEAK — enforced, not documented

§0.1 lists `calculate_delays_heterogeneous_medium` as a free win: zea can beamform
using the ground-truth sound-speed map. It is a win, and it is also a **direct label
leak**, because that beamformer *consumes the label it would then be probed for*.

**Data beamformed with the ground-truth SoS map cannot be used to probe sound speed
or phase error.** The SoS field is an input to the delay calculation, so it is
imprinted in the output; a probe recovering it is measuring the beamformer, not the
representation. The same applies to `phase_error`, which is 58.5 % a linear function
of the SoS statistics (§4.1) — so an SoS leak is substantially a phase-error leak too.

**Two strictly separated beamforming stages, enforced in code:**

| Stage | Delays from | Valid probe targets | Invalid |
|---|---|---|---|
| **`c0_const`** | constant `c0 = 1540 m/s` | **all** — `sos_*`, `phase_error`, cyst, geometry | — |
| **`gt_sos`** | ground-truth `sound_speed_map` | cyst / tissue axes; **oracle ceiling only** for physics | **`sos_mean`, `sos_std`, `phase_error`** |

`c0_const` is the primary training and evaluation stage. `gt_sos` is an *oracle*: it
answers "how much better would the cyst probe be if aberration were perfectly
corrected?", which is a genuine and useful upper bound — and nothing else.

**Enforcement is a hard failure in the eval harness, not a note in the docs.** Every
feature array carries a `beamform_stage` tag; every probe target carries a
`leaks_from` set; `assert_no_label_leak(stage, target)` raises before any fit. A test
constructs the `gt_sos` × `sos_mean` pair and expects the raise. This sits alongside
`assert_no_frame_leakage` in the same guard path (§4.2) — same class of error, same
treatment.

**⚠ `bmode_variant_2` is contaminated if it resolves to DBUA.** NVIDIA calls it
"focused transmit"; OpenH-RF calls it "DBUA aberration-corrected" (§0.2, §6.19). If
OpenH-RF is right, it is a `gt_sos`-stage product and **must not be used as a signal
stage for the physics probes** — which is currently how §4.5 lists it. WI-0 resolves
this, and until it does, `bmode_variant_2` is tagged `gt_sos` (the conservative
assumption) so the guard blocks it by default.

**A bonus target falls out of the same structure.** The difference between the two
beamformed versions of one frame — `gt_sos` minus `c0_const`, in log-envelope or
complex — is close to a **pure aberration signal**: same scatterers, same transmits,
same receive aperture, differing only in the delay model. That makes it a candidate
regression target in its own right (predict the correction map from the uncorrected
data), and a candidate supervision signal for the cross-aperture arm. Noted as an
opportunity, scoped below WI-15 in priority, and subject to the same guard — it is
derived from the label, so it can be a *target* but never an *input*.

### 4.5 Results matrix

`(signal stage) × (augmentation policy) × (label axis) × (method)`, evaluated on
**both corpora** — NVIDIA (physics labels) and Stanford (real tissue, transfer).

Signal stages, each tagged `c0_const` or `gt_sos` (§4.4b):

| Stage | Beamform tag | Notes |
|---|---|---|
| complex IQ, K ∈ {16, 8, 4, 2, 1} | `c0_const` | primary |
| PW(5) × K=4 stack + per-angle marginal | `c0_const` | conditional on §4.5a(b) |
| envelope, log-compressed envelope | `c0_const` | |
| NVIDIA `bmode` | `c0_const` | free, no beamforming needed |
| NVIDIA `bmode_variant_2` | **`gt_sos` (conservative)** | §4.4b — blocked for physics probes until WI-0 resolves it |
| our SoS-corrected reconstruction | `gt_sos` | oracle ceiling only |

The `bmode` row is free and gives the "across signal stages" axis the project
description asks for without any beamforming — and it is the stage all four frozen
pretrained baselines run on, so they meet our models in the same column.

### 4.5a ⚠ Reciprocity: consequences of a symmetric Tx–Rx matrix

The §0.2 observation that the Tx–Rx matrix is near-symmetric by acoustic reciprocity
was raised there only to explain why the axis-order bug is invisible. It has three
further consequences, and they cut across the design. `‖S − Sᵀ‖/‖S‖` is measured in
WI-0 and reported in §10.

**(a) Transmit-side cross-aperture masking is shortcut-solvable — mitigate
unconditionally.** If `S[i,j,:] ≈ S[j,i,:]`, then a held-out transmit *i* is
recoverable by *transposing the receive data*: the network can read off
`S[:,i,:]` and copy it. That is a pure indexing shortcut with no physics in it, and
it would show up as a suspiciously easy loss curve and a useless representation —
the §4.7 "loss ↓, probe flat, RankMe healthy" signature.

**Mitigation, applied regardless of what the measurement says, because it costs
nothing: never expose both aperture axes in the same tensor.** Sum fully over one
axis, keep sectors on the other. Concretely:

- Transmit-masked arm: input is `(2, Z, X)` per retained transmit angle — **receive fully summed**. The transposition shortcut has nothing to index into.
- Receive-masked arm: input is `(2, Z, X, K)` receive sectors with **transmit fully compounded** — which is already what the primary store is (§3.2).

This is a constraint on the *dataloader*, not on the store, and WI-15 asserts it.

**(b) The PW stack loses further ground, and is re-ranked.** If transmit and receive
diversity are near-redundant, then a transmit-angle positive pair and a
receive-sector positive pair are close to the same augmentation. Combined with the
already-known asymmetry in angular leverage — **±10° transmit (grating-lobe capped)
versus ±26.6° receive** (§3.2b, §3.2) — the transmit axis is both *narrower* and
*partly redundant*.

⇒ **Re-ranked: the PW stack moves from cut candidate #1 to cut candidate #1 with a
lower bar for cutting.** Rev-2 justified it as "the physically exact positive-pair
generator", which is still true but is worth much less if receive sectors generate
nearly the same pairs over a wider angular span. The decision rule for WI-4b becomes:
**build the PW stack only if WI-0's reciprocity residual is materially non-zero.**
If it is near zero, build the FSA × K=16 store alone (27.3 GB instead of 61.5 GB) and
treat transmit diversity as an explicitly-dropped axis with this measurement as the
reason. See §10 for the resolution.

**(c) ⚠ Reciprocity will NOT hold on Stanford — new named transfer failure mode.**
NVIDIA is a k-Wave simulation with identical modelled transducers, so its Tx–Rx
matrix is symmetric essentially by construction. Stanford is real hardware:

- separate transmit and receive electronics (different amplifier chains, different noise);
- element-to-element sensitivity and impulse-response variation;
- HVMux receive-aperture switching that has no transmit-side counterpart;
- and, once `Refocus` is applied (§0.4), **decoding residuals** — REFoCUS inverts an ill-conditioned encoding matrix, and its error is not symmetric in tx/rx.

**Any representation that leans on simulation symmetry will degrade on Stanford, and
sim-only evaluation cannot see it happening.** That is the dangerous part: the
failure is invisible until transfer, which is exactly when it is most expensive.

Added to the transfer failure-mode list, alongside depth shift (§4.5c) and grid
mismatch (§3.6). Concrete instrument: **measure `‖S − Sᵀ‖/‖S‖` on Stanford's
post-`Refocus` multistatic data too, and report the two side by side.** The gap
between them upper-bounds how much sim-symmetry the encoder could have absorbed.
Mitigation (a) also protects here — it is the reason to apply it unconditionally.

### 4.5b Cross-aperture masking: transmit or receive?

The brief and the first draft both assumed **receive**. It is a design choice with
reasoning, not an assumption, and on this dataset **transmit is better posed.**

First, a correction to the shared premise: the brief's justification — "all
sub-apertures observe the same scatterers, so the speckle realization is shared" — is
not strictly true on either axis. Speckle is the coherent sum of sub-resolution
scatterers weighted by the PSF, and changing *either* the transmit or the receive
aperture changes the PSF and hence the speckle realization. That is precisely why
coherent compounding reduces speckle. What is genuinely shared is the **scatterer
field**, not its realization.

| | Receive-sector masking | Transmit-angle masking |
|---|---|---|
| What varies | receive PSF, receive-path coherence | insonification angle, transmit-path coherence |
| Why predictable | van Cittert–Zernike: spatial coherence across the receive aperture for a diffuse field is a known smooth function | geometry: a ±10° view change is a small, near-affine warp of the same scatterer field |
| What the residual carries | receive-path aberration | **aberration and sound-speed error** |
| Ground truth available? | indirectly | **yes — `phase_error` and `sound_speed_map`** |
| Free strong baseline? | **yes** — the closed-form vCZ coherence function | no closed form |
| Extra data needed? | none | the PW stack (§3.2b) |

**Decision (revised in rev-4): `--mask-axis rx` is now the default; `tx` only if
WI-0 shows a materially non-zero reciprocity residual.** Rev-3 defaulted to `tx` on
the strength of its ground-truth row. §4.5a(a) undercuts that: on a symmetric matrix
the transmit-masked task is trivially solvable by transposition, so it would measure
indexing rather than physics. The receive variant additionally carries the
closed-form van Cittert–Zernike baseline, which the transmit axis cannot offer.

**Either way, the §4.5a(a) mitigation is mandatory and unconditional:** the masked
arm never sees both aperture axes in one tensor. `tx`-masked inputs have receive
fully summed; `rx`-masked inputs have transmit fully compounded.

Caveat carried from §3.2b: ±10° of usable steering is modest, so transmit views are
*very* similar. That makes the prediction well-posed but also possibly too easy;
WI-15 reports the trivial-copy baseline (predict the held-out angle as the mean of
the others) first.

### 4.5c Depth-agnosticism — four measures, in priority order

Goal: **one encoder usable on both corpora without a depth restriction.**

Framing correction worth stating up front: the encoder *already* accepts any crop
from any depth — ResNet-18 + global average pooling is fully convolutional, so there
is no architectural barrier. What is missing is **comparability** across depth,
because depth is *inferable from content*: attenuation, lateral resolution, speckle
grain size, and sector occupancy all vary with z. Nothing below is about making the
model *run* at new depths; it is about making its features mean the same thing there.

**1. Depth-stratified crop sampling.** Deep regions contain more pixels, so uniform
random sampling over image area over-represents them. **Sample uniformly in depth**:
bin z into equal-width bands and draw equally from each. Cheap, no downside, and it
is a precondition for every measurement below being interpretable.

**2. Analytic gain compensation — the highest-value single change.** Apply a
**fixed, known, invertible** depth gain to both datasets: nominal exponential
attenuation at an assumed α ≈ 0.5 dB/cm/MHz plus geometric spreading.

**This is categorically not the per-depth RMS normalization banned in §1.3 / WI-7.**
That one is *data-driven* — it measures the actual per-depth RMS and divides it out,
which destroys attenuation by construction. This one is a *fixed analytic curve* that
removes the **gross, predictable** depth trend while preserving **deviations from
nominal** — and the deviations are exactly where the tissue signal lives. It is
invertible, so nothing is lost; it is a change of basis, not a normalization.

**zea implements it**: `zea.func.ultrasound.make_tgc_curve(n_ax, attenuation_coef,
sampling_frequency, center_frequency, sound_speed)`, computing
`attenuation_db = 2·α·dist_cm·f_MHz` and gain `10^(dB/20)`. `attenuation_coef` is
already a first-class field on `zea.Parameters`.

⚠ **"Identically to both datasets" means physically identical, not numerically
identical.** Attenuation is frequency-dependent, so the curve must be evaluated with
**each dataset's own `fc`** — 6.5 MHz for NVIDIA, 7.813 MHz for Stanford. Using one
numeric curve for both would leave a residual 20 % depth trend that is pure domain
gap. This is the single most likely way to implement this measure wrongly.

Applied at **load time**, not build time, so it stays a switchable ablation and the
store keeps raw amplitudes.

**3. Depth conditioning, not depth invariance.** Feed normalized depth in
**explicitly** — a FiLM layer on the backbone, or an extra constant input channel
carrying z in mm. This lets the network *compensate using* depth rather than spending
latent capacity *encoding* it. Depth in mm is probe-independent, so it transfers
directly. **One experiment**, on top of measure 2.

**4. Wavelength-based resampling** — already planned as the §3.6 grid ablation
(λ-matched vs mm-matched). Runs in the same table.

**⚠ Explicitly NOT pursued: depth invariance via augmentation** (blur, TGC
randomization, resolution jitter). It would work, and that is the problem — it would
**delete depth from the latent**, and depth is a probe axis we deliberately kept
(§4.1) and which we now know is *not* confounded with sound speed
(corr = −0.023, §6.12). Available under **Policy A ablation only, never default.**

**The depth-restricted control row is kept regardless**, at every K. It is the
instrument that tells us whether measures 1–4 worked: **if unrestricted transfer
matches depth-restricted transfer, we have demonstrated depth-agnosticism rather
than assumed it.** Without that row the other four measures are unfalsifiable.

### 4.6 Bootstrapping the harness before any beamforming exists

**This is the key de-risking move.** `bmode` is a 507×456 image on the same grid as
`segmentation_map`, and it is already in the dataset. So the entire eval harness —
crop sampling, label attachment, leakage assertion, probes, clustering, UMAP,
baselines, results matrix — can be built and validated end-to-end on `bmode` crops
with a random-init encoder, **before line one of the beamformer is written.** When
the K=16 store appears, only the reader changes.

That satisfies the brief's "validate it end-to-end on randomly-initialized encoder
features" requirement literally, and it removes the beamformer from the eval
harness's critical path entirely.

### 4.7 Training-time instrumentation — built before the first real run

**VICReg loss value alone does not indicate success.** It can fall while the
representation collapses into a low-dimensional subspace. All of the following ship
with the harness in WI-12, before any run is launched — not added after a run looks
wrong.

**Tier 1 — every epoch (cheap, no labels or crop-level only):**

| Instrument | Detail | Cost |
|---|---|---|
| **Three VICReg terms logged separately** | invariance, variance, covariance — **never log only the sum**. Plus the variance term's hinge margin. | free |
| **Per-dimension std of projector output** | 2048-d, logged as a histogram | free |
| **Singular-value spectrum of the BACKBONE feature covariance** | 512-d, **pre-projector, not the projector** | one 512×512 SVD/epoch |
| **RankMe** | `exp(−Σ p_k log p_k)`, `p_k = σ_k / Σσ_j + ε`, over that **backbone** spectrum. **Primary collapse diagnostic; needs no labels.** | free given the SVD |
| **Online linear probe — cyst/background** | crop-level, balanced accuracy, on **detached** features | small |
| **kNN evaluation** | held-out subset, no training required | small |
| **Random-init floor** | identical architecture, plotted as a visible line **on the same axes** as every chart above | one-off |

**Tier 2 — every N epochs (needs crop→frame aggregation):**

| Instrument | Detail |
|---|---|
| **Frame-level sound-speed regression** | ridge R², crops mean-pooled to frames (§6.12 — SoS is a frame-level target) |
| **Frame-level phase-error regression** | ridge R², same aggregation, plotted against the **0.585 oracle line** (§4.1) |

Backbone (512-d) and projector (2048-d) are kept deliberately distinct: **collapse
appears in the backbone spectrum well before the projector's variance hinge reacts**,
so RankMe is computed on the backbone and the per-dimension histogram on the
projector. Reporting RankMe on the projector would be the standard mistake here.

**Stop-gradient on the online probe is a correctness requirement, not a detail.**
The probe head trains on `features.detach()`; a test asserts that a probe-loss
backward pass leaves every encoder parameter's `.grad` unchanged. Without it the
"linear probe" silently becomes supervised fine-tuning and every frozen-feature
number in the results matrix is invalid.

Backbone is ResNet-18, so the spectrum is 512-d; the projector output (2048-d) gets
the per-dimension histogram, and the two are kept distinct on purpose — collapse
shows up in the backbone spectrum well before the projector's variance hinge reacts.

**Failure signatures — to be written verbatim into `train/README.md`:**

| Symptom | Diagnosis | Action |
|---|---|---|
| loss ↓ **and** RankMe ↓ | **dimensional collapse** | raise covariance weight; check projector width; check augmentation strength |
| loss ↓, probe flat, RankMe healthy | **augmentation shortcut** — the model solved the pretext task without learning content | inspect which augmentation makes views trivially matchable |
| probe **degrading** over training | augmentations are destroying the probed property | **see below — this is not automatically a failure** |

**The third row is the experiment, not a bug.** For **Policy A**, degradation of the
*physics* probes (`z_centre`, `x_centre`, `t0_group`) over training is the
**expected and desired** outcome — Policy A randomizes acquisition factors precisely
to make the encoder invariant to them, and whatever the encoder is invariant to is
deleted from the latent and can no longer be probed. The instrumentation therefore
tracks **probe trajectories per label axis**, never a single aggregate, and the
harness must not flag per-axis degradation as a failure. The Policy A ↓ geometry /
↑ tissue versus Policy B ↑ geometry / ↓ tissue contrast in §4.1 is the deliverable;
seeing it *develop over training* is the strongest form of that evidence.

---

## 5. Branch structure, module layout, and work items

### 5.1 Module layout

```
src/ssl_us/                       # renamed: the old package name is MAE-specific
  beamform/            <-- ZEA VENV (KERAS_BACKEND=jax), offline only
    nv_parquet.py       stream NVIDIA shards -> zea File (from OpenH-RF convert.py)
    stanford_mat.py     zea verasonics reader + Refocus -> multistatic (WI-16)
    sectors.py          sector_id + occupancy maps (all that survives of geometry.py)
    sector_sum.py       AngularSectorSum: zea Operation replacing DelayAndSum
    planewave.py        synthesize PW(5) along the tx axis, beamform per angle
    build.py            offline CLI: shards -> store, resumable
  data/                <-- PYTORCH VENV
    store.py            read (N,2,Z,X,16) + PW stack + labels; HF chunk-selective download
    transforms.py       raw ladder 16->1, occupancy normalization, analytic TGC, signal stages
    crops.py            depth-stratified crops (uniform in z), label attachment
    augment.py          Policy A / Policy B
    splits.py           split construction + assert_no_frame_leakage
  models/
    resnet.py           ResNet-18, stem adapted to in_channels
    vicreg.py           3-layer MLP projector + inv/var/cov loss
    mae2d.py            spatial MAE baseline arm
    xaperture.py        cross-aperture prediction arm, --mask-axis {tx,rx}
  train/
    common.py           device/AMP, LR schedule, loop scaffolding (ported)
    instrument.py       RankMe, SV spectrum, detached probe, kNN, random-init floor
    README.md           failure-signature table (S 4.7), verbatim
    vicreg.py  mae.py  xaperture.py
  eval/
    features.py  probes.py  cluster.py  viz.py  baselines.py  matrix.py
    pretrained.py       frozen baselines: DINOv3/v2 (torch) + HVAE/Speckle2Self/TAESD
                        (extracted in the zea venv, probed here)
  utils/
    monitoring.py       ported from the old repo, snapshot hook rewritten
tests/                  pytest, incl. a CI-fast tier on the smoke subset
configs/                YAML per arm
scripts/                publish_hf.py, run_matrix.py
```

### 5.2 Ordered work items

Each is independently reviewable. **Done-condition** is the merge gate.

---

**WI-0 — Read one NVIDIA frame end-to-end and assert it. `[no deps]`**

Start from `open-h/OpenH-RF/examples/nv-raw2insights-us/convert.py` (Apache-2.0,
§0.2) — it already streams one sample and maps it to the zea format, so this is a
verification item, not a writing item.

Assert against the card: IQ shape `(180,180,1024)` float32 both parts; `elpos` pitch
201.0 µm ± 0.1 µm; `fs`/`fc`/`fd`/`c0` exact; `t0` ∈ the 3 measured values; `bmode`
and `segmentation_map` `(507,456)`; `segmentation_map` values ⊆ {0,1}; extents equal
the measured constants (in particular **z spans 0.0–40.0 mm**, §3.2c).

**Three ambiguities this item exists to resolve, all from §0.2:**

1. **⚠ IQ axis order.** NVIDIA's card says `[n_tx, n_el, n_ax]`; OpenH-RF's converter's transpose implies `[n_el, n_tx, n_ax]`. **Undetectable by shape (`n_tx == n_el == 180`) and nearly undetectable by image (reciprocity).** Test: for transmit *i*, the earliest-arriving energy on receive element *i* is the shallowest of any receive element — assert onset-sample is minimised on the diagonal, and that the asymmetry `‖S − Sᵀ‖/‖S‖` is small but the *onset* structure is not. **This must land before any transmit-axis operation** (PW synthesis WI-4b, transmit masking WI-15).
2. **⚠ What is `bmode_focused`?** NVIDIA says "focused transmit"; OpenH-RF says "DBUA aberration-corrected". Resolve by comparing it against zea's own SoS-corrected reconstruction (`reconstruct.py` does exactly this). Until resolved, label it `bmode_variant_2`.
3. **Absolute SoS calibration** (§6.11, §7.1).

*Done when:* `pytest tests/test_nv_frame.py` passes against a real download; all
three ambiguities are resolved in writing in the PR; and the test prints measured
shape, dtype, per-field summary stats, wall-clock and peak RSS. **Nothing below WI-3
is planned in detail until this returns real numbers.**

---

**WI-1 — Cut the old tree. `[dep: none]`**

Delete `mae/`, `dataset/`, `eval/latent.py`, `train.py`, `utils/beamform.py`,
`utils/load_hdf5_data.py`, `scripts/dataset_scripts/*`, `configs/beamform_params.json`,
`src/dataset.py`. Port §1.2's keepers into `src/ssl_us/` with comments intact.
Rewrite `pyproject.toml` (new package name; **unpin torch from `>=2.4.1,<2.5`+cu118**
— that pin exists for Pascal sm_61 and blocks the 4090's preferred builds; target
torch ≥ 2.6 + cu124). Add `pytest`, `ruff`, `umap-learn`, `pyarrow`, `datasets`,
`timm`/`transformers` for the DINO baseline. Add a CI workflow running lint + the
smoke tier.

**Two virtualenvs (§0.6):** `.venv` (PyTorch, training + eval) and `.venv-zea`
(`zea==0.1.4` **pinned exactly**, `KERAS_BACKEND=jax`, offline beamforming + Keras
feature extraction). The on-disk store is the only interface; nothing imports across.
Wrap `zea.File.create` behind one adapter (§0.7 — its signature is known-unstable).

**Submit the `facebook/dinov3-*` gated-access request now** (§6.14) — manual
approval, blocks nothing until WI-13, but only if requested early.

*Done when:* `poetry install` succeeds, `ruff check` is clean, `pytest` collects and
runs (WI-0's test only), and `git grep -iE '192|215|focal_zone|subject|MultiFocal'`
returns nothing outside `PLAN.md` and the docs.

---

**WI-2 — Eval harness on `bmode`, random encoder. `[dep: WI-0]`**

Per §4.6. Crop sampler, label attachment, `assert_no_frame_leakage`, ridge/logistic
probes grouped by `frame_id`, k-means + DBSCAN, UMAP/t-SNE with the depth-first
colour default, all four baselines, results-matrix writer. Encoder is random-init
ResNet-18 (stem = 1 channel for `bmode`).

*Done when:* `python -m ssl_us.eval.matrix --stage bmode --encoder random` emits a
complete results table over every axis in §4.1 and the figure set, on ≥ 100 frames;
**both guards are proven live by tests that expect a hard failure** —
`assert_no_frame_leakage` on a deliberately corrupted split (§4.2) and
`assert_no_label_leak` on the `gt_sos` × `sos_mean` pair (§4.4b); the full §4.4
protocol is in place (CV'd ridge penalty, k-fold over train frames, bootstrap CIs,
PCA-to-64 capacity control); and every baseline column is populated. Expected outcome
to record in the PR: `z_centre` probes well above chance even from random features (a
random CNN encodes position through boundary effects and amplitude), `cyst_fraction`
near chance. If `cyst_fraction` is *not* near chance from random features, the
harness is leaking and we stop.

---

**WI-3 — Sector + occupancy maps on top of zea's geometry. `[dep: WI-0]` — much smaller in rev-3**

Adopt `zea.Probe` / `zea.Parameters` / `zea.beamform.pixelgrid.cartesian_pixel_grid`
as the coordinate frame (§0.3). Our `geometry.py` reduces to the **angular sector
machinery only**: `sector_id[z,x,e]` int8 (41.6 MB) exploiting angle-monotonicity in
element index (§3.2), and the frame-invariant occupancy map `n[z,x,k]` uint16
(7.4 MB). All 180 elements, full 0–40 mm depth (§3.2c).

Deleted from rev-2 scope, now zea's: delay computation, the `(τ−t0)·fs` mapping,
interpolation, the transmit/receive angular masks, grid construction.

*Done when:* **each sector's element set is asserted contiguous in index for every
pixel** (the property the fast path depends on); occupancy matches the closed form in
the §3.2 table to ±1 element; the occupancy-vs-depth curve is committed as a figure
for f# ∈ {1.5, 1.0, 0.75} and for K ∈ {16, 8, 4, 1}; and zea's delay round-trip is
verified once against a hand-computed value (we trust it, but we check it once).

---

**WI-4 — `AngularSectorSum` op + reference validation. `[dep: WI-3]` ⚠ highest risk, much smaller in rev-3**

Implement **one** `zea.ops.Operation` replacing `DelayAndSum` (§0.1): a segment-sum
over the element axis of `tof_correction`'s `(n_tx, n_pix, n_el, n_ch)` output into
16 sector bins, then the transmit sum. **Raw sums** (§3.2). Runs inside
`Beamform(PatchedGrid(TOFCorrection → AngularSectorSum) → ReshapeGrid)`, so pixel
chunking, delays, interpolation, phase rotation and f-number masking are all zea's.

Beamform **three** sector-span variants — unlimited receive aperture, f# = 1.5,
f# = 1.0 — since the span is an empirical question (§3.4).

Reuse `open-h/OpenH-RF/examples/nv-raw2insights-us/reconstruct.py` (Apache-2.0) as
the reference-reconstruction harness rather than writing one.

*Done when:* — **REVISED, rev-5. The original pixel-wise gate was withdrawn as
unachievable; see "Why the original WI-4 gate was withdrawn" below before
changing this.**

On ≥ 5 frames spanning all three `t0` groups, for at least one of the three
variants:

- **gCNR agreement:** our K = 1 image's generalised contrast-to-noise ratio on
  the shipped `segmentation_map` (inclusion vs a depth-matched background) is
  within **0.10 of the reference `bmode`'s own gCNR on the same frame**. Graded
  against the reference per-frame, not against an absolute number, because
  frame difficulty varies enormously here (measured reference gCNR spans 0.15
  to 0.86).
- **Speckle statistics:** per-patch speckle SNR in a lesion-free background
  band lies in **[1.60, 2.15]** (Rayleigh ideal 1.913) *and* within **0.25 of
  the reference's own** speckle SNR measured on its native grid.
- **Negative control still fails**, unchanged in spirit and now a decisive
  contrast rather than a near-tie: beamforming with the `exp(j2πf_dτ)` rotation
  removed must measurably degrade gCNR. Applied only on frames whose reference
  gCNR ≥ 0.30, since below that no one resolves the inclusion and the
  comparison is noise. That negative control is still the point of this work
  item.

The winning variant sets `θ_max` for the build; the side-by-side + difference +
negative-control figure is committed per frame, now with the gCNR regions drawn
on it. Frames whose segmentation map is empty (they exist) run the speckle
checks only.

**Why the original WI-4 gate was withdrawn.** It required pixel-wise NCC ≥ 0.98
and median |dB| ≤ 1.5 against `bmode`. That is not reachable by any beamformer,
including NVIDIA's own, and the evidence is measured rather than argued:

1. **The reference disagrees with itself.** Scoring NVIDIA's two shipped
   reconstructions of the *same frame* against each other with that very
   metric gives mean NCC **0.197** over 10 frames (range −0.071 to 0.510 — on
   frame 1 they are *anti-correlated*). Our reconstruction scores ~0.26 against
   `bmode`, i.e. it already matches the reference better than the reference's
   own sibling does. Numbers in `tests/data/wi4_reference_selfconsistency.json`.
2. **It is not our beamformer.** `zea`'s stock `DelayAndSum` — the one
   OpenH-RF's reference `reconstruct.py` uses — scores 0.2587 where our
   `AngularSectorSum` scores 0.2573 on the same frame with the same parameters.
   Two independent implementations agreeing to 0.001 while both sit far below
   the gate places the discrepancy in the reference, not the code under test.
3. **`bmode` really is a constant-c DAS**, as OpenH-RF's converter documents:
   re-running through zea's heterogeneous mode with the ground-truth
   sound-speed map makes the score *worse* (0.257 → 0.134), so the residual is
   not uncorrected aberration.

The cause is physical. Fully-developed speckle is a random interference
pattern, not a picture of scatterers; any sub-wavelength difference between two
reconstructions reshuffles which grains are bright while leaving every
macroscopic feature identical. Pixel-wise correlation therefore measures
agreement of *speckle realisation*, which a correct beamformer is under no
obligation to reproduce. (§10.4's "residual sub-wavelength timing offset"
hypothesis was chasing this artifact. Two genuine timing bugs were nonetheless
found and fixed by that work — `t_peak`, worth NCC 0.076 → 0.257, and the `t0`
sign convention, 0.078 → 0.257 — and a 16-way sweep over {`t_peak`, `t0` sign,
near-field mute, tx/rx axis order} confirms the current settings are the best
of all combinations, and that tx/rx order is immaterial as §10.2 predicted.)

Metrics live in `src/ssl_us/beamform/image_quality.py` (`test_image_quality.py`),
which documents each one and why pixel-wise correlation was unfit.

---

**WI-4b — Plane-wave synthesis. `[dep: WI-4]` — CONDITIONAL in rev-4**

**Gate: build this only if WI-0's reciprocity residual is materially non-zero
(§4.5a(b)).** If it is near zero, transmit and receive diversity are near-redundant,
the transmit axis is also the narrower one (±10° vs ±26.6°), and the PW stack costs
34 GB for an augmentation the receive sectors already supply. Skipping it drops the
store from 61.5 GB to 27.3 GB. See §10 for the resolution and the decision taken.

If built: synthesize A = 5 plane waves at {−10, −5, 0, +5, +10}° from the Tx–Rx
matrix, then beamform each to K = 4 angular sectors (§3.2b). Per-angle beamforming is
a single sum over rx, so this is ~1/180 the cost of an FSA beamform per angle.

*Done when:* the α = 0° synthesized plane wave's beamformed image is verified against
a directly-computed PW-0 reference; total added compute is measured and confirmed
≤ 5 % of the FSA cost; **and a grating-lobe check is committed** — beamform one frame
at α = 15° (past the measured 10.30° limit) and show the grating lobe is visibly
present, confirming the angle-set bound is real and not merely asserted.

---

**WI-5 — Ladder + normalization. `[dep: WI-4]`**

`transforms.py`: pairwise reduction 16 → 8 → 4 → 2 → 1 on **raw** sector sums, and
the separate occupancy-divide normalization at each K.

*Done when:* a test asserts, on a real frame, that reducing raw 16 → 1 by four
pairwise sums equals a direct single-sector beamform over the same angular span to
float32 tolerance (rel. error < 1e-5); that each intermediate K equals a direct
K-way beamform; and — the new one — that **normalize-then-reduce ≠ reduce-then-
normalize is handled correctly**, i.e. the runtime path reduces raw and normalizes
by the merged occupancy, and a test constructs the naive wrong order and shows it
diverges. Empty sectors (`n_k = 0`) round-trip as 0 without NaN. This is the
property the whole ablation axis rests on; it gets its own item so it cannot be
quietly broken.

---

**WI-6 — Build pipeline: deterministic, resumable, parallel. `[dep: WI-4, WI-5]`**

Iterate shards; for each, stream rows, beamform, write frame chunks, append to
manifest, `fsync`, record shard completion. Restart skips completed shards. All
source metadata carried through per frame, keyed by `frame_id`. Beamforming
parameters (grid, element selection, f-number, K, `t0` per frame, code git SHA)
recorded in the manifest.

*Done when:* the 32-frame smoke subset builds; a re-run is a no-op; a `SIGKILL`
mid-shard followed by a restart yields **byte-identical** output to an uninterrupted
run; the manifest round-trips every source field; and the build logs measured
per-frame wall-clock and GPU memory.

---

**WI-7 — Normalization. `[dep: WI-6]`**

Compute a **single dataset-level** complex-magnitude RMS over a sample of the train
split; store it in the manifest; apply as one scalar divide at load. Additionally
compute and store the per-depth RMS profile **as metadata only, never applied** —
that profile *is* the attenuation signal we intend to probe.

*Done when:* a test asserts that loading applies exactly one scalar; a second test
asserts the per-depth RMS of loaded data is **not** flat (i.e. depth-dependent
amplitude survived), with the measured decay printed. Both tests exist specifically
to prevent regressing to the old `normalize_frame` behaviour (§1.3).

---

**WI-8 — Full build. `[dep: WI-6, WI-7]` — ✅ DONE (rev-5)**

**Built: 923/923 frames at `/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr`**, 23 GB,
`(923, 2, 507, 456, 16)` float32 raw sector sums, f#=1.5 / K=16. Every done-when
clause below is met: manifest complete; WI-4's (rev-5) gate re-run on 10 frames read
back *from the store* passes 10/10 (`tests/data/wi8_store_verification.json`); the
corpus depth-RMS profile is plotted (`figures/wi8/depth_rms_profile.png`, 7.6 dB
decay 5–10 → 30–38 mm); and all 923 frames verified finite and non-empty. Code:
`beamform/{frame,store}.py`, `scripts/wi8_{build,verify,profile}.py`.

Two independent confirmations the store indexes correctly: exactly **75** frames have
an empty `segmentation_map`, matching §2's separately measured 75/923; and per-frame
RMS is trimodal, matching §2.3's three `t0` values (435/116/372).

*Original estimate, superseded:* 6–10 h wall, ~3–4 h GPU. **Actual: ~2 h wall**,
~3.25 s/frame GPU (~50 min total). The wall estimate was wrong in both directions —
see §5.4's rev-5 note: `datasets` streaming runs at 2.7 MB/s here where a plain shard
download runs at ~70 MB/s, so the build fetches one parquet shard at a time and
deletes it. See CLUSTER_HANDOFF.md for the three traps this shook out (blosc thread
deadlock; `hf_hub_download` returning a symlink so per-shard deletes free nothing;
caches defaulting onto a 32 GB overlay).

*Done when:* 923/923 frames present, manifest complete, WI-4's validation re-run on
10 random frames from the finished store still passes the gate, and the depth-RMS
profile is plotted for the whole corpus.

---

**WI-9 — HuggingFace publication. `[dep: WI-8]`**

Full set + a 32-frame `smoke` subset (~950 MB) in an identical layout for CI and
local iteration. Port the chunk-selective download from `dataset/dataset.py:148-183`.

Dataset card must contain: beamforming parameters in full; the sub-aperture
structure and worked code for recovering K = 8/4/2/1; the metadata schema; the
`frame_id` convention and why splits must use it; NVIDIA's four known issues
propagated verbatim; and — required by CC BY 4.0 — attribution to NVIDIA, the BibTeX
citation, a link to the licence, and **an explicit statement of the changes made**
(sub-aperture DAS beamforming to a Cartesian grid). Licence: CC BY 4.0, matching
upstream. Do **not** vendor NVIDIA's CC BY-NC model weights.

*Done when:* a clean-machine round-trip loads the smoke subset from the Hub and
reproduces a stored checksum, and the card renders with every section above.

---

**WI-10 — Splits + leakage assertion. `[dep: WI-8]` — ✅ DONE (rev-5)**

**Verified on the built store**, not just asserted in code
(`scripts/wi10_splits.py`, results `tests/data/wi10_splits.json`, figure
`figures/wi10/split_audit.png`): outer split 830 train / 93 validation = 923,
inner probe split 664/166 carved from train only, guard passes at both levels.

The distributional-matching claim below was measured during planning from the
*source* dataset; re-measured from the *built* store it reproduces closely —
`sos_mean` 1474.0±47.1 vs 1471.4±48.0 (Welch t = 0.60), `phase_error`
1.826±0.278 vs 1.827±0.278 (t = −0.02). Neither is detectably different, so
adopting NVIDIA's boundary wholesale stands. `cyst_fraction` is the least
matched axis (t = 1.86, under the |t| < 2 bar but the closest to it) and is
worth a glance if a cyst probe ever looks odd. Empty-`segmentation_map` frames
split 63 train / 12 validation, matching §2's independent count exactly.

Adopt NVIDIA's train/validation as the outer split (they are distributionally
matched — measured: `sos_mean` 1474.2 ± 47.0 vs 1471.1 ± 47.8; `phase_error`
1.826 ± 0.277 vs 1.827 ± 0.277), and carve an inner probe train/test from the train
split by `frame_id`.

*Done when:* `assert_no_frame_leakage` is called unconditionally in the probe path;
a test constructs a deliberately-leaking split and expects a hard failure; and a
split-audit figure (the old `figures/split_audit_*.png` idea, retargeted) is emitted
per run.

---

**WI-11 — Augmentation policies. `[dep: WI-8]`**

**Policy B (geometry-only):** random 128×128 crop, random lateral flip, mild
spatial jitter (±4 px). **Never axial flip** — depth is physically asymmetric here
(attenuation, transmit f-number geometry, per-depth aperture coverage) and axial
flipping teaches a false invariance. Lateral flip *is* valid: the array is linear,
symmetric, and centred at x = 0.

**Policy A (anatomy-focused), in priority order:**

1. **Receive angular-sector subsetting — PRIMARY.** Random sector subset and random K-reduction level. **⚠ rev-5: the random K-reduction half of this is now OFF by default and reclassified as an ablation** — see "K-jitter" below. The random *sector subset* remains primary and on. Always available, no PW stack needed, and it is the axis the store is built around.
2. **Transmit-angle subsetting — SECONDARY.** From the PW(5) stack: the physically exact positive-pair generator (§3.2b), two angles observing identical scatterers. Secondary because its angular leverage is capped at ±10° by grating lobes and because it depends on the stack that is cut candidate #1.
3. Sector dropout; TGC/gain jitter (random smooth depth-dependent gain); additive complex Gaussian noise at sampled SNR.
4. *Ablation only, never default:* depth-invariance augmentations (blur, resolution jitter, aggressive TGC randomization) — see §4.5c for why these are off by default.

**Policy B (geometry-only)** as above.

**⚠ K-jitter: off by default (rev-5 decision).** Drawing a random K-reduction
level per view confounds the K-ablation this store was built to run. §5.2
WI-13 trains a *separate model per arm* (K ∈ {4,2,1} @128, {16,8} @64) to ask
whether finer receive angular resolution buys anything; jittering K trains the
K=4 arm on views coarsened toward K=1, i.e. teaches it to be invariant to
angular resolution. If that succeeds, the K=4 arm discards precisely what
distinguishes it from K=1, the ablation reports "K doesn't matter", and the
finding is an artifact of the augmentation rather than physics. The arms are
not even augmented comparably — at K=1 the jitter is a no-op, since only one
level exists.

Policy A loses nothing essential: its priority-1 augmentation is the random
contiguous *sector subset*, which produces the intended positive pair (two
receive sub-apertures of the same scatterers) while leaving surviving sectors
at full angular resolution, so the channel ↔ angle correspondence stays stable.
Sector dropout, TGC jitter and additive noise are unaffected.

Two secondary reasons: reconciling a drawn k with the encoder's fixed 2K input
requires repeating surviving sectors back up to K, which produces duplicate
channels (a blur plus a half-bin angular shift) that no acquisition produces;
and at k=1 a view carries 1/K the angular information of its partner, an
aggressive invariance demand for an objective as collapse-prone as VICReg
(§4.7) in its very first runs.

So it becomes a *measured* question rather than an assumption — exactly how
§4.5c already treats the depth-invariance augmentations. Run "Policy A +
k-jitter" as one documented ablation cell in the §5.2 WI-13 matrix. Implemented
as `LoaderConfig.k_jitter` (default `False`); a middle setting restricting the
draw to adjacent levels (k ∈ {K, K/2}) bounds the invariance demand if the full
ladder proves too aggressive.

Crop sampling is **depth-stratified** (uniform in depth, not in area — §4.5c
measure 1) over the **full 0–40 mm** range for K ≤ 4, and over 8–40 mm for K ≥ 8
(§3.2c). `zea.func.ultrasound.make_tgc_curve` supplies the analytic gain
compensation of §4.5c measure 2, evaluated per-dataset at that dataset's own `fc`.

*Done when:* both policies are implemented; a test asserts no axial flip is ever
emitted over 10⁴ samples; a test asserts the sampled-crop depth histogram is uniform
in z (not in area) to within tolerance; a test asserts the K ≥ 8 sampler respects
`z_min = 8 mm` and the K ≤ 4 sampler does not restrict; a figure grid shows 8
augmented views of one crop under each policy; and the transmit-angle pair generator
is verified to draw two *different* angles at the same frame and crop location.

---

**WI-12 — VICReg (primary arm) + full instrumentation. `[dep: WI-2, WI-8, WI-11]`**

ResNet-18, stem `Conv2d(2K, 64, 7, 2, 3)`, 3-layer MLP projector (2048-2048-2048),
VICReg invariance/variance/covariance loss. Depth-stratified 128×128 crops from
the full 0–40 mm range at K ≤ 4 (8–40 mm at K ≥ 8, §3.2c), sampled uniformly in
depth (§4.5c). Analytic TGC compensation on by default. Two runs: Policy A and
Policy B.

**Every instrument in §4.7 ships before the first run** — three loss terms logged
separately, projector per-dimension std histogram, backbone 512-d singular-value
spectrum, RankMe, detached online probe, kNN eval, random-init floor on every chart.
`train/README.md` carries the failure-signature table verbatim.

*Done when:* both runs converge; all three loss terms and RankMe are logged per
epoch with the random-init floor visible on every chart; **RankMe is stable or rising
across training** (a falling RankMe with falling loss halts the run as dimensional
collapse); the detached probe is confirmed non-gradient-carrying by a test; the
frozen-feature results matrix from WI-2 runs unmodified against the trained encoder;
and both runs beat random-init and PCA on at least one label axis. Per-axis probe
degradation under Policy A is **recorded as an expected result, not a failure**.

---

**WI-13 — Results matrix over all cells. `[dep: WI-12]`**

`(K ∈ {16,8,4,1}) × (policy A, B) × (all axes in §4.1) × (probe, cluster)`, with the
four baselines. Note a separate run is needed per K, since K sets the stem's input
channel count.

*Done when:* the matrix is complete, the Policy A vs Policy B contrast is reported
explicitly as the deliverable it is, and every cell that was **not** run is listed
with its reason. No silent truncation.

---

**WI-14 — Spatial MAE baseline arm. `[dep: WI-12]`** *(documented baseline only)*

ViT-S/16 on 128×128 (64 tokens), mask 75 %, same crops, same eval. Included because
the brief asks for it as a documented comparison and because the project's whole
pivot rests on the claim that MAE underperforms joint-embedding on linear probing —
that claim should be measured on *this* data, not cited.

---

**WI-15 — Cross-aperture prediction arm. `[dep: WI-12]`** *(optional)*

Hold out one receive sector, predict it from the others (`--mask-axis rx`, the rev-4
default — §4.5b). `tx` only if WI-0 shows a non-zero reciprocity residual.
Encoder ResNet-18, light conv decoder.

*Done when:* the **§4.5a(a) mitigation is asserted by a test** — the input tensor
never carries both aperture axes (a `tx`-masked batch has receive fully summed, an
`rx`-masked batch has transmit fully compounded); the trivial-copy baseline is
reported **first**, so any learned gain is measured against it; the closed-form van
Cittert–Zernike coherence prediction is reported as the non-learned baseline for
`rx`; and the prediction residual is correlated against `phase_error` and `sos_std` —
the physical claim this arm exists to test. If `--mask-axis tx` is ever run, a
transposition-shortcut control is reported: train with the receive data *also*
available and confirm the loss collapses, demonstrating the shortcut is real.

---

**WI-16 — Stanford transfer eval. `[dep: WI-12]` — first-class deliverable, not optional**

Per §3.6. **Rev-6: the `Refocus` step is deleted** — the source ships native FSA
acquisitions, so the transmit representation already matches NVIDIA's (see §3.6's
rev-6 correction). What remains is: read the FSA `.mat` files with
`beamform/stanford_mat.py`'s HVMux recombination, then beamform to the shared store
format using the identical `AngularSectorSum` op. Run both grid variants (mm-matched
and λ-matched) as an ablation. Frozen-feature probes only — no retraining.

**Step 1 is done (2026-08-16):** `scripts/wi16_fetch_stanford_fsa.py` builds the
corpus at `/mnt/home/benbarkow/stores/wi16_stanford_fsa.zarr` — 117 frames of
`(192, 192, 287)` baseband IQ, read with `data/stanford_fsa.open_fsa_store`. What is
left is the beamforming pass, which needs a GPU instance.

*Done when:* both grid variants are built; Stanford's own reference B-mode passes the
same K=1 validation gate as WI-4; the headline transfer table uses the **full
0.4–21.2 mm range at K ≤ 4** with the 8–21.2 mm restricted row reported alongside at
every K (the §4.5c falsification instrument); the 64-px secondary crop row is
reported for K ≥ 8; and the Stanford probe axes (session, file, depth, echogenicity)
are reported against the same baselines including random-init.

⚠ **"Subject" is not a published field.** figshare flattened the per-rat directory
tree its own `mainLoader.m` walks, so the corpus records an *inferred* `session`
(acquisitions clustered by timestamp: 30 sessions over 117 files) and the exact
`source_file`. Split on `session`, and do not claim it is an animal id. The
per-animal ground truth that *is* published — ex vivo speed of sound, liver fat
percentage, steatosis grade for 20 rats, in `SoSDataAll.xlsx` — cannot currently be
joined to the acquisitions for the same reason. Recovering that join would turn
Stanford from an unlabelled transfer corpus into one with a **real, non-simulated
sound-speed target**, which is the most valuable single addition available to §4.1.

### 5.3 Sequencing by risk, and what gets cut first

Critical path: **WI-0 → WI-3 → WI-4 → WI-4b → WI-5 → WI-6 → WI-8**, with
**WI-2 running fully in parallel** (it needs only WI-0).

Risk ranking (rev-3: WI-0 rose, WI-4 fell):
0. **WI-0** — the IQ axis ambiguity (§6.18). Two authoritative sources disagree, and the error is invisible in both shape and image. It silently corrupts every transmit-axis operation downstream. Rev-2 treated WI-0 as a formality; it is not.
1. **WI-4** — beamforming correctness. Still the largest blast radius, but **materially de-risked by zea**: delays, interpolation, phase rotation and f-number masking are now library code that others have validated (§0.1), and OpenH-RF's `reconstruct.py` is a working reference for this exact dataset. Our residual risk is the sector segment-sum alone. The negative control remains non-negotiable.
2. **WI-5** — the raw-store / normalize-at-runtime split (§3.2). Getting the order wrong breaks the ladder silently: every K still produces an image, and only the deep-K numbers are quietly wrong. Second only to WI-4 in blast radius.
3. **WI-8** — 271 GB ingest, 61.5 GB output. Bounded but slow; resumability (WI-6) is what keeps it from being a repeated cost.
4. **WI-12** — corpus size (§6.7). Mitigated, not solved.
5. Everything else.

**Cut order if scope must shrink** — first cut first:
1. **PW stack down-scoping**: PW(5)×K=4 → PW(5)×K=1 (61.5 → 35.8 GB). Policy A keeps its exact positive-pair generator; only the joint angle × sector structure is lost. Cheapest real saving available.
2. **WI-15** cross-aperture arm. Explicitly scoped optional in the brief.
3. **WI-14** MAE baseline. Cut *reluctantly* — it is the evidence for the pivot.
4. **K ablation cells in WI-13**: drop K ∈ {8, 4}, keep K = 16 and K = 1.
5. **WI-9** HF publication. Purely a distribution concern; the science does not need it.
6. **WI-16 λ-matched variant only** — keep mm-matched, drop the ablation.

**WI-16 is no longer a cut candidate**, and it moved from last to protected: it is
now the only real-tissue evidence in the project (§3.6, §6.8). Dropping the PW stack
entirely is also *not* on this list — it is a deliberate decision with the named cost
in §3.2b, not a scope trim.

Never cut: WI-0, WI-2, WI-4, WI-5, WI-7, WI-10, WI-16, and the §4.7 instrumentation.
Those tests (frame assert, leakage assert, reference validation, ladder identity,
normalization guards) plus RankMe are what stand between this project and
confidently-reported noise.

### 5.4 Resource estimates — single RTX 4090 (24 GB)

**Everything below fits.** Nothing in this plan requires more than 24 GB.

| Job | Time | Peak GPU mem | Notes |
|---|---|---|---|
| WI-8 build, per frame | ~10–15 s | ~1.6 GB | IQ 265 MB + delay tables ~333 MB + sector map 41.6 MB + 16 accumulators 59 MB |
| WI-4b PW(5) synthesis + beamform | **+2.8 %** on the above | negligible | per-angle beamform is Σ_rx only: 1/180 of an FSA beamform |
| WI-8 build, total | **~3–4 h GPU**, 6–10 h wall | | wall is download-bound: 271 GB |
| VICReg (WI-12), per run | **4–6 h** | **~8–10 GB** | ResNet-18 @128², batch 256×2 views, AMP; ~400 epochs × 115 steps |
| §4.7 instrumentation | **< 2 % overhead** | +~0.5 GB | one 512×512 SVD/epoch; probes on detached features |
| MAE (WI-14), per run | ~2–3 h | ~6 GB | ViT-S/16, 64 tokens |
| Cross-aperture (WI-15) | ~3 h | ~8 GB | |
| Frozen DINOv3 baseline | ~20 min | ~3 GB | inference only, K=1 log-envelope |
| Eval matrix (WI-13) | ~30 min/cell | < 4 GB | dominated by t-SNE/UMAP on CPU |

Angular binning adds **no** beamforming cost over element grouping (§3.2 — sectors
are contiguous index ranges, so it is a scatter-add, not a K-fold loop).

**⚠ Measured, rev-5 (TITAN RTX 24 GB, cluster `24g`).** Two corrections to the WI-8
row above, both from the real build (`scripts/wi8_build.py`):

- **Per-frame GPU is ~3.2 s, not 10–15 s** — measured repeatedly at 3.22–3.27 s
  once the frame is on local disk, at the full 507×456 grid with K=16. So the
  GPU half of the build is ~50 min, not 3–4 h.
- **The wall-clock estimate depended entirely on *how* the data is read.**
  `datasets` streaming sustains only **~2.7 MB/s** here (271 GB ⇒ ~28 h, i.e. far
  *worse* than the 6–10 h assumed), while `hf_hub_download` of the same parquet
  shards sustains **68–76 MB/s** (⇒ ~1 h). The build therefore fetches one shard,
  beamforms its frames from local disk, and deletes it — bounding disk at one
  shard rather than 271 GB, and putting the realistic total at **~2 h**. The two
  read paths were verified to produce bit-identical output.

Storage came out at **~26 MB/frame compressed (~24 GB)** against the 29.6 MB/frame
uncompressed figure in §3.7.

**⚠ The binding constraint is host I/O, not GPU, and the PW stack makes it worse.**
A naive crop loader reads 2 MB per 128×128×16 crop; at 64 crops/frame × 923 frames
that is ~118 GB per epoch, and the store is now 61.5 GB rather than 27.3 GB.
Mitigation, planned into WI-12: a **frame-cached shuffle buffer** — hold ~64 frames
resident per worker, draw many crops from each, then rotate. At 66.6 MB/frame that
is 4.3 GB per worker, up from 1.9 GB, so the worker count must be capped
accordingly. Fallbacks, in order: (i) load the FSA K=16 and PW stacks *separately*,
since Policy B never touches the PW stack at all; (ii) float16 storage (30.8 GB
total). The plan assumes (i) — it is free and it means Policy B runs at the original
I/O cost.

Disk: 271 GB transient (one shard at a time if streamed, ~6 GB) + **61.5 GB output**
+ published copy. Add ~50 MB for the frame-invariant sector and occupancy maps.

---

## 6. ⚠ Disagreements and contradicted assumptions

**6.1 — There are no acquisition sessions, and the confounding failure mode as
described does not exist.** The brief's central worry is that "the latent may
organize by acquisition session rather than by tissue, because probe settings and
depth dominate". This is a *simulated* dataset: 923 independent k-Wave runs with
byte-identical geometry (`fs`, `fc`, `fd`, `c0`, `elpos`, `bmode_extent`,
`sound_speed_extent` each take exactly one value across all 923 frames). There are
no probe settings to confound with. The half of that sentence that *does* apply is
**depth** — and it applies with full force. §4.3 keeps the mechanism and retargets
it: the default UMAP colour is crop depth, not acquisition ID.

**6.2 — The leakage assertion needs a different key, and the risk is lower than
feared.** `frame_idx` is 0 for all 923 rows; there are no adjacent frames within an
acquisition, because each sample *is* one acquisition. The assertion is kept, keyed
to `frame_id`, and its real job becomes patch-level (§4.2). Measured evidence in
§4.6/§8 supports this.

**6.3 — WITHDRAWN, and I had this wrong.** The first draft claimed transmit-angle
subsetting was "not directly available". That was true as-stored and wrong in
principle: the full 180 × 180 Tx–Rx matrix synthesizes any transmit sequence post
hoc, and — measured — doing so costs **+2.8 % compute**, not 5.3 TB. The 5.3 TB
figure was for storing all 180 raw transmits, which was never the proposal. Policy A
keeps its physically exact positive-pair generator. See §3.2b for the storage
costing and the decision (FSA × K=16 + PW(5) × K=4, 61.5 GB).

What survives of the original point, now measured: pitch = 0.848 λ caps usable
steering at **±10.30°** before a grating lobe enters the visible region, so transmit
angular diversity (±10°) is *narrower* than the receive sector span (±26.6°). The
transmit axis is exact but its leverage is modest.

**6.4 — `t0` is a beamformer input, so probing it may be at chance by
construction — and that is the correct result.** `t0` is the only categorical
acquisition parameter (3 levels, 435/116/372). But a *correct* beamformer consumes
each frame's own `t0`, which removes it from the output. A `t0` probe at chance is
therefore evidence the beamformer is right, not evidence the representation is bad.
It is worth including precisely for that reason, and it must be reported with this
caveat inline or it will be misread.

**6.5 — Element-granularity dropout is impossible after sub-aperture summation
(unchanged, but the granularity is now depth-dependent).** Policy A's "element
dropout" cannot be implemented on a `(2,Z,X,16)` store. Under angular sectors the
finest available granularity is one sector, which contains a **depth-dependent**
number of elements — ~3 at z = 10 mm, ~11 at z = 40 mm (§3.2). So sector dropout
removes a roughly constant *angular* fraction (1/16 of the view cone) but a varying
*element* fraction. That is the physically meaningful invariant of the two, and it is
another reason angular binning is the right choice — element-group dropout would have
removed a constant element fraction and a wildly varying angular fraction.

**6.6 — `phase_error` and sound-speed heterogeneity are 58.5 % the same variable, and
partialling does not rescue it.** Measured: corr = +0.756; partial corr given
`sos_mean` = **+0.717** (barely reduced — the link is *direct*, not mediated);
R²(`phase_error` ~ `sos_mean` + `sos_std`) = **0.585**. ⇒ the `phase_error` row
carries 0.585 as a printed oracle-baseline column; a probe at or below it is
consistent with representing sound speed only (§4.1).

**6.7 — 923 frames is still a small corpus, and VICReg does not fix that.** It is
5.5× the Stanford set, and moving to the beamformed domain removes the delay-geometry
learning burden the brief correctly identifies. Both help. But a 507 × 456 frame
holds only ⌊507/128⌋ × ⌊456/128⌋ = **9 non-overlapping 128×128 crops**, so the corpus
is ~8.3k semi-independent patches — orders of magnitude below where SSL pretraining
normally lives. **The honest target is: beat
random-init, PCA, and handcrafted speckle statistics on linear probing, with the
Policy A vs Policy B contrast as the scientific deliverable.** It is not: produce a
foundation model. The plan should be judged against the former.

**6.8 — RESOLVED by reinstating Stanford.** Every NVIDIA frame is "cysts in
background tissue with a heterogeneous sound-speed field", so its tissue label is
binary and there is no tissue-type taxonomy to cluster into. That remains true — but
under the sim-pretrain / real-probe structure (§3.6) it is no longer the project's
only tissue evidence. Stanford supplies real in-vivo rat tissue for the probe stage.
The NVIDIA corpus is now doing what simulation is good at (dense physics ground
truth, unlimited controlled variation) and Stanford is doing what real data is good
at (tissue realism), which is the right division.

**6.9 — RESOLVED.** The concern was that switching the sole corpus to an NVIDIA
simulation changes the dataset the project is *about*, contradicting the project
description's explicit naming of Stanford. Under §3.6 there is no switch: Stanford
remains the evaluation corpus and NVIDIA becomes the pretraining corpus. Sim-pretrain
/ real-probe is a standard and defensible structure, and it arguably serves the
project description's "benchmark representation quality across signal stages" better
than Stanford-only would, since NVIDIA supplies ground truth Stanford does not have.
Worth stating to the supervisor as a design choice, but it no longer needs sign-off
as a scope change.

**6.10 — Two pieces of the brief are already better served than assumed.**
(i) The dataset ships `bmode` and `bmode_focused`, so the "benchmark across signal
stages" axis and the beamforming reference both come for free, and the eval harness
can be validated before the beamformer exists (§4.6). (ii) `segmentation_map` is
pixel-aligned with the output grid, so per-crop tissue labels need no resampling.

**6.11 — `sound_speed_map` absolute calibration is unverified.** Values sit below
the declared `c0`, and NVIDIA's own inference script scales its network output by
1000. Probe *relative* structure; treat absolute m/s claims as unconfirmed pending
§7.

**6.12 — NEW, measured: the sound-speed field is nearly constant within a frame, so
spatially-resolved SoS probing is not viable.** 98.2 % of crop-mean SoS *variance* is
between frames; within-crop spread (7.58 m/s) exceeds within-frame between-crop
signal (6.32 m/s). ⇒ SoS is a **frame-level** target with effective n = 923, not a
crop-level target with n = 144 000. The good news is that the specific failure the
review anticipated — the probe silently measuring depth — is **disconfirmed**:
corr(crop SoS, depth) = −0.023 pooled, −0.177 within-frame. Full numbers and the
substitute control in §4.1.

**6.13 — NEW: angular sectors are sparse at shallow depth and no binning scheme
fixes it.** At f# = 1.0, K = 16 sectors hold < 3 elements each above z = 9.5 mm and
< 1 above z ≈ 4 mm. This is a property of a 180-element array, not of the sub-aperture
definition. Mitigated by setting the span from f# = 1.0 and excluding crops shallower
than 8 mm (20 % of the depth range), with the occupancy map stored so the threshold
is a config change rather than a rebuild. Full table in §3.2.

**6.14 — DINOv3 is access-gated.** `facebook/dinov3-*` on HuggingFace is
`gated: manual` — a human must approve. Request access in WI-1 so the lead time does
not block WI-13. Ungated fallback `facebook/dinov2-small`; report which was used.

**6.15 — REV-3: the 8 mm depth floor was mine, not the dataset's.** Rev-2 read
ambiguously. Measured: `bmode_extent` z spans **0.0–40.0 mm**. Stanford's 0–21.6 mm
is fully in-distribution, and at K ≤ 4 no depth restriction is needed anywhere.
Full reasoning in §3.2c. Rev-2's "restrict the headline transfer to 8–21.6 mm" is
withdrawn for K ≤ 4.

**6.16 — REV-3: zea's augmentation library is one class.** `zea/data/augmentations.py`
contains only `RandomCircleInclusion`. Policies A and B remain entirely ours. Stated
because the review's instruction was "don't rewrite anything they have" — they have
essentially nothing here.

**6.17 — REV-3: zea's TAESD is a natural-image model, not an ultrasound one.**
`taesdxl` is the stock Stable Diffusion XL Tiny AutoEncoder re-implemented in Keras.
It belongs in the same baseline group as DINOv3. The genuinely ultrasound-domain
frozen baselines zea offers are **HVAE** (EchoNet-LVH cardiac B-mode) and
**Speckle2Self** (in-vivo). Both added; this is a net gain over rev-2, just not the
one expected.

**6.18 — REV-3: the IQ axis order is ambiguous between two authoritative sources,
and the ambiguity is invisible.** NVIDIA's card says `[n_tx, n_el, n_ax]`;
OpenH-RF's converter transpose implies otherwise, and its own inline comment
contradicts its code. Because `n_tx == n_el == 180` the shape is identical either
way, and because the Tx–Rx matrix is near-symmetric by acoustic reciprocity the
reconstructed B-mode looks correct either way. **This is the highest-quality example
of the failure mode this whole plan is built to catch.** WI-0's diagonal-onset test
resolves it and must land before any transmit-axis operation.

**6.19 — REV-3: `bmode_focused` is documented two different ways.** NVIDIA: "focused
transmit b-mode". OpenH-RF: "B-mode (DBUA aberration-corrected)". These are not the
same thing, and it changes what the field means as a signal stage. Labelled
`bmode_variant_2` until WI-0 resolves it.

**6.20 — REV-3: OpenH-RF does not contain Stanford.** The hope that it might carry
both corpora behind one abstraction does not pay off — it has three sub-datasets
(`nv-raw2insights-us`, `pala-ulm-ratbrain`, `tracked-cirs-phantom`). WI-16's
cross-probe work stands. What *is* removed is the file-reading half, by zea's
Verasonics converter and `Refocus` (§0.4) — a bigger win than the one that was hoped
for, since it unifies the *transmit representation*, not just the file format.

---

## 7. Open questions

1. **Absolute SoS calibration** (§6.11). Resolve against the DBUA repo or the IEEE TMI 2026 "Ultrasound Autofocusing" paper before quoting m/s.
2. **Does NVIDIA's `bmode` use a receive f-number, and which?** WI-4 answers this empirically by testing three sector spans (unlimited, f# = 1.5, f# = 1.0); the winner sets `θ_max` for the whole build (§3.2, §3.4). This is now a *blocking* question rather than a contingency, because the sector span defines the stored tensor.
2b. **Is the 8 mm shallow-crop exclusion the right threshold?** Set from the occupancy table (§3.2) at f# = 1.0. WI-3 commits the measured occupancy curve; the threshold is a config value, so it can be revised without a rebuild.
2c. **Does λ-matched or mm-matched resampling transfer better to Stanford?** Untested hypothesis from review (§3.6); cheap, run as an ablation in WI-16.
3. **Is the IQ axis order really `[n_tx, n_rx, n_ax]`?** The card says so and `inference.py`'s flattening is consistent with it, but the array is square in its first two axes so nothing has *proven* it. WI-0 has a concrete test (diagonal onset).
4. **`t0`'s three levels — what varies?** In cycles of `fc` they are −1.1453 / −1.0817 / −1.0181, evenly spaced by 0.0636 cycles. Likely a transmit pulse-length variant. If it is *only* a time origin, §6.4 applies fully; if it also changes pulse bandwidth, the probe has real signal.
5. **Is float16 storage acceptable?** Halves the store to 13.6 GB and would let the whole corpus live in host RAM (§5.4). Needs a dynamic-range check against the measured per-depth RMS decay before committing.
6. **Supervisor sign-off on the corpus switch** (§6.9).

---

## 8. Empirical leakage check — measured results

Performed during planning on **all 923 samples**, over HTTP range reads of the small
parquet columns only — no IQ payload was downloaded. Two independent phantom
signatures were compared: the 32×32 sound-speed field, and the 507×456 cyst
segmentation mask reduced to a 169×152 occupancy signature.

**Sound-speed field (all 923):**

- 0 exact duplicates.
- Nearest-neighbour RMS difference: min 3.64 m/s, 1st pct 3.99, median 9.44, max 21.20 m/s.
- val → train NN median 9.36 m/s vs overall median 9.44 — **validation frames are not systematically closer to train than train frames are to each other.**

**Cyst geometry (all 923):**

- 74 exact duplicates — **all 75 members of the empty-mask group** (§2.6 issue 6). A single hash, no other collisions.
- Restricted to the 848 non-empty frames: **0 exact duplicates.**
- NN distance: min = 12.7 % of the median pairwise distance; 1st pct = 18.9 %; median = 50.9 %.
- val → train: min NN = 23.2 % of median pairwise, median NN 21.10 vs train-internal median NN 22.29. **Indistinguishable.**

**Distributional match across NVIDIA's split:** `sos_mean` 1474.2 ± 47.0 (train) vs
1471.1 ± 47.8 (val); `phase_error` 1.826 ± 0.277 vs 1.827 ± 0.277; all three `t0`
groups present on both sides (390/108/332 train, 45/8/40 val).

**Conclusion.** Every frame is an independent k-Wave simulation with a distinct
phantom. There is no near-duplicate structure within the corpus and none across
NVIDIA's train/validation boundary. **The frame-adjacency leakage mode the brief
identifies as "the single most likely way this project produces meaningless numbers"
does not exist in this dataset**, and NVIDIA's own split can be adopted as-is.

This does not retire the assertion. The residual risk moves to **patch level** —
overlapping 128×128 crops from one frame landing on both sides of a probe split —
which is what `assert_no_frame_leakage` (WI-10) actually guards. It stays
unconditional in the probe path.

---

## 10. WI-0 RESULTS — measured on 4 real frames

Run: `N_FRAMES=4 python scripts/wi0_probe.py`; output checked in at
`tests/data/wi0_results.json`; asserted by `tests/test_nv_frame.py`. Frames span two
`t0` groups (2× g0, 2× g2). ~280–400 s per frame to stream 265 MB.

### 10.1 Documentation assertions — all pass

| Assertion | Measured | Verdict |
|---|---|---|
| `iq_real`/`iq_imag` shape, dtype | `(180, 180, 1024)` float32 | ✅ |
| element pitch | **201.0 µm** on all 4 (σ < 0.001) | ✅ |
| `fs`/`fc`/`fd`/`c0` | 13.330 MHz / 6.5 / 6.5 / 1540 | ✅ |
| `t0` | both values ∈ the 3 measured levels | ✅ |
| `bmode`, `segmentation_map` | `(507, 456)`; seg ⊆ {0, 1} | ✅ |
| `bmode_extent` | `[-17.9895, 17.9895, 40.0, 0.0]` mm | ✅ **z starts at 0.0 mm** |
| `frame_idx` | 0 | ✅ |
| per-frame IQ RMS | 0.602, 0.629, 0.681, 0.730 | card's global 0.6616 sits mid-range ✅ |

**§3.2c is confirmed against the payload, not just the metadata columns: the
simulation spans 0–40 mm and the 8 mm floor is entirely ours.**

### 10.2 ⚠ Axis order — NOT DETERMINABLE, and that is the answer

| Metric | Measured (mean of 4) |
|---|---|
| onset-matrix asymmetry `mean\|onset − onsetᵀ\|` | **0.000000** (exactly) |
| per-index energy CV, axis 0 vs axis 1 | 0.00907 vs 0.00893 |
| corr(axis-0 energy, axis-1 energy) | **0.9994** |
| mean relative energy difference between axes | **0.000266** |

The two axes are **statistically indistinguishable**. The onset structure is exactly
symmetric; the per-element energy profiles agree to 0.03 %. No test on this data can
say which axis is transmit.

**Resolution: the NVIDIA-card / OpenH-RF contradiction (§6.18) is *dissolved*, not
decided.** For every operation we perform on NVIDIA data the choice is immaterial to
within the residual in §10.3. This is a stronger outcome than picking a side — but it
is a fact about *this* dataset only.

**⚠ Carry-forward: the convention must be pinned explicitly for Stanford**, where
reciprocity will not hold (§4.5a(c)). There it must be read from the Verasonics
`TX`/`Receive` metadata, never inferred from the data. Added to WI-16.

### 10.3 ⚠ Reciprocity residual — the headline number is misleading

| Metric | mean | min | max |
|---|---|---|---|
| `‖S − Sᵀ‖ / ‖S‖` (whole matrix) | **0.0248** | 0.0244 | 0.0253 |
| **`‖S − Sᵀ‖ / ‖S‖` off-diagonal only** | **0.1656** | 0.1508 | 0.1716 |
| magnitude-only `‖\|S\| − \|S\|ᵀ‖ / ‖\|S\|‖` | 0.0161 | 0.0156 | 0.0166 |

**The whole-matrix 2.5 % figure is dominated by the diagonal** (which is symmetric by
definition and, per §10.4, carries overwhelming energy). The number that matters for
every consequence in §4.5a is the **off-diagonal residual: 16.6 %.**

Consequences, re-derived from the correct number:

- **(a) mitigation — unchanged, still mandatory.** Transposition predicts an off-diagonal trace to 16.6 % relative error, i.e. it explains **97.2 % of the off-diagonal variance**. As an SSL pretext shortcut that is devastating. Never expose both aperture axes in one tensor.
- **(b) PW stack — the call is closer than rev-4 assumed, and it stays cut.** My §4.5a(b) gate said "build only if the residual is materially non-zero". 16.6 % amplitude *is* materially non-zero, so the gate as written would say build. I am overriding it, with the reasoning stated: (i) by variance the axes are 97.2 % redundant; (ii) transmit leverage is still the narrower span (±10° vs ±26.6°); (iii) **the residual is most likely k-Wave source/sensor discretisation asymmetry, not physics** — exploiting it would be learning the simulator, and it would not transfer. ⇒ **Build FSA × K=16 alone: 27.3 GB, not 61.5 GB.** Reversible; the gate is now a documented judgement rather than a threshold. See §10.6 open question 2.
- **(c) Stanford — unchanged and now quantified.** 16.6 % is what a near-ideal simulator produces. Real hardware plus `Refocus` residuals will be substantially higher, so the sim→real gap on this axis is real. The instrument stands: measure the same quantity on Stanford's post-`Refocus` data and report both.

### 10.4 ⚠ NEW FINDING — the diagonal is direct coupling, ~10⁴× the scattered field

Not anticipated anywhere in revisions 1–4, and it affects the beamformer directly.

- `onset_diag_mean = 0.0` — every `tx == rx` trace exceeds 1 % of the global maximum **at sample 0**.
- `onset_offdiag_mean = 1011.6` against a 1023 fallback — with a *global* 1 % threshold, essentially **no off-diagonal trace ever crosses it**.
- The total-energy-per-sample profile falls below 1e-4 of its peak by sample **5–6** on 3 of 4 frames (585 on the fourth).

⇒ The `tx == rx` and near-diagonal traces carry transmit ringdown / direct
electro-acoustic coupling that is roughly **four orders of magnitude stronger than the
scattered field**. A naive DAS sum over all 32 400 (tx, rx) pairs will be swamped by
it at shallow depth.

**Actions, added to WI-4:**
1. Apply standard **near-field / early-sample muting** before beamforming, with the mute depth measured rather than guessed.
2. Evaluate excluding or down-weighting the near-diagonal `|i − j| ≤ d` traces; report the K=1 validation correlation with and without.
3. **Re-measure the reciprocity residual after muting** — the §10.3 whole-matrix figure is contaminated by this coupling, and the post-mute number is the one that should govern §4.5a going forward.
4. The onset-vs-separation diagnostic must use a **per-trace** threshold; the global threshold used here is uninformative for exactly this reason.

### 10.5 `bmode_variant_2` — weak evidence for NVIDIA's description, stays tagged `gt_sos`

| Metric | mean of 4 |
|---|---|
| corr(`bmode` dB, `variant_2` dB) over the −60 dB region | 0.392 |
| median absolute difference | 7.22 dB |
| lateral sharpness ratio, `variant_2` / `bmode` | **0.976** |

The low correlation is **not** discriminative — any change of beamformer re-randomises
the speckle realisation while preserving coarse structure, so both hypotheses predict
it. The sharpness ratio is: DBUA aberration correction should *improve* lateral
resolution, and `variant_2` is measured 2.4 % **less** laterally sharp. That mildly
favours NVIDIA's "focused transmit b-mode" (a focused transmit is worse than full
synthetic aperture away from the focus) over OpenH-RF's "DBUA aberration-corrected".

**Not conclusive, so the conservative tag holds: `bmode_variant_2` stays `gt_sos` and
is blocked from the physics probes (§4.4b).** The decisive test is WI-4 — beamform
ourselves with and without the ground-truth SoS map and see which one `variant_2`
matches. Resolving it *upgrades* a signal stage; leaving it mis-tagged would
contaminate the primary physics results, so the asymmetry of cost is clear.

### 10.6 Open questions raised or changed by WI-0

1. **Mute depth for the direct-coupling diagonal** (§10.4). Measure in WI-4.
2. **Is the 16.6 % off-diagonal residual numerics or physics?** Discriminator: it should be roughly depth-independent if numerical, and structured with `|i − j|` if physical. One measurement on an already-downloaded frame; feeds the §10.3(b) override.
3. **Which beamformer does `variant_2` match?** WI-4 (§10.5).
4. `last_sample_above_1e-4` is meaningless as computed (diagonal-swamped). The FOV-coverage assertion should be recomputed post-mute; the analytic bound (worst-case sample 934 < 1024) is unaffected.

---

## 11. WI-2 RESULTS — eval harness, validated on random features

Built and run **before any beamformer or trained model exists**, on NVIDIA's own
`bmode` field (§4.6). 245 frames (93 validation + 152 train), 1960 depth-stratified
128×128 crops, 42 tests passing.

```
python -m ssl_us.eval.matrix --data bmode_subset.npz --out figures/eval_boot
```

### 11.1 Both guards fired on real data

- **`assert_no_frame_leakage` fired on the first run** — against my own driver, which merged the nested outer and inner splits into one flat dict. Nesting is not leakage, so the *caller* was fixed (`split_audit` now audits levels separately and additionally asserts every inner id is drawn from outer-train). The guard was right and I was wrong, which is the outcome that justifies having it.
- **`assert_no_label_leak` verified live:** running `--stage gt_sos` prints `BLOCKS targets (label leak): ['sos_mean_frame', 'sos_std_frame', 'sos_mean', 'phase_error']` and probes only the tissue and geometry axes.

Split audit: 245 frames → outer train 152 / validation 93, inner probe_train 122 /
probe_test 30. Geometry (z, x) explains **11.4 %** of random-ResNet feature variance.

### 11.2 Baseline table (R², or balanced accuracy where marked BA)

| Target | level | random ResNet-18 | PCA-64 raw | **speckle stats (23-d)** | trivial (mean+std) |
|---|---|---:|---:|---:|---:|
| `contains_cyst` BA | crop | 0.805 | 0.613 | **0.883** | 0.813 |
| `cyst_fraction` | crop | 0.676 | 0.290 | **0.750** | 0.660 |
| `sos_mean_frame` | frame | 0.578 | −0.005 | **0.862** | 0.250 |
| `sos_std_frame` | frame | 0.138 | 0.038 | **0.388** | 0.159 |
| `sos_mean` | crop | 0.225 | −0.010 | **0.605** | 0.033 |
| `phase_error` | frame | 0.142 | 0.010 | **0.443** | 0.160 |
| `z_centre` | crop | 0.459 | 0.370 | **0.693** | 0.049 |
| `x_centre` | crop | 0.038 | 0.069 | −0.001 | 0.001 |
| `t0_group` BA (chance 0.333) | frame | 0.394 | 0.310 | 0.489 | 0.406 |

All cells carry bootstrap 95 % CIs and PCA-64 capacity control; frame-level cells
have n_eff = 245, crop-level n_eff = 1960.

### 11.3 ⚠ My stated WI-2 stop-condition was wrong, and the harness proved it

Rev-1 through rev-4 all said: *"Expected outcome: `z_centre` probes well above chance
from random features, `cyst_fraction` near chance. If `cyst_fraction` is not near
chance, the harness is leaking and we stop."*

**Measured: `cyst_fraction` = 0.676 from random features — nowhere near chance.** By
my own written criterion the project should stop here. That criterion was wrong.

The diagnosis is in the `trivial` column: **two features (crop mean and std) reach
0.660 R² and 0.813 balanced accuracy.** Cysts are *defined* by being anechoic, so
"how dark is this crop" is very nearly the label. This is a property of the label, not
a defect in the harness — and `z_centre` = 0.459 vs `x_centre` = 0.038 is exactly the
asymmetry a correct harness should show (depth is inferable from content, lateral
position is not).

**Corrected stop-condition, for the record:** the harness is healthy if
`x_centre` ≈ 0 and `sos_mean` (crop) ≈ 0 from *raw-pixel PCA* — both hold
(0.069 and −0.010). `cyst_fraction` is not a leakage indicator on this dataset.

### 11.4 ⚠ The bar for the learned model is much higher than assumed

This is the substantive finding, and it lands before a single GPU-hour is spent.

- **Handcrafted speckle statistics beat a random ResNet-18 on every single axis**, often by a wide margin — `sos_mean_frame` 0.862 vs 0.578, `sos_mean` (crop) 0.605 vs 0.225, `phase_error` 0.443 vs 0.142.
- **`sos_mean_frame` = 0.862 from 23 classical features** is the headline. Frame-level sound speed is strongly encoded in speckle statistics — physically sensible (SoS mismatch defocuses, which changes speckle first- and second-order statistics). **The bar on the primary physics axis is R² = 0.862, not zero.**
- `phase_error` = 0.443 from speckle stats sits **below** the 0.585 SoS-oracle (§4.1), so that axis still has headroom for a representation that captures aberration beyond sound speed.
- `x_centre` is unpredictable from everything, as it should be.

**Consequences for the plan:**

1. **§4.4's baseline ranking was wrong about which baseline matters.** Rev-1 listed speckle statistics third, after random-init and PCA. It is in fact **the strongest baseline by a wide margin and the one the whole project must beat.** Promoted to the headline comparison in the results matrix; random-init and PCA are floors, not competitors.
2. **VICReg must clear speckle stats, not random-init**, to be worth reporting. Beating random-init is now known to be a low bar (`trivial` beats it on `contains_cyst`).
3. The §4.7 online-probe instrumentation should plot **the speckle-stats line, not just the random-init floor**, on every chart — otherwise a run will look successful while sitting below a 23-feature classical baseline.
4. Clustering on random features gives ARI ≈ 0.01–0.04 against every axis (cyst, `t0_group`, depth quartile), i.e. no structure — the correct null.

**Caveat carried:** these numbers are on `bmode` — NVIDIA's own beamformed output, not
our sub-aperture store — and on 245 of 923 frames. They are the *bootstrap* baseline.
They will be recomputed on the K-sector stages, where the comparison of interest
(§4.4 internal control) actually lives.

---

## 9. Note on what was and was not verified

Verified live, against the real artefacts: every number in §2.1–§2.6 and §8; the
Arrow schema; element pitch, aperture, and geometry constancy; the three `t0` levels
and their split breakdown; all label distributions and correlations; the absence of
duplicate phantoms; the licence terms; the repo file inventory and every line
reference in §1.3; and the true size of the legacy corpus (168 frames / 84 files /
20 subjects, not ~160 — the local `data/processed/` copy is a 14-frame subset).

**Not verified — and the reason WI-0 exists:** the `iq_real` / `iq_imag` payload was
deliberately not downloaded (265 MB/frame). Its declared shape, dtype, axis order,
and amplitude scale are taken from the card and from NVIDIA's `inference.py`. Every
beamforming estimate in §3 and §5.4 rests on that. **WI-0 must land before WI-3
onward is planned in any more detail than it is here.**

