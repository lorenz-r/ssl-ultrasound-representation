# MEMORY — everything known as of 2026-08-17

One file, written to be read cold. It records **what was run, what was
measured, what it means, and what I got wrong**. PLAN.md remains the design
doc and CLUSTER_HANDOFF.md the per-WI map; this is the state of knowledge.

Scope: the 2026-08-14/15 session on the cluster `24g` instance, which trained
seven models (six VICReg arms + two MAE variants) and probed all of them
(§1–§8); the 2026-08-15/16 session on a `data` instance, which fetched the
two corpora that were missing — NVIDIA's per-pixel label maps (§9) and the
Stanford FSA transfer set (§10); and the 2026-08-16/17 session, which **ran**
the crop-level anatomy columns and the qualitative pass (§11, §12).

---

## 1. Headline results

**1. Policy A and Policy B specialise in opposite directions, stably.**
Across all three K values and both splits: Policy A wins `x_centre` and
`sos_mean_frame` at *every* K; Policy B wins `t0_group` at every K. Policy B's
`x_centre` R² is **−0.000 at all three K** — its features carry literally no
horizontal position, which is the geometry-only augmentation behaving exactly
as specified. This is the cleanest and most reproducible finding in the set.

**2. MAE loses to an untrained encoder of comparable capacity.** Even the
*fair* MAE variant (raw targets, which genuinely learns — 12.4 % of masked
patch variance) ties `random_resnet18` on three targets and loses on four. It
beats it on none. VICReg beats it on all seven. WI-14's premise is confirmed,
and by a wider margin than "underperforms".

**3. Why MAE fails here, mechanistically.** A speckle patch factors as
`(local scale) × (interference pattern)`. Measured across neighbouring patches:

| component | neighbour correlation |
|---|---|
| per-patch scale (local echogenicity) | **0.87** lateral, **0.64** axial |
| normalised within-patch pattern | **−0.0001** — zero |

The interference realisation is uncorrelated even between *adjacent* patches,
so pixel-space reconstruction has almost nothing to predict. This is the same
physics WI-4 measured from the other side: NVIDIA's own two reconstructions of
one frame score pixel-wise NCC 0.197 against each other.

**4. Training against a signal-free objective actively degrades the
representation.** Both MAE variants peak at RankMe ≈ 332 at **epoch 4**, then
decay to ~160 over the remaining 219 epochs. The barely-trained model has the
better representation. A falling RankMe is therefore not automatically
"collapse" — check where it started.

**5. Policy A carries crop-level anatomy, and this is the project's strongest
evidence for the actual thesis** (added 2026-08-17, §11). Policy A wins all
three crop-level anatomy targets at **every K and on both splits**, with CI
separation in 17 of 18 cells:

| target (K=4 train) | A | B | random | trivial |
|---|---|---|---|---|
| contains_cyst (BA) | **0.951** | 0.909 | 0.792 | 0.784 |
| cyst_fraction_crop | **0.792** | 0.786 | 0.464 | 0.362 |
| sos_mean_crop | **0.720** | 0.615 | 0.482 | 0.300 |

This matters more than the frame-level `cyst_fraction` row (§3.2), which tops
out at +0.331 against a +0.170 trivial control. ⚠ **Not yet controlled for
geometry** — cysts have a spatial distribution and `z_centre` alone probes at
+0.699, so the residual question "is this partly *where the crop came from*"
is open. §12 has the one number that bears on it.

---

## 2. What exists on disk

| artifact | path | notes |
|---|---|---|
| K-sector store | `/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr` | 923 frames, `(N,2,507,456,16)` f32, 23 GB, **raw sums** |
| local staged copy | `/tmp/wi8_fsa_k16.zarr` | **ephemeral** — wiped on container restart |
| per-pixel label maps | `/mnt/home/benbarkow/stores/wi8_label_maps/` | 923 frames, 1.9 GB — `seg` `(N,507,456)` u8, `sos` `(N,32,32)` f32, `bmode`/`bmode_focused` `(N,507,456)` f32. Read with `ssl_us.data.label_maps.open_label_maps` |
| Stanford FSA corpus | `/mnt/home/benbarkow/stores/wi16_stanford_fsa.zarr` | 117 frames, `(N,192,192,287)` complex64, 8.8 GB — baseband IQ, **not** beamformed. Read with `ssl_us.data.stanford_fsa.open_fsa_store`. See §10 |
| VICReg K=4 | `~/checkpoints/wi12_vicreg_policy{A,B}/` | |
| VICReg K=16, K=1 | `~/checkpoints/wi13_vicreg_k{16,1}_policy{A,B}/` | |
| MAE normalized | `~/checkpoints/wi14_mae_k4/` | trains on noise — see §4 |
| MAE raw-target | `~/checkpoints/wi14_mae_k4_rawtarget/` | the fair baseline |
| all logs | `/mnt/home/benbarkow/logs/` | |
| committed results | `tests/data/wi13_*.json`, `tests/data/wi14_*.json` | `figures/` is gitignored |
| qualitative pass | `tests/data/wi13_k4_{train,validation}_{qual_results.json,clusters.md}` | §12 |

⚠ **The checkpoints are at `/mnt/home/benbarkow/checkpoints/`, one level above
the repo — not the repo's own `checkpoints/`,** which still holds two retired
phase-1 files (`best_dnorm_checkpoint.pt`, `best_mae_checkpoint.pt`) and none
of this work. That is what `--checkpoint-root` defaults to. The directory
totals 19 GB. Each `<dir>/last.pt` resumes; `history.json` and
`run_config.json` sit beside it.

---

## 3. Measured results

### 3.1 Training cost and convergence (all 223 epochs, 11.82 M samples)

| cell | hours | crop | final loss | RankMe start → end |
|---|---|---|---|---|
| VICReg K=4 A | 1.53 | 128 | 8.33 | 286.1 → 287.1 |
| VICReg K=4 B | 1.46 | 128 | 7.82 | 298.0 → 304.5 |
| VICReg K=16 A | 1.20 | 64 | 8.63 | 347.0 → 298.7 |
| VICReg K=16 B | 1.13 | 64 | 7.84 | 348.4 → 310.1 |
| VICReg K=1 A | 1.93 | 128 | 8.60 | 264.0 → 289.6 |
| VICReg K=1 B | 1.47 | 128 | 7.71 | 280.7 → 302.9 |
| MAE normalized | 1.96 | 128 | 0.9974 | 173.8 → 165.5 (peak 331.5 @ 4) |
| MAE raw-target | 1.95 | 128 | 0.6722 | 173.8 → 160.0 (peak 332.6 @ 4) |

**Policy B converges to lower loss and higher RankMe at every K — and is not
the better encoder for it.** Do not use training loss to pick an arm.

Every VICReg arm lands at final RankMe 289–310 regardless of start (K=16 falls
from ~347, K=1 climbs from ~264).

### 3.2 Probe matrix, K=4, train split (n_eff 830 frame / 6640 crop)

| target | vicreg_A | vicreg_B | mae_norm | mae_raw | random | pca_raw | trivial |
|---|---|---|---|---|---|---|---|
| phase_error | +0.366 | +0.438 | +0.256 | +0.346 | +0.393 | −0.007 | **+0.400** |
| t0_group † | +0.398 | **+0.526** | +0.374 | +0.428 | +0.427 | +0.318 | +0.397 |
| sos_mean_frame | **+0.899** | +0.865 | +0.861 | +0.800 | +0.800 | −0.001 | +0.671 |
| sos_std_frame | +0.277 | **+0.339** | +0.180 | +0.270 | +0.273 | −0.011 | +0.280 |
| cyst_fraction | **+0.331** | +0.312 | +0.025 | +0.097 | +0.128 | +0.001 | +0.170 |
| z_centre | +0.699 | **+0.756** | +0.314 | +0.388 | +0.645 | −0.001 | +0.541 |
| x_centre | **+0.537** | −0.001 | +0.002 | +0.013 | +0.177 | −0.000 | −0.000 |
| contains_cyst † | **+0.951** | +0.909 | +0.587 | +0.709 | +0.792 | +0.510 | +0.784 |
| cyst_fraction_crop | **+0.792** | +0.786 | +0.072 | +0.204 | +0.464 | −0.005 | +0.362 |
| sos_mean_crop | **+0.720** | +0.615 | +0.567 | +0.431 | +0.482 | −0.001 | +0.300 |

R² except † (balanced accuracy). Validation split (n_eff 93) is in
`tests/data/wi13_k4_validation_results.json` and agrees closely.

The last three rows are the crop-level anatomy targets, run 2026-08-17 (§11).
`n_eff` is 6136 rather than 6640 for the two cyst rows — crops from the 75
empty-`segmentation_map` frames are dropped (PLAN.md §2.6). The other seven
rows reproduced **bitwise** when these were added; that check is §11.1.

### 3.3 K-ablation, train split, each column vs its **own** controls

⚠ **Read columns down, not across.** §3.2c/d ties K to crop geometry by
design: K ≤ 4 → 128 px full depth, K ≥ 8 → 64 px below z = 8 mm. A K=16 cell
differs from K=4 in sector count, crop size *and* depth range simultaneously,
so cross-K differences are not attributable to K. The controls move too —
`trivial` scores +0.400 on `phase_error` at K=4 but only +0.114 at K=16.

| target | K=1 A | K=1 B | K=4 A | K=4 B | K=16 A | K=16 B |
|---|---|---|---|---|---|---|
| phase_error | +0.378 | +0.399 | +0.366 | +0.438 | +0.377 | +0.247 |
| t0_group | +0.407 | +0.495 | +0.398 | +0.526 | +0.390 | +0.466 |
| sos_mean_frame | +0.918 | +0.876 | +0.899 | +0.865 | **+0.931** | +0.849 |
| sos_std_frame | +0.294 | +0.322 | +0.277 | +0.339 | +0.335 | +0.162 |
| cyst_fraction | **+0.365** | +0.223 | +0.331 | +0.312 | +0.214 | +0.126 |
| z_centre | **+0.768** | +0.561 | +0.699 | +0.756 | +0.429 | +0.681 |
| x_centre | +0.528 | −0.000 | +0.537 | −0.001 | **+0.789** | −0.000 |

Every column clears its own controls on every target, by margins from +0.04
to +0.65.

**95 % CI separation** (`−` = genuinely tied, not merely noisy):

| target | K=1 (830/93) | K=4 (830/93) | K=16 (830/93) |
|---|---|---|---|
| phase_error | − / − | − / − | **A** / − |
| t0_group | B / B | B / B | B / − |
| sos_mean_frame | A / − | A / − | A / A |
| sos_std_frame | − / − | − / − | A / − |
| cyst_fraction | A / − | − / − | − / − |
| z_centre | A / A | B / − | B / B |
| x_centre | A / A | A / A | A / A |

### 3.4 MAE patch/mask sweep

Speckle autocorrelation on this grid (pixel pitch 0.0791 mm both axes):
**axial half-width 2 px, lateral 4 px**; ACF at lag 16 px = 0.075 / 0.142.

With `norm_pix_loss` the target is unit-variance, so `1 − loss` is the
fraction of masked-patch variance predicted:

| patch | mask | visible tokens | batch 256 | batch 64 |
|---|---|---|---|---|
| 16 | 0.75 / 0.50 / 0.25 | 16 / 32 / 48 | −0.4 % | −0.1 / −0.2 / −0.2 % |
| 8 | 0.75 / 0.50 / 0.25 | 64 / 128 / 192 | −0.3 / −0.4 / −0.4 % | 0.0 / 0.0 / −0.0 % |
| **4** | 0.75 | 256 | — | **+0.7 %** |

**Mask ratio is irrelevant** — tripling visible context at patch 16 moves the
loss by 0.0002. **Patch scale is the whole story**, and signal appears only at
patch 4, i.e. 1–2 correlation lengths. p4 needed batch 64 for VRAM so it also
got 4× the optimizer steps; the batch-64 column is the matched control and
shows that accounts for ~0.35 pp, with the patch effect surviving.

---

## 4. Things I got wrong (do not repeat)

1. **`norm_pix_loss` is backwards for speckle.** I enabled it citing He et al.
   ("improves linear probing"). True for natural images, where within-patch
   texture is predictable. Here it divides out the per-patch scale — the
   *only* predictable component (§1.3) — leaving pure noise, which is why that
   loss sits at exactly 1.0. The raw-target run I discarded as "misconfigured"
   was the correct baseline. **Default it off for this data.**
2. **I over-claimed "MAE's objective is information-theoretically empty on
   speckle."** The evidence supports something narrower: across patch 4–16 and
   mask 25–75 %, *pixel-space* masked reconstruction extracts < 1 % of
   masked-patch variance. A target in envelope or spectral space is untested.
3. **The `separated` column was garbage in the first WI-13 run.**
   `ProbeResult.row()` formats `ci95` as a *string*, so `a["ci95"][1] <
   b["ci95"][0]` compared characters. Fixed by carrying `score_lo`/`score_hi`
   as floats. Any conclusion drawn before commit `6c67042` is suspect.
4. **My own ballast caused the sweep's OOM.** `--reserve-mb 12288` left only
   12 GB and the 256-token configs need more. Not a data limit.
5. **A 93-frame validation split is underpowered** for the five frame-level
   targets, which made most A-vs-B cells look tied. The 830-frame train split
   resolves roughly twice as many. It is in-sample for the encoders (they saw
   the images, never the labels — no leak, standard SSL protocol), and the two
   splits agree closely, so it is trustworthy for *ranking*.

---

## 5. Two bugs that were in the code, now fixed

**`_is_collapsing` could never return** — it compared a window against its own
tail under `zip(..., strict=True)`, which is ragged by construction, so it
raised `ValueError` on every call past its length guard. Gated behind
`collapse_grace_epochs = 50`, it killed each arm at exactly epoch 50. Ruff's
`B905` demands an explicit `strict=`; the correct value is `False`. It had no
test coverage; `tests/test_train_vicreg.py` now has four.

**Crash and scientific halt were indistinguishable** — the collapse halt
returned exit 1, and Python also exits 1 on any uncaught exception, so
`wi12_run_both.sh`'s "rc=1 → don't retry" rule logged a hard crash as
`HALTED (collapse)` and skipped the arm. The halt now exits **3**.

---

## 6. Operating this cluster

- **Launch detached, always:**
  `setsid nohup bash scripts/<runner>.sh >> log 2>&1 < /dev/null &`.
  A job left as a child of an interactive or agent shell dies *with it* —
  silently, mid-epoch, no traceback, `oom_kill 0`. That killed WI-12 arm A at
  epoch 48; the idle GPU then got the instance reaped 1h45m later, wiping
  `/tmp`.
- **Never run `pytest` while training.** 4 cores, 4 loader workers — the suite
  starves them and GPU util reads 0 %, indistinguishable from a hang. It also
  triples the suite's own runtime (102 s vs 38 s).
- **Re-staging the store costs ~7 min.** NFS measured **58 MiB/s** under
  contention, not the 241 MB/s quoted in CLAUDE.md. Plan with the slow number.
- **Nothing inside the container survives a reap.** Watchdogs die with what
  they watch. Recovery is manual but cheap — checkpoints are on NFS, so
  re-running the launch command resumes every arm from `last.pt`.
- **`scripts/wi12_ballast.py` holds VRAM near the cap for free** (no SMs used;
  measured 1539–1672 pairs/s with it vs 1520–1726 without). Capped at
  `TOTAL − 10 GB` deliberately: a fill-free-memory ballast expands into arm A's
  footprint when A exits and then OOMs arm B. **Use the default reserve** —
  raising it to 12 GB is what broke the sweep. Pass `--done-log/--done-marker`
  per chain; a stale marker makes it exit instantly.
- **`pkill -f '<script>'` matches the agent's own shell** and kills the
  command mid-way (exit 144). Kill by PID.
- **Editing a running shell script can corrupt it** — bash reads incrementally.
  Stop, edit, restart.
- **`CLAUDE.md` is gitignored** (`.gitignore:6`). Durable notes go here, in
  CLUSTER_HANDOFF.md, or in PLAN.md.

Throughput reference: VICReg K=4 ~1690 pairs/s (compute-bound, 95–99 % GPU);
K=16 ~2800 pairs/s (64 px crops, *loader*-bound, GPU swings 11–80 % — judge
health by the log's pairs/s, not instantaneous utilisation); MAE ~1750 crops/s.

---

## 7. Reproducing any of it

```bash
source /opt/venvs/.venv/bin/activate          # PyTorch; .venv-zea for beamforming
pytest -q                                     # 331 passed, 6 skipped (skips = data/rats/)
ruff check .

setsid nohup bash scripts/wi12_run_both.sh   >> logs/wi12_night.log 2>&1 </dev/null &   # VICReg K=4 A,B
setsid nohup bash scripts/wi13_kablation.sh  >> logs/wi13_kablation.log 2>&1 </dev/null & # K=16,1 x A,B
setsid nohup bash scripts/wi14_run.sh 4      >> logs/wi14_chain.log 2>&1 </dev/null &   # MAE

python scripts/wi13_matrix.py --k 4 --split train --crops-per-frame 8 --out figures/wi13_k4_train
python scripts/wi14_mask_sweep.py --out figures/wi14_sweep
bash scripts/wi12_status.sh                   # read-only health summary

# per-pixel label maps (data instance; ~34 min, resumable, re-verify with --verify-only)
HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
  python scripts/wi13_fetch_label_maps.py --out /mnt/home/benbarkow/stores/wi8_label_maps

# Stanford FSA corpus (data instance; ~70 min, resumable, deletes each .mat as consumed)
python scripts/wi16_fetch_stanford_fsa.py \
  --out /mnt/home/benbarkow/stores/wi16_stanford_fsa.zarr --scratch /tmp/stanford_raw
```

Crop geometry is always derived from K by `LoaderConfig.for_arm` — never pass
`--crop-size` unless deliberately ablating it, or the encoder gets evaluated
on crops it never saw.

---

## 8. What is left

| # | item | where | cost |
|---|---|---|---|
| 1 | **WI-16 transfer.** The Stanford FSA corpus is built (§10); beamforming it to K-sector stores — both grid variants — and probing it is not. `AngularSectorSum` applies unchanged; the constants and `t0_s` do not. **The whole project is currently simulation-only.** | `12g` | ~2 h |
| 2 | **Write-up.** Results exist only as JSON + tables under `tests/data/`. Nothing outside this repo is readable. | — | — |
| 3 | **Geometry-controlled anatomy probes.** §1.5's headline is uncontrolled for crop position; re-probe on geometry-residualised features (`viz.regress_out_geometry` already computes them for the figures). | `24g` | ~20 min |
| 4 | **An external frozen baseline** — DINOv2-small on the free `bmode` stage. Baselines 5–8 of PLAN.md §4.4 are all unimplemented, so "does this beat an off-the-shelf encoder" is unanswered. | `24g` | ~20 min |
| 5 | MAE at patch 4 as a full run — the only config that showed signal | `24g` | ~4 h+ |
| 6 | WI-15 cross-aperture arm (optional in PLAN.md; `models/xaperture.py` exists and is tested, never trained) | `24g` | ~3 h |
| 7 | WI-9 HuggingFace publication (distribution, not science) | `data` | — |
| 8 | K = 8 column — dropped per PLAN.md §5.3 cut-list item 4. Recorded in every column's `not_run`. | — | — |

~~Fetch the per-pixel maps, wire them in, and run the columns.~~ **DONE** —
maps 2026-08-16 (§9), wiring 2026-08-16, columns run 2026-08-17 (§11).

Stanford is no longer missing — see §10. It came from figshare, not from the
Mac, and it is **native FSA**, which deletes WI-16's `Refocus` step.

**Three open questions worth stating plainly.**

1. **Every trained encoder ties or loses to `trivial` on `phase_error`** at
   K=4/K=1 — two numbers, envelope mean and std, handicapped 32:1 by
   `capacity_control`. Only K=16 Policy A clears it. Either phase error is not
   recoverable from these crops beyond global brightness, or the probe protocol
   is not the right instrument for it.
2. **The physics story is thinner than the headline number suggests.**
   `sos_mean_frame` = +0.899 is the best cell in the matrix, but `trivial`
   scores +0.671 and an *untrained* ResNet-18 scores +0.800 — so the learned
   contribution is +0.10, and most of the signal is global brightness. On
   `sos_std_frame` (+0.277 vs +0.280 trivial) there is nothing at K=4. Of the
   three physics axes only one has a real margin. Report deltas, not absolutes.
3. **K is confounded with crop geometry by design** (§3.3), so the ladder
   cannot attribute anything to sector count. One K=4 arm trained at 64 px
   below z = 8 mm would decouple it for ~1.5 h + a probe column.

---

## 9. The per-pixel label maps (2026-08-15)

The three crop-level anatomy targets were blocked because the store carries
frame-level *scalars* only. They are no longer blocked, and it did not cost a
re-download.

**The corpus never needed re-fetching.** `iq_real`/`iq_imag` are 265.4 MB of
each frame's 267.5 MB; parquet is columnar, so a column-pruned read over HTTP
range requests takes **1.9 GB instead of 271 GB**. 923 frames in **33.6 min**
on the `data` instance, ~1 MB/s decoded, dominated by per-shard overhead
(45 s per 19-row train shard, 3 s per 1-row validation shard) rather than
bandwidth. `pre_buffer=True` made no measurable difference — don't bother.
This is the same technique every number in PLAN.md §2 was measured with;
`scripts/wi13_fetch_label_maps.py` is now the durable version of it.

**Alignment is asserted, not assumed.** Frame ids are `"{split}/{index}"`, the
convention `wi8_build.py` assigns, so row *i* is store frame *i* — and the
fetch re-derives `cyst_fraction`, `sos_mean`, `sos_std`, `phase_error` and `t0`
from the fetched maps and compares them to the store's independently-written
`frame_metadata.json`. All 923 agree; worst relative error **1.2e-7** (float32
round-trip), `cyst_fraction` and `t0` exact. It also recovers **75 empty
segmentation maps**, matching PLAN.md §2's independently measured 75/923.
A misalignment here would attach the wrong anatomy to every crop and look
entirely plausible downstream, so the check is fatal and runs by default.

Re-read of PLAN.md §2.4's numbers straight off the fetched arrays, all
reproduced exactly: cyst fraction over the 848 non-empty frames median 3.8 %,
IQR 2.2–5.7 %, max 14.4 %; SoS global 1283.3–1629.5 m/s, frame means
1380.8–1530.8 (σ_between 47.1), σ_within 3.7–28.6.

**Reading them.** `ssl_us.data.label_maps.open_label_maps()` memory-maps the
four arrays and refuses to open an unfinished or unverified directory — the
`.npy` files are allocated at full size up front, so un-fetched frames read
back as **zeros**, i.e. an empty cyst mask and a 0 m/s sound speed, neither of
which looks like an error at the probe. `crop_labels(frame_ids, origins, size)`
takes the `(row, col)` origins `StoreCropBatch` already records — not the mm
centres, because recovering the origin from those means inverting the grid
mapping approximately and a half-pixel slip shifts every mask. `to_bootstrap()`
hands the WI-2 harness the real corpus in the shape it expects.

---

## 10. Stanford is native FSA (2026-08-16)

**The headline: WI-16 does not need `Refocus`.** PLAN.md §3.6 and WI-16 were
written on the belief that Stanford is focused multi-focal data that must be
inverted into multistatic form before the NVIDIA pipeline can touch it. That
belief came from the phase-1 repo, which only ever handled the MultiFocal
acquisitions. The published source —
[figshare 28291985](https://doi.org/10.25452/figshare.plus.28291985),
*Pulse-Echo Ultrasound Murine In Vivo Acquisitions*, Apache-2.0, 366 files /
239 GB — has **the same animals acquired three ways**:

| type | files | size | note |
|---|---|---|---|
| **FSA** | **117** | **76.3 GB** | one transmit per element — what we want |
| Hadamard | 117 | 50.9 GB | encoded FSA, needs decoding |
| MultiFocal | 81 | 37.1 GB | what WI-16 was written against |
| MF_Fast | 33 | 75.1 GB | |

`TX.Apod` on an FSA file has exactly **one** non-zero entry per transmit, 192
transmits for 192 elements. That is NVIDIA's scheme with different constants,
so the transmit axis matches by construction — no inversion, no
adjoint/Tikhonov/rSVD/tSVD choice, no residual to attribute transfer loss to.

### 10.1 What is on disk

`/mnt/home/benbarkow/stores/wi16_stanford_fsa.zarr` — 117 frames of
`(192, 192, 287)` complex64 baseband IQ, **8.8 GB**, one FSA acquisition per
frame, plus `coverage`, `frame_metadata.json` and `manifest.json`. Read it with
`ssl_us.data.stanford_fsa.open_fsa_store`. Built by
`scripts/wi16_fetch_stanford_fsa.py` in **62.9 min** on the `data` instance
(66–77 MB/s download, ~29 s decode per frame, raw `.mat` deleted as consumed so
peak scratch is one file rather than 76 GB).

Verified over all 117 after the build: no all-zero or non-finite frame,
per-frame max |IQ| 11 211–23 411, **84 `FSA` + 33 `FSA_Fast`** across 10 dates
/ 30 inferred sessions / 117 distinct source files, and the raw signal cutoff
is inside the 861-sample window on **every** file with room to spare. HVMux
coverage takes values 1, 2 and 3 — the `FSA_Fast` sequence uses three receive
apertures where plain `FSA` uses two, so per-element averaging differs and the
`coverage` array is worth carrying. Frames from both types beamform to
recognisable rat abdominal wall and liver.

⚠ **Depth: the store's axis runs −0.97 → 20.18 mm**, and the negative start is
correct, not a bug — see §10.2 on `t0`. Real signal ends at ~19.9 mm.

**76 GB → 9 GB, by three reductions, measured not assumed:**

| reduction | effect on the beamformed image |
|---|---|
| one frame per 10-frame cine | n/a — the other nine are the same view |
| truncate to 861 fast-time samples | envelope NCC **1.00000**, mean 0.003 dB |
| decimate ×3 after IQ demodulation | envelope NCC **0.99669** as stored; **0.99991** re-sampled back to 31.25 MHz |

The truncation is free because **the publishers already truncated the data** —
every file is identically zero past raw sample ~846 (19.9 mm), which is what the
"_truncated" in their directory names means. Storing 2560 samples would be
storing 1700 zeros. The 0.99991-vs-0.99669 gap says the decimation itself loses
almost nothing and the visible residual is *linear interpolation at the lower
rate* in the beamformer, not lost information.

**Decimating by 4 instead is measurably worse** — 0.99662 by the same
re-sampled measure, 4× the error of ×3. `Trans.Bandwidth` is 4–12 MHz, i.e.
±4 MHz about the 7.813 MHz carrier; ×3 leaves ±5.2 MHz, ×4 leaves ±3.9 MHz and
clips the band edges. The retired `benbarkow/us-ssl-mf` set used ×4.

For scale: NVIDIA's own two reconstructions of one frame score pixel-wise NCC
**0.197** against each other (§1.3). Everything above is far inside that.

### 10.2 Constants that differ from NVIDIA

| | NVIDIA | Stanford FSA |
|---|---|---|
| elements / pitch | 180 / 201.0 µm | 192 / 200.0 µm |
| carrier, λ | 6.5 MHz, 236.9 µm | 7.813 MHz, 197.1 µm |
| fast-time rate | 13.33 MHz | 10.42 MHz (31.25 ÷ 3) |
| depth | 0–40 mm | −0.97 to 20.18 mm axis, signal to ~19.9 mm |
| `t0` | 3 discrete levels | one value, **negative** |

⚠ **The `t0` row is the trap.** Stanford's time origin carries
`(TW.peak + 2*lensCorrection)·λ/c` = 1.77 µs ≈ 1.4 mm of depth. WI-4 forces
`t_peak = 0` for NVIDIA and that is correct there; doing it here misregisters
every pixel. The store's `t0_s` attribute already has the correction applied —
sample *n* is at two-way time `t0_s + n/fs_hz`.

### 10.3 Two things the source gives that we cannot yet use

1. **Per-animal ground truth exists.** `SoSDataAll.xlsx` publishes, for 20
   rats, ex vivo speed of sound (1511–1580 m/s), liver fat percentage,
   steatosis grade and lipidosis grade — this is the Telichko et al. 2022
   steatosis study. A **real, non-simulated sound-speed target** would be the
   most valuable single addition to §4.1.
2. **But the acquisition→animal join is missing.** figshare flattened the
   per-rat directory tree that the dataset's own `mainLoader.m` walks, so
   nothing in the flat file list names an animal. The corpus therefore records
   an *inferred* `session` — acquisitions clustered by timestamp, 30 sessions
   over 117 files — and the exact `source_file`. **Split on `session`; do not
   call it a subject id.**

### 10.4 The HVMux remap is now vendor-confirmed

`combine_rx_halves`'s cyclic element→channel map was ported from the phase-1
repo and verified against the MATLAB reorder expression. The figshare files
carry `Trans.HVMux.ApertureES`, a 192×65 table of the probe's own assignments,
and it is exactly `chan = (elem − 1) % 128 + 1` for every aperture — at
aperture 65, elements 65–128 read channels 65–128 and elements 185–192 read
channels 57–64. The remap is right, from the vendor's table rather than by
inference.

---

## 11. The crop-level anatomy cells are RUN (2026-08-17)

WI-13 reported three cells in `not_run` from the day it was written —
`contains_cyst`, `cyst_fraction_crop`, `sos_mean_crop`. **All six columns have
now been run with them** (K ∈ {1,4,16} × {train, validation}), so every WI-13
column carries all ten targets and `not_run` is down to the K=8 cut alone.
Results are the numbers in §1.5 and §3.2; the durable copies are the six
`tests/data/wi13_k*_results.json`, which the anatomy run **supersedes**.

```bash
bash scripts/wi13_anatomy_columns.sh      # all six columns, ~2 h on `data`
python scripts/wi13_anatomy_verify.py     # the reproduction gate, BEFORE promoting
```

Per-column settings are baked into the runner because they are not preferences:
`--crops-per-frame` is 32 for validation (93 frames) and 8 for train (830),
which is what every committed column used, and crop geometry comes from K via
`LoaderConfig.for_arm`. Changing either would change which crops each encoder
sees and silently invalidate the comparison below.

**Naming, deliberately.** The crop-level targets are suffixed `_crop`.
`cyst_fraction` in every table above is the *frame-level* quantity from
`frame_metadata.json`, and redefining that name would make old and new numbers
look comparable when they are not.

**Cyst probes drop the empty-`segmentation_map` frames** (PLAN.md §2.6: an
empty mask is an unknown mix of true negatives and unlabelled edge-touching
cysts). The mask is applied at the probe call site, not inside `probe`, and the
dropped count is reported in the log and in `results.json`. `sos_mean_crop`
keeps every crop — the sound-speed map is dense on those frames; only the
segmentation is missing.

### 11.1 The reproduction gate, and the one thing that did move

Adding targets must not perturb the seven that were already reported: same
crops, same seed, same checkpoints, same probes. So every pre-existing
`(encoder, target)` cell had to come back identical, and
`scripts/wi13_anatomy_verify.py` is that check.

**Result: all 238 shared cells reproduced `score` and `n_eff` bitwise.** The
anatomy targets are purely additive.

⚠ **One caveat, unexplained.** The two MAE encoders' bootstrap bounds
(`score_lo`/`score_hi`) moved on the two K=4 columns — by **≤ 3.6e-6**, against
interval widths of ~0.1, so five orders of magnitude below what any conclusion
rests on. No non-MAE encoder moved at all, and no `score` or `n_eff` moved
anywhere. The obvious causes were checked and **ruled out**: the MAE encoder is
bit-identical on repeat, invariant to batch chunking, and invariant to BLAS
thread count. The feature cache and staged store were wiped by a container
restart before this was chased further, and re-deriving costs a full NFS
re-read for no change in any reported number. `--rtol 1e-5` is what the gate
needs to pass; the default stays exact so this cannot be forgotten.

⚠ The gate is **pre-promotion only**. Now that the anatomy columns have been
copied into `tests/data/`, re-running it compares a file to its own copy and
passes for free — the script's docstring carries the git incantation to
reproduce the real check against `1147c19`.

### 11.2 Two things this cost, worth knowing

1. **The `data` instance has a 4 GB cgroup, not the 31 GB `free` reports.**
   Materializing all 6640 train crops is 3.5 GB and got OOM-killed. Check
   `/sys/fs/cgroup/memory.max`, not `free`. Crops are now materialized
   `--frame-chunk` frames at a time, and **origins are drawn for the whole
   corpus before any pixel is read** (`sample_store_origins`) — sampling per
   chunk would re-seed and silently give different crops. A test asserts a
   per-frame chunked walk reproduces the whole-corpus call exactly.
2. **`StoreCropBatch.__len__` read `images`.** Freeing them after feature
   extraction therefore made every per-target mask zero-length, and every probe
   was skipped with a PCA error rather than a mask error. It counts `frame_id`
   now.

Feature extraction on the `data` instance is ~3 min per 120 crops across seven
encoders on **one core** — the `24g` box does the whole column in minutes.
CLAUDE.md already maps results-matrix runs to `24g`; that is not a preference.

---

## 12. The qualitative pass — and why clustering is a null result (2026-08-17)

`scripts/wi13_qualitative.py` (+ `wi13_qualitative_columns.sh`) produces, for
K=4 on both splits: a PCA/t-SNE/UMAP **encoder grid** — one row per colour key,
one column per encoder, so the label is held fixed and the encoder varies —
each in a plain and a geometry-residualised variant, plus per-encoder k-means
(k ∈ {2,4,8,16}) and DBSCAN. Durable copies: `tests/data/wi13_k4_*_qual_results.json`
and `tests/data/wi13_k4_*_clusters.md`.

**The headline is negative, and it should be reported as a finding rather than
left as a table of small numbers: the learned representations have no discrete
cluster structure at all.**

| encoder | silhouette (k=4) | best ari_cyst | ari_depth_quartile (k=4) |
|---|---|---|---|
| vicreg_A | +0.031 | +0.131 | +0.124 |
| vicreg_B | +0.022 | +0.147 | +0.043 |
| mae_raw | +0.041 | +0.064 | +0.093 |
| random_resnet18 | +0.310 | +0.021 | +0.182 |
| pca_raw | **+0.658** | +0.030 | +0.002 |
| trivial | +0.475 | +0.036 | +0.193 |

⚠ **A high silhouette here is a bad sign, not a good one.** `pca_raw` and
`trivial` score 0.48–0.66 because they cluster on overall brightness — one
strong axis, cleanly separable, carrying no tissue. The VICReg arms sit at
0.02–0.05 and top out at ARI 0.147 against the cyst label. DBSCAN agrees:
7 clusters at **91–97 % noise** for the VICReg arms, 1 cluster for everything
else. The representation is a **continuous manifold that linear probes read
well and k-means cannot** — which is consistent with §3.2, not in tension with
it, but it does close off any "the model discovers tissue classes" reading.

**Geometry variance explained** (linear (z,x) regressed out of the features):

| | vicreg_A | vicreg_B | mae_raw | random_resnet18 |
|---|---|---|---|---|
| train | 7.1 % | 5.1 % | 3.0 % | **37.0 %** |
| validation | 7.9 % | 5.6 % | 3.6 % | **34.9 %** |

So Policy A spends ~7 % of its feature variance on where the crop came from,
against **35–37 % for the untrained ResNet-18** — a random CNN is largely a
position encoder, and training buys a five-fold reduction in that. This is the
one number that bears on §1.5's open question, and it is *suggestive but not
sufficient*: it bounds the linear geometry component of the features globally,
not the geometry component of the `contains_cyst` probe specifically. §8 item 3
is the probe that would actually settle it.
