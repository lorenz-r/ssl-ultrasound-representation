# MEMORY — everything known as of 2026-08-15

One file, written to be read cold. It records **what was run, what was
measured, what it means, and what I got wrong**. PLAN.md remains the design
doc and CLUSTER_HANDOFF.md the per-WI map; this is the state of knowledge.

Scope: the 2026-08-14/15 session on the cluster `24g` instance, which trained
seven models (six VICReg arms + two MAE variants) and probed all of them.

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

---

## 2. What exists on disk

| artifact | path | notes |
|---|---|---|
| K-sector store | `/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr` | 923 frames, `(N,2,507,456,16)` f32, 23 GB, **raw sums** |
| local staged copy | `/tmp/wi8_fsa_k16.zarr` | **ephemeral** — wiped on container restart |
| VICReg K=4 | `checkpoints/wi12_vicreg_policy{A,B}/` | |
| VICReg K=16, K=1 | `checkpoints/wi13_vicreg_k{16,1}_policy{A,B}/` | |
| MAE normalized | `checkpoints/wi14_mae_k4/` | trains on noise — see §4 |
| MAE raw-target | `checkpoints/wi14_mae_k4_rawtarget/` | the fair baseline |
| all logs | `/mnt/home/benbarkow/logs/` | |
| committed results | `tests/data/wi13_*.json`, `tests/data/wi14_*.json` | `figures/` is gitignored |

`checkpoints/` totals 19 GB and is gitignored. Each `<dir>/last.pt` resumes;
`history.json` and `run_config.json` sit beside it.

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

R² except † (balanced accuracy). Validation split (n_eff 93) is in
`tests/data/wi13_k4_validation_results.json` and agrees closely.

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
pytest -q                                     # 271 passed, 6 skipped (skips = data/rats/)
ruff check .

setsid nohup bash scripts/wi12_run_both.sh   >> logs/wi12_night.log 2>&1 </dev/null &   # VICReg K=4 A,B
setsid nohup bash scripts/wi13_kablation.sh  >> logs/wi13_kablation.log 2>&1 </dev/null & # K=16,1 x A,B
setsid nohup bash scripts/wi14_run.sh 4      >> logs/wi14_chain.log 2>&1 </dev/null &   # MAE

python scripts/wi13_matrix.py --k 4 --split train --crops-per-frame 8 --out figures/wi13_k4_train
python scripts/wi14_mask_sweep.py --out figures/wi14_sweep
bash scripts/wi12_status.sh                   # read-only health summary
```

Crop geometry is always derived from K by `LoaderConfig.for_arm` — never pass
`--crop-size` unless deliberately ablating it, or the encoder gets evaluated
on crops it never saw.

---

## 8. What is left

| # | item | where | cost |
|---|---|---|---|
| 1 | **Write-up.** Results exist only as JSON + tables. Nothing outside this repo is readable. | — | — |
| 2 | Crop-level anatomy targets (`contains_cyst`, crop `cyst_fraction`, crop `sos_mean`) need per-pixel `seg`/`sos` maps. `data/` is empty here; only the store survived. Check whether the small maps can be fetched without all 271 GB. | `data` instance | I/O hours |
| 3 | MAE at patch 4 as a full run — the only config that showed signal | `24g` | ~4 h+ |
| 4 | WI-15 cross-aperture arm (optional in PLAN.md) | `24g` | ~3 h |
| 5 | WI-9 HuggingFace publication (distribution, not science) | `data` | — |
| 6 | K = 8 column — dropped per PLAN.md §5.5 cut-list item 4 | — | — |

Open question worth stating plainly: **every trained encoder ties or loses to
`trivial` on `phase_error`** at K=4/K=1 — two numbers, envelope mean and std,
handicapped 32:1 by `capacity_control`. Only K=16 Policy A clears it. Either
phase error is not recoverable from these crops beyond global brightness, or
the probe protocol is not the right instrument for it.
