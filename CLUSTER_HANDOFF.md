# Cluster hand-off — status as of this session

Written for whoever (human or agent) picks this up on the CUDA cluster next.
**Start with [MEMORY.md](MEMORY.md)** — one file with every measured result
from the training/eval session, what they mean, and what was got wrong along
the way. Then **[PLAN.md](PLAN.md)** — it is the actual design doc, with the
full reasoning, measured numbers, and revision history. This file is a map of
what is built and what is left, in PLAN.md's own WI numbering.

## Why this file exists

Written across two phases. **Phase 1 (CPU-only dev laptop)** implemented and
tested every work item that does *not* strictly require a CUDA GPU or the
271 GB / 61 GB downloads — that machine had no `nvidia-smi`, so GPU-bound and
full-corpus work was scoped out deliberately, not skipped by accident.
**Phase 2 (cluster, TITAN RTX `24g`)** did the GPU and full-corpus work:
WI-4's real-data gate (which had to be *rewritten* — see below) and WI-8's
full 923-frame build, which now exists at
`/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr`. **271 tests pass, `ruff
check .` is clean, `pip install -e .` reproduces the environment** — no
Poetry needed on the cluster; `pyproject.toml` declares its metadata in the
standard PEP 621 `[project]` table, so plain `pip` builds it correctly
(`poetry-core` is only a build-time dependency, fetched automatically by
pip's build isolation — the `poetry` CLI itself is never required). Run both
before doing anything else, to confirm the baseline still holds on the
cluster:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check .
```

(`requirements.txt` at repo root mirrors the same dependency list, unpinned,
if you ever need `pip install -r requirements.txt` instead — e.g. into an
environment you don't want the local package installed into.)

`zea` picks a Keras backend at import time; this project defaults to
`KERAS_BACKEND=torch` (`tests/conftest.py`) so it works with zero extra
installs. **On the cluster, prefer `KERAS_BACKEND=jax` in a separate
virtualenv for the offline beamforming step**, per PLAN.md §0.6 — torch is
still what training uses, and mixing JAX-CUDA and torch-CUDA in one env is a
known source of conflicts. Nothing here hardcodes `torch` as the backend;
it's only the *default* when `KERAS_BACKEND` is unset. A second venv follows
the same pattern (note: as actually built, both venvs live under `/opt/venvs`
rather than in the repo — see "Env setup done this session" below for why):

```bash
python3 -m venv .venv-zea
source .venv-zea/bin/activate
pip install -e .
pip install "jax[cuda12]"   # not a zea dependency; must be added separately
```

---

## Which cluster instance for which step

Three instance types are available here: `data` (CPU only — dataset/env
management), `12g` (12 GB VRAM), `24g` (24 GB VRAM). The beamforming rows are
now **measured** (WI-4 and WI-8 both ran on a TITAN RTX `24g`); the training
rows are still a judgement call from what each step computes — fall back to
`24g` if something OOMs on a smaller instance.

| Instance | Use it for | Why |
|---|---|---|
| `data` | `pip install -e ".[dev]"`, `pytest -q`, `ruff check .`; downloading the NVIDIA frames, Stanford `.mat` files, and any HuggingFace assets (incl. the `facebook/dinov3-*`/`dinov2` weights once the gated request clears); the I/O-bound parts of WI-8's resumable build; submitting the `facebook/dinov3-*` gated-access request itself | Nothing here touches a GPU — running it on `12g`/`24g` would just burn GPU-instance time on CPU work. |
| `12g` | Any *re*-build or extension of the K-sector store (`scripts/wi8_build.py`) and re-runs of `scripts/wi4_reference_validation.py` / `wi8_verify.py` (`KERAS_BACKEND=jax`, 507×456). Both are done as of now; measured ~3.25 s/frame, so these are download-bound, not compute-bound. | Beamforming is inference-only delay-and-sum over a fixed pixel grid — no gradients, no optimizer state, no large activation stacks — so 12 GB should comfortably cover it. Try this before reaching for `24g`. Confirmed: WI-4's full-resolution run used well under 12 GB on a 24 GB card. |
| `24g` | WI-12 VICReg training, WI-14's MAE baseline, WI-15's cross-aperture arm, and WI-13's results-matrix runs (feature extraction, incl. any DINOv3/DINOv2 baselines) | Training holds activations, gradients, and optimizer state at once, and VICReg's variance/covariance terms are computed per-batch, so batch size (and headroom for it) matters — give training runs the bigger card. |

If you're unsure which bucket a new script falls into: does it call
`.backward()` / an optimizer step? → GPU training tier (`24g`, or `12g` if
it's small). Is it pure forward-pass signal processing (beamforming,
inference-only feature extraction)? → `12g` first. Is it neither? → `data`.

---

## Running an unattended multi-hour job on `24g`

Learned the hard way on 2026-08-14, when the WI-12 run lost ~2 h to a reap.
Each of these is cheap to get right up front and expensive to rediscover.

**Launch detached, always:**

```bash
setsid nohup bash scripts/wi12_run_both.sh \
  >> /mnt/home/benbarkow/logs/wi12_night.log 2>&1 < /dev/null &
```

A job started as a plain background child of an interactive (or agent) shell
dies *with that shell* — silently, mid-epoch, no traceback, `oom_kill 0`. That
is what killed WI-12 arm A at epoch 48. The GPU then sat idle and the
orchestrator reaped the whole instance 1h45m later, which wiped `/tmp`.

- **Don't run `pytest -q`, or anything CPU-heavy, while a job is training.**
  This box has **4 cores** and the loader runs 4 workers; the suite starves
  them and GPU util drops to 0%, which is indistinguishable from a hung run.
  Run the suite on a `data` instance instead.
- **`/tmp` does not survive a container restart**, so the local copy of the
  23 GB store must be re-staged after every reap (`wi12_run_both.sh` does it
  automatically). Budget ~7 min: NFS read measured **58 MiB/s under
  contention**, not the 241 MB/s quoted elsewhere. Plan with the slow number.
- **Nothing inside the container can restart a reaped job** — a watchdog dies
  with the container it watches. Checkpoints are on the NFS home, so recovery
  is manual but cheap: re-run the launch command, and each arm resumes from
  `last.pt`.
- **To hold VRAM near the cap without paying throughput, run
  `scripts/wi12_ballast.py`** alongside training. It allocates idle tensors in
  a separate CUDA context and sleeps, so it uses no SMs — measured 1539–1672
  pairs/s with it against 1520–1726 without, i.e. free. Useful if the
  orchestrator's idle metric is memory rather than utilization.
  - It is capped at `TOTAL - 10 GB` deliberately. A ballast that merely fills
    *free* memory expands into arm A's footprint the moment A exits, and then
    OOMs arm B, which allocates its whole pool at once. The fixed reservation
    holds that space open regardless of what is alive.
  - Do **not** use `--cache-frames` for this — 32 and 48 both benchmarked
    *slower* than 24 (see `scripts/wi12_run_both.sh`).

⚠ **`CLAUDE.md` is gitignored in this repo** (`.gitignore:6`), so it is
local-only scratch. Anything a future session must not lose belongs here or in
PLAN.md, not there.

---

## Done, with tests (safe to build on top of)

| WI | What | Where | Tests |
|---|---|---|---|
| WI-1 | Repo cut over, new `pyproject.toml`, CI, ported keepers | `pyproject.toml`, `.github/workflows/ci.yml`, `src/ssl_us/train/common.py`, `src/ssl_us/utils/monitoring.py` | — |
| WI-2 | Eval harness (pre-existing, not touched this session) | `src/ssl_us/eval/*`, `src/ssl_us/data/{crops,splits}.py` | `test_crops.py`, `test_guards.py`, `test_probes.py` (34) |
| WI-3 | Sector + occupancy geometry | `src/ssl_us/beamform/sectors.py` | `test_sectors.py` (18) |
| WI-4 | `AngularSectorSum` beamformer + speckle-invariant validation gate (rev-5; real-data gate now **passes 8/8 frames**) | `src/ssl_us/beamform/{sector_sum,image_quality}.py`, `scripts/wi4_reference_validation.py` | `test_sector_sum.py` (6), `test_zea_delays.py` (3), `test_image_quality.py` (21), `test_wi4_reference_validation.py` (6) |
| WI-5 | K-ladder reduction + occupancy normalization | `src/ssl_us/data/transforms.py` | `test_transforms.py` (15) |
| WI-6 | Resumable build pipeline (+ pluggable `write_frame`, used by WI-8's zarr store) | `src/ssl_us/beamform/build.py` | `test_build.py` (7) |
| WI-7 | Dataset-level RMS normalization | `src/ssl_us/data/normalization.py` | `test_normalization.py` (7) |
| WI-11 | Augmentation Policies A & B | `src/ssl_us/data/augment.py` | `test_augment.py` (17) |
| WI-12 | VICReg model, training loop, §4.7 instrumentation | `src/ssl_us/models/{resnet,vicreg}.py`, `src/ssl_us/train/{vicreg,instrument}.py`, `src/ssl_us/train/README.md` | `test_models.py` (12), `test_instrument.py` (10), `test_train_vicreg.py` (7), `test_wi12_eval_integration.py` (1) |
| WI-14 | Spatial MAE baseline — **trained and probed**, both target variants | `src/ssl_us/models/mae2d.py`, `src/ssl_us/train/mae.py`, `scripts/wi14_{train,run,mask_sweep}.*`, results in `tests/data/wi14_*.json` | `test_mae2d.py` (17), `test_train_mae.py` (4) |
| WI-15 | Cross-aperture prediction arm (optional in the plan) | `src/ssl_us/models/xaperture.py` | `test_xaperture.py` (15) |
| WI-16 | Stanford `.mat` reader + HVMux remap | `src/ssl_us/beamform/stanford_mat.py` | `test_stanford_mat.py` (12, run against the real files in `data/rats/`) |
| WI-8 | Full K-sector store: per-frame beamforming, zarr store + reader, resumable build — **923/923 frames built** | `src/ssl_us/beamform/{frame,store}.py`, `scripts/wi8_{build,verify,profile}.py` | `test_store.py` (11), `test_wi8_build.py` (6) |
| WI-11/12 | Augmentation policies (**batched torch, GPU**) + store-backed crop loader with frame-cached shuffle buffer | `src/ssl_us/data/{augment,dataset}.py` | `test_augment.py` (24), `test_dataset.py` (12), `test_transforms.py` (18) |
| WI-10 | Splits + leakage guard, **verified on the real 923-frame store**; per-run split-audit figure | `src/ssl_us/data/splits.py`, `src/ssl_us/eval/viz.py` (`plot_split_audit`), `scripts/wi10_splits.py` | `test_wi10_splits.py` (12), `test_guards.py` |
| WI-13 | Results matrix, **K ∈ {1,4,16} × {A,B}** — store-backed crops, frozen WI-12 encoders, A-vs-B contrast | `src/ssl_us/eval/store_features.py`, `scripts/wi13_matrix.py`, results in `tests/data/wi13_k{1,4,16}_{train,validation}_results.json` | `test_store_features.py` (5) |
| WI-0 | (pre-existing) NVIDIA frame verification | `scripts/wi0_probe.py` | `test_nv_frame.py` (8) |

Every *test* above runs without a GPU or the full corpus — they use synthetic
data, a small real download (a couple of NVIDIA frames), or the real local
Stanford `.mat` files in `data/rats/`. ⚠ **`data/rats/` is not present on the
cluster** (it was local to the phase-1 laptop), so `pytest -q` here reports
**271 passed, 6 skipped** — those 6 are `test_stanford_mat.py` and they skip
cleanly. Re-download the Stanford `.mat` files before WI-16 work. The two exceptions are the artefacts
rather than the tests: WI-4's gate numbers and WI-8's 923-frame store were
both produced on a `24g` GPU instance against the full corpus.

**Two phase-1 bugs were found and fixed along the way**, worth knowing about
(the three WI-8 build traps are listed under WI-8):
1. `.gitignore`'s bare `data/` pattern was also matching `tests/data/`,
   silently excluding `tests/data/wi0_results.json` from any future commit.
   Fixed to `/data/` (anchored to repo root).
2. zea's `Parameters.t_peak` defaults to `1/center_frequency` when not
   supplied; the OpenH-RF FSA scan dict (PLAN.md §0.2) doesn't set it.
   Forcing `t_peak=0` measurably improved WI-4's real-data calibration (see
   below). `scripts/wi4_reference_validation.py` uses the fixed value now.

---

## Not done — what's actually left, and why

(WI-4 and WI-8 are both now **done**; their entries are kept at the top of
this section because the *reasons* — the original WI-4 gate being
unachievable, and the three traps in the WI-8 build — are what a future reader
most needs to know before touching the beamformer or rebuilding the store.)

### WI-4 — DONE. Gate passes on 8/8 frames — but the *gate itself* was rewritten

`AngularSectorSum` was already verified algebraically (bit-exact against zea's
`DelayAndSum` on synthetic data, `tests/test_sector_sum.py`). The real-data
gate now passes too — **after establishing that the originally specified gate
was impossible and replacing it.** Do not "restore" the old thresholds.

**The original gate (pixel-wise NCC ≥ 0.98 vs `bmode`) is unachievable by any
beamformer, including NVIDIA's own.** Measured, not argued:

| Evidence | Result |
|---|---|
| NVIDIA's own two shipped reconstructions (`bmode` vs `bmode_focused`) scored against **each other**, 10 frames | mean NCC **0.197**, range −0.071 … 0.510 (frame 1 is *anti*-correlated) |
| Our `AngularSectorSum` vs `bmode` | 0.2573 — i.e. **better than the reference's own sibling** |
| zea's stock `DelayAndSum` (what OpenH-RF's `reconstruct.py` uses) vs `bmode` | 0.2587 — two independent beamformers agreeing to 0.001 |
| Same, but beamformed with the ground-truth SoS map | 0.134 — *worse*, confirming `bmode` is the constant-c DAS its converter claims |

Numbers in `tests/data/wi4_reference_selfconsistency.json`. The cause is
physical: fully-developed speckle is a random interference pattern, so
pixel-wise correlation measures agreement of *speckle realisation*, which a
correct beamformer needn't reproduce. The old docstring's "residual
sub-wavelength timing bug" hypothesis was chasing that artifact — though the
two timing bugs it did find (`t_peak`, `t0` sign) are real and remain fixed; a
16-way sweep over {`t_peak`, `t0` sign, near-field mute, tx/rx axis order}
confirms the current settings beat all alternatives.

**The gate now** (PLAN.md WI-4, rev-5; metrics in
`src/ssl_us/beamform/image_quality.py`, 21 tests in `tests/test_image_quality.py`):
gCNR within 0.10 of the reference's own per-frame gCNR; per-patch speckle SNR
in [1.60, 2.15] and within 0.25 of the reference's; negative control must
degrade gCNR. All reference-relative, because frame difficulty varies hugely
(reference gCNR spans 0.15–0.86).

**Result: 8/8 frames pass**, with our gCNR tracking the reference to within
~0.01–0.03 across that whole range, and the negative control collapsing
decisively (e.g. 0.85 → 0.076) rather than the old near-tie. The winning
variant is **f# = 1.5** on 7 of 8 frames (f# = 1.0 on frame 0), so that sets
`θ_max` for the WI-8 build.

Reproduce (ran on `24g`; `12g` is likely fine — each variant is ~3 s of GPU):

```bash
source /opt/venvs/.venv-zea/bin/activate
KERAS_BACKEND=jax N_FRAMES=8 GRID=507 python scripts/wi4_reference_validation.py
```

Two measurement details in that script are load-bearing and easy to
accidentally undo: speckle SNR must be computed **per patch** (attenuation/TGC
vary the mean with depth, dragging a pooled ratio to 1.41–1.69 vs ~1.89
patched), and the inclusion must be **excised** from the background region
(0.93 on frame 1 otherwise). Some frames have an empty segmentation map and
run the speckle checks only.

### WI-4b — plane-wave stack: **stays cut, don't rebuild it**

PLAN.md §10.3(b) measured the reciprocity residual (16.6% off-diagonal) and
*overrode* building the PW(5) stack anyway — judged mostly numerical, not
physical, and not worth 61.5 GB vs 27.3 GB. `augment.py`'s
`transmit_angle_pair` and `xaperture.py`'s `mask_axis_tx` both raise
`NotImplementedError` on purpose, citing this decision. Don't "fix" those by
building the PW stack without re-reading §10.3(b) and deciding the override
was wrong — that's a judgement call, not a bug.

### WI-8 — **DONE. 923/923 frames built and verified.**

The full store exists and meets every clause of PLAN.md WI-8's done-condition:

| Done-when clause | Result |
|---|---|
| 923/923 frames present | **923/923**, manifest complete |
| Manifest complete | `wi8_fsa_k16.zarr.build/manifest.json`, both shards closed |
| WI-4's gate re-run on 10 frames *from the finished store* | **10/10 pass** (`tests/data/wi8_store_verification.json`) |
| Depth-RMS profile plotted for the whole corpus | `figures/wi8/depth_rms_profile.png`, **7.6 dB** decay 5-10 → 30-38 mm |
| (extra) every frame finite and non-empty | **923/923** |

**Where it is:** `/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr` (23 GB, on the
persistent NFS home; verified frame-identical to the build output, which was
written to ephemeral `/tmp`). A 25 GB probe write to home succeeded, so the
quota has room. Per-frame labels — `t0`, `phase_error`, `sos_mean`, `sos_std`,
`cyst_fraction`, `segmentation_empty` — are in `frame_metadata.json` beside it.

**Build took ~2 h wall**, matching the revised estimate: ~3.25 s/frame GPU in
steady state, the rest shard download. Two independent cross-checks that the
store is really right, beyond the gate:

- The metadata reports **exactly 75 frames with an empty `segmentation_map`**,
  matching PLAN.md §2's independently measured "75 of 923 (8.1 %)" — so frame
  ordering and label extraction both line up with NVIDIA's own indexing.
- The per-frame RMS histogram is visibly **trimodal**, consistent with §2.3's
  finding that `t0` takes exactly three values across the corpus (435/116/372).

The depth-RMS profile is **measured and stored, never divided out** — PLAN.md
§1.3's warning about the old repo's `normalize_frame` deleting the attenuation
signal that §4.5c exists to probe.

WI-9 (HF publication), WI-13 (results matrix) and WI-12 (VICReg training) are
all unblocked by this. **WI-10 is now done** — see below.

### WI-10 — DONE. Splits verified against the real corpus.

`splits.py` + the leakage guard already existed from WI-2; what WI-10 needed
was running them on the real store and emitting the audit figure. Both done
(`scripts/wi10_splits.py` → `tests/data/wi10_splits.json`,
`figures/wi10/split_audit.png`):

- Outer 830 train / 93 validation = 923; inner probe split 664/166 carved from
  train only; guard passes at both levels; nothing straddles.
- **PLAN.md §8's distributional-matching claim re-verified from the built
  store** — `sos_mean` Welch t = 0.60, `phase_error` t = −0.02, both far inside
  the |t| < 2 bar, and both reproducing the planning-time means to within a
  fraction of a standard deviation. This is a real cross-check on WI-8's frame
  ordering, not a tautology: the planning numbers came from the source dataset,
  these from the store.
- Empty-`segmentation_map` frames split **63 train / 12 validation** — matching
  §2's independently measured counts exactly.
- ⚠ `cyst_fraction` is the least-matched axis (t = 1.86). Under the bar, but
  the closest to it — worth remembering if a cyst probe ever looks strange.

`scripts/wi10_splits.py` **exits non-zero** if the matching ever stops holding,
so it doubles as a regression check when the corpus changes.

#### How it was built (kept for whoever rebuilds or extends it)

The missing pieces are written and tested end-to-end against real data:

| Piece | Where |
|---|---|
| Per-frame beamforming (the `beamform_frame` callback) | `src/ssl_us/beamform/frame.py` — WI-4's validated constants live here as named constants |
| Zarr store, PLAN.md §3.7 layout + the **store reader WI-13 needs** | `src/ssl_us/beamform/store.py` (`open_store`) |
| Build entry point, resumable | `scripts/wi8_build.py` |
| Acceptance check (WI-4's gate re-run *from the store*) | `scripts/wi8_verify.py` |

Tests: `test_store.py` (11), `test_wi8_build.py` (6). `build_store` gained an
optional `write_frame` argument so the real zarr build runs through WI-6's
*same* tested resumability code path rather than a parallel copy of it.

Resume was verified against the real store (4.9 s, no download, no
beamforming), and the shard-based and streaming source paths were checked
against each other and produce **bit-identical** frames.

**Three traps found by running it. All three are fixed in the code, and all
three are the kind that come back if someone "simplifies" it:**

1. **Never use `datasets` streaming for this corpus — it is the bottleneck,
   not the network.** Measured on this cluster:

   | Path | Throughput | 271 GB corpus |
   |---|---|---|
   | `load_dataset(..., streaming=True)` | ~2.7 MB/s | ~28 h |
   | `hf_hub_download` of the same shards | **68–76 MB/s** | **~1 h** |

   So the build downloads one parquet shard at a time, decodes it locally
   through `datasets`, beamforms its frames, then deletes it (`--source shard`,
   the default; `--source stream` keeps the old path as a fallback). Disk stays
   bounded at ~one shard rather than 271 GB. **The two paths were verified to
   produce bit-identical frames and metadata**, so this is purely a speedup.
   GPU cost was never the issue and matches §5.4 well: **~3.2 s/frame** once
   data is local. Realistic full-build estimate is now **~2 h**, not 6–10 h.

   (`hf_transfer` is a red herring here — it is deprecated and made no
   difference in the benchmark, 68 vs 76 MB/s; `huggingface_hub` 1.27 already
   uses Xet. `HF_XET_HIGH_PERFORMANCE=1` is the current knob if you want one.)
2. **A blosc deadlock will wedge the build if you undo one line.**
   `numcodecs.blosc.use_threads = False` in `store.py` is load-bearing: with
   blosc's internal thread pool live alongside HuggingFace's fsspec/aiohttp
   threads, a chunk write blocked forever in uninterruptible I/O — observed
   directly (all 3 smoke frames beamformed, only 2 reached the manifest, then
   the process hung in `D` state and had to be `kill -9`'d). The build also
   writes the label sidecar every frame rather than once at the end, and
   `os._exit`s after the final flush, so a hung shutdown cannot lose work.
3. **Deleting a downloaded shard does not free it, unless you download it
   right.** `hf_hub_download`'s default cache returns a **symlink into
   `blobs/`**; unlinking that frees nothing, so "download shard, use it, delete
   it" silently accumulates the entire 271 GB. Caught mid-launch: the HF cache
   sits on the 32 GB overlay and went 537 MB → 7.0 GB on the *first* shard.
   The build therefore downloads with `local_dir=` into `--scratch` (a real
   file that can actually be deleted) and clears the arrow cache too. Verified
   across shard boundaries: `/` stays flat instead of growing per shard.

**Storage, measured:** ~26 MB/frame compressed ⇒ **23 GB** for 923 frames
(29.6 MB/frame uncompressed, per §3.7). `/tmp` on these instances is
**ephemeral** — its contents were empty at container start and do not survive
an instance change — so the build wrote there for speed and the result was
copied to the NFS home, then checked frame-identical. Home is quota'd but has
room: a **25 GB** probe write succeeded with home already at 50 GB (33 GB of
which is `new_arch_custom.sqsh`).

Run it:

```bash
source /opt/venvs/.venv-zea/bin/activate
# Full build: resumable, deletes each shard after use, ~2 h expected.
KERAS_BACKEND=jax python scripts/wi8_build.py --out <store>.zarr
# WI-8's done-condition: WI-4's gate, re-run on frames read back from the store.
KERAS_BACKEND=jax python scripts/wi8_verify.py --store <store>.zarr --n 10
```

Re-running the same build command after a crash or preemption resumes from the
manifest at frame granularity and re-beamforms nothing.

Settings the WI-4 run pins down for it:

- **`f_number = 1.5`** — the winning aperture on 7 of 8 frames (f#=1.0 won only
  on frame 0, whose inclusion is near-invisible in the reference too). This
  sets `θ_max`.
- **`t_peak = 0`** (not zea's `1/center_frequency` default) and **`t0` used
  with its stored sign**; `MUTE_SAMPLES = 32`. All four were swept; these beat
  every alternative combination.
- Axis order: treating raw axis 0 as transmit is fine — measured immaterial
  (0.2587 vs 0.2573), as PLAN.md §10.2 predicted from reciprocity.

**Caches must be pointed off the small overlay.** `/` on these instances is a
32 GB `fuse-overlayfs`, and both the HF hub cache (`~/.cache/huggingface`) and
the `datasets` arrow cache default onto it. The build as run set:

```bash
export HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache
KERAS_BACKEND=jax python scripts/wi8_build.py \
    --out /tmp/wi8_full.zarr --scratch /tmp/wi8_shards
```

With that, `/` stayed flat at 537 MB for the whole 923-frame run.

### WI-12 — **DONE. Both arms trained to completion 2026-08-14.**

Ran on `24g` via `scripts/wi12_run_both.sh`. Checkpoints in
`/mnt/home/benbarkow/checkpoints/wi12_vicreg_policy{A,B}/` (`last.pt` +
every-25-epoch snapshots), per-epoch metrics in `history.json` beside them,
logs in `/mnt/home/benbarkow/logs/`. Both arms ran the full 223 epochs —
derived from PLAN.md §5.4's 11.8M-pair budget against the verified 830-frame
train split at 207 steps/epoch, **not** the plan's stale 115.

| | Policy A | Policy B |
|---|---|---|
| status | completed, 1.53 h | completed, 1.46 h |
| pairs | 11,817,216 | 11,817,216 |
| loss | 21.32 → **8.33** | 21.17 → **7.82** |
| RankMe | 286.1 → **287.1** | 298.0 → **304.5** |
| invariance | 0.096 → 0.019 | 0.081 → 0.006 |
| variance | 0.692 → 0.100 | 0.690 → 0.084 |
| covariance | 1.628 → 5.346 | 1.896 → 5.575 |
| mean throughput | 1646 pairs/s | 2110 pairs/s |

**Neither arm collapsed**, and the evidence is stronger than "the detector
didn't fire": RankMe *ended above where it started* in both, so the embedding
occupied more dimensions at convergence than at init. Both traced the same
shape — a dip over the first ~40 epochs (A bottomed near 263, B near 193) then
a full recovery — so an early RankMe decline in a rerun is not by itself cause
to intervene. Final RankMe is flat to ±1 over the last 8 epochs of each.

At 1.5 h/arm this came in well under §5.4's 4–6 h/run budget. The throughput
table further down is the older crop-shipping measurement, kept only for the
progression it shows.

> ⚠ **The epoch-50 landmine, fixed 2026-08-14 — don't reintroduce it.**
> `_is_collapsing` compared a window against its own tail with
> `zip(..., strict=True)`. That pairing is ragged by construction (`n` values →
> `n-1` comparisons), so the function could never return a verdict — it raised
> `ValueError` on *every* call that got past its length guard. Gated behind
> `collapse_grace_epochs = 50`, it therefore killed each arm at exactly epoch
> 50. It had no test coverage; `tests/test_train_vicreg.py` now has four.
> Ruff's `B905` requires an explicit `strict=`, which is the likely origin of
> the wrong value — the correct one here is `strict=False`.
>
> A second bug hid the first: the collapse halt exited **1**, and Python also
> exits 1 on any uncaught exception, so `wi12_run_both.sh`'s "rc=1 → don't
> retry" rule logged a hard crash as `HALTED (collapse)` and skipped the arm
> instead of resuming it. The halt now exits **3**; keep those codes distinct.

`src/ssl_us/data/dataset.py` reads the WI-8 store and feeds VICReg. Two things
about it are load-bearing and were driven by measurement, not taste:

1. **Frame-cached shuffle buffer.** The store chunks one *frame* per zarr chunk
   (§3.7), so any single crop costs a full-frame decompress (~30 ms). The cache
   amortises that across `crops_per_frame` crops **and** holds `cache_frames`
   frames resident at once, drawing each crop from a random resident frame.
   The second part is correctness, not polish: VICReg's variance/covariance
   terms are computed *across the batch*, so a batch drawn from one frame has
   collapsed batch statistics.
2. **Augmentation runs on the GPU, and there is only one implementation of it.**
   `augment.py` is batched torch (it runs on CPU tensors too, so tests and the
   WI-11 figure need no GPU). A NumPy copy "for tests" was deliberately *not*
   kept — it would be a second definition of the semantics, free to drift from
   the one that trains the model.

Measured on the `24g` instance (4 CPU cores — that count is the whole reason
for the GPU split):

| Loader | pairs/s | 11.8M pairs (400 ep × 115 steps × 256) |
|---|---|---|
| Naive crop-indexed | ~17 | ~190 h |
| + frame cache | 28 | 117 h |
| + numeric fixes (2.08×) | 59 | 55 h |
| + 4 DataLoader workers | 157 | 21 h |
| **+ GPU augmentation** | **588 (A) / 442 (B)** | **5.6 h / 7.4 h** |

PLAN.md §5.4 budgets 4–6 h/run, so Policy A is in band and Policy B is a little
over. Peak GPU memory for the data path is 2.0 GB, leaving plenty beside the
model. The remaining lever, if it ever matters: the loader ships the occupancy
crop alongside the sums (~1/3 of PCIe traffic) even though occupancy is
frame-invariant and could be gathered device-side from a single resident copy.

Usage — `train_vicreg` takes any iterable of `(view1, view2)` batches:

```python
cfg = LoaderConfig.for_arm(store, k=4, policy="A", frame_ids=splits["train"])
loader = make_dataloader(cfg, batch_size=256, num_workers=4)
for x1, x2 in GpuAugmentor(cfg, device="cuda").wrap(loader):
    ...
```

`LoaderConfig.for_arm` encodes PLAN.md §3.2c/d's two internally-consistent
arms (`K ≤ 4 @ 128` full depth, `K ≥ 8 @ 64` with z ≥ 8 mm) so K and crop size
cannot drift apart.

**K-jitter is OFF by default — decided, not merely defaulted.** Policy A's
priority-1 augmentation is listed in PLAN.md as "random sector subset *and*
random K-reduction level". The sector-subset half is on; the K-reduction half
is off, because enabling it confounds the experiment this store exists for:
WI-13's K-ablation trains a separate model per arm to ask whether finer receive
angular resolution helps, and jittering K trains the K=4 arm to be *invariant*
to angular resolution — discarding exactly what distinguishes it from K=1, so
the ablation would report "K doesn't matter" as an artifact of the augmentation.
The arms are not even augmented comparably (at K=1 jitter is a no-op).

Policy A loses nothing essential: the random contiguous sector subset still
supplies the intended positive pair while leaving surviving sectors at full
angular resolution. Run "Policy A + k-jitter" as a documented ablation cell
instead — `LoaderConfig(k_jitter=True)`. Full reasoning in PLAN.md WI-11
("K-jitter: off by default"); `test_dataset.py` pins the default so a future
flip has to be deliberate.

### WI-9 — HuggingFace publication

Needs WI-8's output. Not started — no code written for the dataset card or
the smoke-subset extraction. PLAN.md §5.2 WI-9 lists exactly what the card
must contain (attribution, license, changes-made statement, etc.).

### WI-13 — DONE for the reduced ladder: K ∈ {1, 4, 16} × {A, B}, all probed

Six trained arms, all six probed on both splits. Per-column results are
committed at `tests/data/wi13_k{1,4,16}_{train,validation}_results.json`
(`figures/` is gitignored, so the JSON is the durable artifact). Re-run a
column with:

```bash
python scripts/wi13_matrix.py --k 16 --split train --crops-per-frame 8 \
  --out figures/wi13_k16_train
```

Crop geometry is derived from K by `LoaderConfig.for_arm` — never passed as a
flag — so an encoder cannot be evaluated on crops it was not trained on.

#### ⚠ Read the columns down, not across

**K is confounded with crop geometry by design.** §3.2c/d ties K ≤ 4 to 128 px
at full depth and K ≥ 8 to 64 px below z = 8 mm, because those are the
*internally consistent* arms. So a K=16 cell differs from a K=4 cell in three
ways at once (sector count, crop size, depth range), and cross-K differences
cannot be attributed to K. The controls move too — `trivial` scores +0.400 on
`phase_error` at K=4 but only +0.114 at K=16, because global brightness stops
predicting phase error once crops are small and deep.

Each column is therefore scored **against its own controls**. Numbers are the
830-frame train split; ✓/✗ is best-arm minus best-control.

| target | K=1 A | K=1 B | vs ctrl | K=4 A | K=4 B | vs ctrl | K=16 A | K=16 B | vs ctrl |
|---|---|---|---|---|---|---|---|---|---|
| phase_error | +0.378 | +0.399 | ✓ +0.04 | +0.366 | +0.438 | ✓ +0.04 | **+0.377** | +0.247 | ✓ **+0.21** |
| t0_group † | +0.407 | +0.495 | ✓ +0.12 | +0.398 | **+0.526** | ✓ +0.10 | +0.390 | +0.466 | ✓ +0.04 |
| sos_mean_frame | +0.918 | +0.876 | ✓ +0.14 | +0.899 | +0.865 | ✓ +0.10 | **+0.931** | +0.849 | ✓ **+0.38** |
| sos_std_frame | +0.294 | +0.322 | ✓ +0.08 | +0.277 | +0.339 | ✓ +0.06 | +0.335 | +0.162 | ✓ +0.19 |
| cyst_fraction | **+0.365** | +0.223 | ✓ +0.23 | +0.331 | +0.312 | ✓ +0.16 | +0.214 | +0.126 | ✓ +0.10 |
| z_centre | **+0.768** | +0.561 | ✓ +0.28 | +0.699 | +0.756 | ✓ +0.11 | +0.429 | +0.681 | ✓ +0.05 |
| x_centre | +0.528 | −0.000 | ✓ +0.49 | +0.537 | −0.001 | ✓ +0.36 | **+0.789** | −0.000 | ✓ **+0.65** |

R² except † (balanced accuracy). `pca_raw` is ≈0 everywhere and is omitted.
**Every column clears its own controls on every target** — but by margins
ranging from +0.04 to +0.65, and the small margins are the interesting ones.

#### Which arm wins, by 95 % CI separation

`-` means the intervals overlap, i.e. genuinely tied, not merely noisy.

| target | K=1 (830 / 93) | K=4 (830 / 93) | K=16 (830 / 93) |
|---|---|---|---|
| phase_error | − / − | − / − | **A** / − |
| t0_group | B / B | B / B | B / − |
| sos_mean_frame | A / − | A / − | A / A |
| sos_std_frame | − / − | − / − | A / − |
| cyst_fraction | A / − | − / − | − / − |
| z_centre | A / A | B / − | B / B |
| x_centre | A / A | A / A | A / A |

**The policies specialise, consistently and in opposite directions:**

- **Policy A owns spatial and anatomical structure** — it wins `x_centre` at
  every K without exception, `sos_mean_frame` at every K, and `cyst_fraction`
  where that separates. Policy B's `x_centre` R² is **−0.000 at all three K**:
  its features carry literally no horizontal position, which is the
  geometry-only augmentation doing exactly what it was specified to do.
- **Policy B owns acquisition metadata** — it wins `t0_group` at every K.

That split is the cleanest result in the matrix and it is stable across K,
across splits, and across the geometry confound.

#### Four things worth carrying forward

1. **Report separation, not ordering.** Rows marked `−` are tied even at
   n = 830, so their A-vs-B ordering is not a finding. The 93-frame validation
   split resolves far fewer rows; the 830-frame train split is the one to rank
   on (see the power note below).
2. **The `phase_error` control failure is a K=4/K=1 result, not a universal
   one.** At K=4 the two-number `trivial` control (+0.400) matches both arms,
   so the encoders have learned nothing about phase error beyond global
   brightness — that is §4.1's control working. **At K=16 it reverses**: the
   control collapses to +0.114 while Policy A holds +0.377, clearing by +0.21
   and separating from B. On small deep crops, brightness stops being a
   shortcut and the encoder is carrying real phase information. Do not quote
   the K=4 conclusion as though it holds for the ladder.
3. **`z_centre` inverts across the ladder** — A wins at K=1, B wins at K=4 and
   K=16. Do not read this as a K effect: the K=16 crops only span z ≥ 8 mm, so
   the target's own variance differs between columns.
4. **Geometry explains only ~7–8 %** of Policy A's feature variance, so the
   representation is not merely encoding where the crop came from.

#### Statistical power — use the train split for ranking

The 93-frame validation split is underpowered for the five frame-level
targets, and that, not the encoders, is what makes most cells look tied there.
The 830-frame train split narrows CIs from ±0.15 to ±0.05 and resolves roughly
twice as many rows. It is *in-sample* for the encoders (they saw those images,
never the labels — no leak, and standard SSL protocol), so it is reported
alongside the held-out numbers, never instead of them. **The two agree
closely** at K=4 — sos_mean_frame 0.899 vs 0.890, x_centre 0.537 vs 0.524 —
which is itself a result: no overfitting to the pre-training frames, so the
cheap high-powered split can be trusted for ranking.

#### Training cost, measured

| cell | hours | crop | final loss | RankMe start → end |
|---|---|---|---|---|
| K=4 A | 1.53 | 128 | 8.33 | 286.1 → 287.1 |
| K=4 B | 1.46 | 128 | 7.82 | 298.0 → 304.5 |
| K=16 A | 1.20 | 64 | 8.63 | 347.0 → 298.7 |
| K=16 B | 1.13 | 64 | 7.84 | 348.4 → 310.1 |
| K=1 A | 1.93 | 128 | 8.60 | 264.0 → 289.6 |
| K=1 B | 1.47 | 128 | 7.71 | 280.7 → 302.9 |

Two patterns across all six: **Policy B always converges to lower loss and
higher RankMe than A** — and it is *not* the better encoder for it, which is
worth remembering before using training loss to pick an arm. And every arm
lands in a narrow final RankMe band (289–310) regardless of start: K=16 comes
*down* from ~347, K=1 climbs *up* from ~264. A falling RankMe is therefore not
by itself a collapse signature; check where it started.

**Not run, with reasons** (also machine-readable under `not_run` in the JSON):
`K = 8` is dropped per PLAN.md §5.5's cut-list item 4. The crop-level
anatomy targets (`contains_cyst`, crop-level `cyst_fraction`, crop-level
`sos_mean`) need the per-pixel `seg`/`sos` maps, which live in the source
NVIDIA frames; `data/` is empty on this cluster and only the beamformed store
survived, so they cannot be run without re-downloading.

### WI-16 — Stanford, past the reader

The reader + HVMux remap is done and verified against all 6 real local
`.mat` files (`test_stanford_mat.py`). **Confirmed and worth knowing:** `zea`
0.1.4's Verasonics converter (`zea/data/convert/verasonics.py`, ~1700 lines)
has zero handling for aperture multiplexing — grepped for
"aperture"/"hvmux"/"multiplex", zero hits. So the old repo's cyclic remap
(`stanford_mat.combine_rx_halves`) is genuinely still needed; this isn't a
temporary workaround.

Not done: decoding Stanford's focused multi-focal transmits into multistatic
form via `zea.ops.Refocus` (PLAN.md §0.4), then running the same
`AngularSectorSum` pipeline on it, then the two grid-matching variants
(mm-matched vs λ-matched, PLAN.md §3.6). This needs real `zea.ops.Refocus`
API exploration (signature, inversion-method choice — adjoint/Tikhonov/
rSVD/tSVD) that this session didn't have time for. `combine_rx_halves`
hands back exactly the per-transmit 192-element complex IQ signal that
should feed into it.

### Manual action (not code)

**Submit the `facebook/dinov3-*` gated-access request on HuggingFace now**
(PLAN.md WI-1) — it's a human-approval queue, so the lead time doesn't block
WI-13 later only if requested early. Ungated fallback is
`facebook/dinov2-small`; `eval/features.py` doesn't wire up either yet
(baselines 5-8 in PLAN.md §4.4 — DINOv3/v2, HVAE, Speckle2Self, TAESD — are
all still unimplemented; only baselines 1-4, the from-scratch ones, exist).

---

## Env setup done this session — venvs live on local disk, not NFS

Both venvs from CLAUDE.md's cluster instructions are built and verified on
this `data` instance, but **at `/opt/venvs/.venv` and `/opt/venvs/.venv-zea`,
not `<repo>/.venv*`** as CLAUDE.md's literal commands would put them. Two
reasons, discovered this session:

1. **NFS quota.** `/mnt/home/benbarkow` (where the repo lives) is an NFS
   mount with a real quota — installing two venvs there (torch pulls in a
   full CUDA-enabled wheel, ~6 GB) hit `OSError: Disk quota exceeded`
   mid-install even after clearing space. `/` (`scratch-scratch`, local disk)
   had 1.6 TB free and no quota.
2. **Container bake.** This instance is running inside an enroot/pyxis
   container (`new_arch` shows up in the enroot mount path — matches this
   repo's git branch). If you're about to `enroot export`/snapshot this
   container into a new image, only the container's **local** filesystem
   (`/`, `/opt`, `/root`, etc.) gets captured — the NFS home mount is
   attached at container *start*, not baked in. Venvs on NFS would silently
   not exist in the new image.

Both venvs use an **editable install** (`pip install -e .`) pointing back at
`/mnt/home/benbarkow/ssl-ultrasound-representation` — i.e. they hold no copy
of `src/ssl_us`, just a path reference. This is fine and intentional as long
as future containers built from this image still get the same NFS home
mounted at the same path (standard for this cluster's per-user home setup) —
if that ever stops being true, `pip install -e .` needs to be re-run wherever
the repo actually ends up.

What's installed, and verified working:

| Venv | Command used | Backend check | Tests |
|---|---|---|---|
| `/opt/venvs/.venv` | `pip install -e ".[dev]"` | `KERAS_BACKEND=torch python -c "import torch, zea"` → torch 2.13.0+cu130 (`cuda.is_available()` correctly `False` here, no GPU on `data`) | `pytest -q` → 172 passed, 6 skipped (Stanford `.mat` tests — need `data/rats/`, not downloaded yet), `ruff check .` clean |
| `/opt/venvs/.venv-zea` | `pip install -e .` (no `[dev]` — dev tooling only lives in the torch venv) then `pip install "jax[cuda12]"` (not a `zea` dependency, has to be added separately; CUDA build chosen so this image is GPU-ready on `12g`/`24g` without a second install pass — same idea as torch's auto-selected CUDA wheel) | `KERAS_BACKEND=jax python -c "import zea"` → succeeds, `jax.devices()` → `[CpuDevice(id=0)]` (correct fallback, no GPU here) | not run (pytest not installed in this venv on purpose) |

**Housekeeping done along the way:** `/mnt/home/benbarkow` was almost out of
quota (49 GB used) mostly from three old `.sqsh` container images sitting at
the top level, not repo content. Deleted `us_ssl_training.sqsh` (11 GB) and
`us_ssl_mf_data_processor.sqsh` (11 GB) per explicit user confirmation;
**kept `us_ssl_training_final.sqsh` (16 GB)**.

**For whoever bakes the image:** after `enroot export` (or equivalent),
sanity-check the new image still has `/opt/venvs/.venv` and
`/opt/venvs/.venv-zea` intact and that activating each and re-running the
backend-check one-liners above still works, since the editable-install path
back to NFS is the one part of this setup that isn't self-contained in the
image.

## Quick sanity checklist before starting new work

- [ ] On a `data` instance: `pip install -e ".[dev]" && pytest -q` → 271 passed, 6 skipped
      (the 6 skips are `test_stanford_mat.py`, which needs `data/rats/*.mat`)
- [ ] `ruff check .` → clean
- [x] On a `12g`/`24g` instance: `nvidia-smi` works — confirmed, TITAN RTX
      24 GB, driver 580.105.08 / CUDA 13.0, `torch.cuda.is_available()` True
- [ ] Decide: one `KERAS_BACKEND=jax` venv for beamforming, torch venv for
      training (PLAN.md §0.6), or continue with the single-venv
      `KERAS_BACKEND=torch` setup this session used. Either works; the
      dependency is declared once in `pyproject.toml` either way — `pip
      install -e .` picks it up the same way in both venvs.

