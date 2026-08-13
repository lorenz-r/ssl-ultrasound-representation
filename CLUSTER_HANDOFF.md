# Cluster hand-off — status as of this session

Written for whoever (human or agent) picks this up on the CUDA cluster next.
**Read [PLAN.md](PLAN.md) first** — it is the actual design doc, with the
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
`/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr`. **212 tests pass, `ruff
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
| `12g` | WI-8's GPU beamform pass, and any re-run of `scripts/wi4_reference_validation.py` (`KERAS_BACKEND=jax`, 507×456 — **now passing**, ~3 s per variant per frame, so it is download-bound, not compute-bound) | Beamforming is inference-only delay-and-sum over a fixed pixel grid — no gradients, no optimizer state, no large activation stacks — so 12 GB should comfortably cover it. Try this before reaching for `24g`. Confirmed: WI-4's full-resolution run used well under 12 GB on a 24 GB card. |
| `24g` | WI-12 VICReg training, WI-14's MAE baseline, WI-15's cross-aperture arm, and WI-13's results-matrix runs (feature extraction, incl. any DINOv3/DINOv2 baselines) | Training holds activations, gradients, and optimizer state at once, and VICReg's variance/covariance terms are computed per-batch, so batch size (and headroom for it) matters — give training runs the bigger card. |

If you're unsure which bucket a new script falls into: does it call
`.backward()` / an optimizer step? → GPU training tier (`24g`, or `12g` if
it's small). Is it pure forward-pass signal processing (beamforming,
inference-only feature extraction)? → `12g` first. Is it neither? → `data`.

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
| WI-12 | VICReg model, training loop, §4.7 instrumentation | `src/ssl_us/models/{resnet,vicreg}.py`, `src/ssl_us/train/{vicreg,instrument}.py`, `src/ssl_us/train/README.md` | `test_models.py` (12), `test_instrument.py` (10), `test_train_vicreg.py` (3), `test_wi12_eval_integration.py` (1) |
| WI-14 | Spatial MAE baseline (documented baseline only) | `src/ssl_us/models/mae2d.py` | `test_mae2d.py` (12) |
| WI-15 | Cross-aperture prediction arm (optional in the plan) | `src/ssl_us/models/xaperture.py` | `test_xaperture.py` (15) |
| WI-16 | Stanford `.mat` reader + HVMux remap | `src/ssl_us/beamform/stanford_mat.py` | `test_stanford_mat.py` (12, run against the real files in `data/rats/`) |
| WI-8 | Full K-sector store: per-frame beamforming, zarr store + reader, resumable build — **923/923 frames built** | `src/ssl_us/beamform/{frame,store}.py`, `scripts/wi8_{build,verify,profile}.py` | `test_store.py` (11), `test_wi8_build.py` (6) |
| WI-0 | (pre-existing) NVIDIA frame verification | `scripts/wi0_probe.py` | `test_nv_frame.py` (8) |

None of the above needed a CUDA GPU or the full corpus — every test either
uses synthetic data, a small real download (a couple of NVIDIA frames), or
the real local Stanford `.mat` files already sitting in `data/rats/`.

**Two real bugs were found and fixed along the way**, worth knowing about:
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

WI-9 (HF publication), WI-10 (splits), WI-13 (results matrix) and WI-12
(VICReg training) are all unblocked by this.

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

**Two measured surprises, both of which change the plan's numbers:**

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

**Storage, measured:** ~26 MB/frame compressed ⇒ **~24 GB** for 923 frames
(29.6 MB/frame uncompressed, per §3.7). Note `/tmp` on these instances is
**ephemeral** — its contents were empty at container start and do not survive
an instance change — so a multi-hour build must not leave its only copy there.
The NFS home persists but is quota'd; a 10 GB probe write succeeded with home
at 50 GB (33 GB of which is `new_arch_custom.sqsh`), so the headroom is real
but not unlimited. Build to fast local scratch, then copy the ~24 GB result
somewhere persistent (or straight to the Hub, WI-9).

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

Estimated 3-4h GPU / 6-10h wall (download-bound), per PLAN.md §5.4 — and the
WI-4 timings support "download-bound": beamforming a full 507×456 frame took
~3.1 s per variant on a TITAN RTX, against minutes per frame to stream the
~265 MB of raw IQ. Note `HF_CACHE_DIR` is already set to `/tmp/hf_cache` on
these instances, which is on a large local disk (816 GB free) rather than the
quota'd NFS home — keep it that way for a 271 GB pull.

### WI-9 — HuggingFace publication

Needs WI-8's output. Not started — no code written for the dataset card or
the smoke-subset extraction. PLAN.md §5.2 WI-9 lists exactly what the card
must contain (attribution, license, changes-made statement, etc.).

### WI-13 — full results matrix

**The store reader now exists** — `ssl_us.beamform.store.open_store` (WI-8), the
piece this section previously flagged as missing.

`eval/matrix.py` (WI-2) already runs the whole protocol end-to-end against
`bmode`; running it against the real K-sector store instead is now mostly a
matter of looping over `(K, policy)` cells per PLAN.md §5.2 WI-13. The probe
protocol, guards, and baselines need no changes.

`scripts/wi8_verify.py` shows the intended read path end to end (reduce the
K-ladder, *then* normalize — never the reverse). `eval/matrix.py`'s
`load_bootstrap` is the shape to match when wiring the matrix onto it. Still
blocked on WI-8's full build actually having been run.

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

- [ ] On a `data` instance: `pip install -e ".[dev]" && pytest -q` → 212 passed, 6 skipped
      (the 6 skips are `test_stanford_mat.py`, which needs `data/rats/*.mat`)
- [ ] `ruff check .` → clean
- [x] On a `12g`/`24g` instance: `nvidia-smi` works — confirmed, TITAN RTX
      24 GB, driver 580.105.08 / CUDA 13.0, `torch.cuda.is_available()` True
- [ ] Decide: one `KERAS_BACKEND=jax` venv for beamforming, torch venv for
      training (PLAN.md §0.6), or continue with the single-venv
      `KERAS_BACKEND=torch` setup this session used. Either works; the
      dependency is declared once in `pyproject.toml` either way — `pip
      install -e .` picks it up the same way in both venvs.

