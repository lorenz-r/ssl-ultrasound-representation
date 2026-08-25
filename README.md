# ssl-ultrasound-representation

Self-supervised latent space learning for ultrasound image representation (TUM SS26 practical).

Several self-supervised objectives are pretrained on **simulated** RF ultrasound
channel data, then probed — frozen — for what their representations actually
encode about anatomy, physics and acquisition geometry, and transferred to
**real** in-vivo tissue they have never seen.

Contact: Magdalena Wysocki (magdalena.wysocki@tum.de)

- **[docs/MEMORY.md](docs/MEMORY.md)** — the design record and every measured result. Read it for *why*.
- **[docs/work-item-map.md](docs/work-item-map.md)** — this repo used to be organised by work item (`WI-N`); that table translates old names and paths to current ones.

---

## Repository layout

```
src/ssl_us/          the library — everything importable
  beamform/          RF channel data -> beamformed K-sector store (offline, zea/JAX)
  data/              store reader, crops, splits, augmentation policies, transforms
  models/            ResNet-18 / ViT-S backbones; VICReg, MAE, I-JEPA, cross-aperture heads
  train/             training loops + collapse instrumentation
  eval/              probes, clustering, embeddings, baselines, results matrices
  utils/             W&B monitoring

scripts/             runnable entry points, one directory per pipeline stage
  data/              acquire and prepare corpora
  beamform/          build and validate the beamformed stores
  train/             one script per objective
  eval/              one script per measurement
  figures/           standalone explanatory figures
  cluster/           unattended multi-hour run chains (.sh) + GPU ops

results/             committed, canonical measured numbers (tracked in git)
outputs/             generated run artefacts: figures, per-run results.json (gitignored)
tests/               pytest suite
docs/                MEMORY.md, the work-item map, project description
```

The split between `results/` and `outputs/` is the important one. A run writes
everything into `outputs/<corpus>/<column>/`; the column that is *reported* is
copied to `results/<corpus>/<column>.json` and committed. Tests read `results/`,
so a re-run that silently moves a published number fails the suite.

---

## The pipeline

Five stages. Each depends only on the artefact the previous one wrote, so any
stage can be re-run alone.

### 1. Data — acquire the corpora

| Script | What it produces |
|---|---|
| `scripts/data/probe_nvidia_frames.py` | verifies the raw NVIDIA frame format against its documentation |
| `scripts/data/fetch_stanford_fsa.py` | the Stanford FSA transfer corpus from figshare |
| `scripts/data/fetch_nvidia_label_maps.py` | per-pixel segmentation + sound-speed maps (column-pruned: ~1.9 GB, not 271 GB) |
| `scripts/data/build_splits.py` | train/validation split ids + the split audit |

**Pretraining corpus** — NVIDIA
[NV-Raw2Insights-US](https://huggingface.co/datasets/nvidia/NV-Raw2Insights-US):
923 k-Wave-simulated frames with dense physics ground truth (sound-speed field,
cyst segmentation, phase error). Simulation is what makes the probe targets
exist at all.

**Transfer corpus** — Stanford RF channel data, 117 in-vivo rat acquisitions
from [figshare 28291985](https://doi.org/10.25452/figshare.plus.28291985)
(*Pulse-Echo Ultrasound Murine In Vivo Acquisitions*, Apache-2.0). Its **FSA**
acquisitions are used — one transmit per element, the same full-synthetic-aperture
scheme as NVIDIA — so no REFoCUS decoding step is needed.

This is a sim-pretrain / real-probe structure, not a dataset switch.

### 2. Beamform — channel data to a K-sector store

```bash
source /opt/venvs/.venv-zea/bin/activate     # zea + JAX
KERAS_BACKEND=jax python scripts/beamform/build_nvidia_store.py --out <store>.zarr
```

`AngularSectorSum` splits the receive aperture into `K` angular sectors and
writes **raw** sector sums, `(N, 2, 507, 456, 16)` float32. Raw is deliberate:
reduce along K *first*, then normalize by occupancy
(`ssl_us.data.transforms.reduce_and_normalize`) — never the reverse. K is then
a knob at training time rather than a rebuild.

`validate_{nvidia,stanford}_reconstruction.py` are the acceptance gates;
`verify_nvidia_store.py` re-runs the NVIDIA gate on frames read back *out of the
finished store*, which is a different claim from the beamformer being correct.

### 3. Train — one script per objective

```bash
source /opt/venvs/.venv/bin/activate         # PyTorch
python scripts/train/vicreg.py --policy A --k 4 --out checkpoints/<name>
```

| Objective | Script | Backbone | Notes |
|---|---|---|---|
| **VICReg** | `scripts/train/vicreg.py` | ResNet-18, or ViT-S via `--backbone vit_s` | the primary arm; two augmentation policies, A and B |
| **MAE** | `scripts/train/mae.py` | ViT-S/16 | pixel reconstruction baseline; one run per K (no A/B — it masks rather than pairing views) |
| **I-JEPA** | `scripts/train/ijepa.py` | ViT-S | predicts a *representation* of the masked block; the only augmentation-free arm |
| Cross-aperture | `src/ssl_us/models/xaperture.py` | ResNet-18 | implemented, not run |

Every training script **resumes** from `<out>/last.pt` when re-invoked with the
same command — that is what lets a six-hour unattended run survive being killed.
Pass `--no-resume` to start clean.

Holding the backbone fixed and swapping the objective (`vicreg_vit_A` → `ijepa`),
or holding the objective fixed and swapping the backbone (`vicreg_A` →
`vicreg_vit_A`), is what makes the comparison attributable to one of the two.

### 4. Evaluate — frozen encoders, identical crops

Nothing is retrained here. Every encoder sees the *same* crops (origins sampled
once for the corpus from one seed), so a difference between two columns is a
difference in weights and nothing else.

| Script | Question it answers |
|---|---|
| `scripts/eval/nvidia_matrix.py` | how much of label *y* is linearly decodable from encoder *e*, on simulation |
| `scripts/eval/stanford_matrix.py` | the same table on real tissue, encoders frozen |
| `scripts/eval/nvidia_qualitative.py`, `stanford_qualitative.py` | encoder-vs-encoder embeddings (PCA / t-SNE / UMAP, raw and geometry-regressed) + per-encoder clustering |
| `scripts/eval/rankme.py` | does the representation *collapse*? — separates a rich encoder from a 1-D one that happens to align with the probed axis |
| `scripts/eval/geometry_control.py` | are the anatomy scores just *where the crop came from*? Adds a `geometry_only` encoder as the concrete test |
| `scripts/eval/mae_mask_sweep.py` | does *any* (patch, mask) configuration let the MAE learn on speckle? |

Encoders are registered by `src/ssl_us/eval/encoders.py` — one definition read
by both corpora — and each trained arm is registered only when its checkpoint
exists, so existing columns are unchanged until a new arm is actually trained.
Baselines always present: `random_resnet18` / `random_vit` (same architecture,
same crops, never trained — the floor), `pca_raw`, `trivial`, and a frozen
natural-image control.

### 5. Report

`outputs/` collects the figures and per-run JSON; `results/` holds the committed
copies.

`scripts/figures/` builds the presentation assets into **`outputs/presentation/`** —
bare light-theme plots (no titles baked in, so a deck supplies its own) plus every
underlying number as Markdown and CSV:

```bash
python scripts/figures/presentation.py   # plots + tables from results/ — seconds, no GPU
python scripts/figures/latent_3d.py      # 3-D Stanford latents — needs store + checkpoints
python scripts/figures/occupancy.py      # sector occupancy vs depth (synthetic, no data)
python scripts/figures/augmentations.py  # the two augmentation policies side by side
```

`outputs/presentation/README.md` indexes what each plot shows and flags three
things worth getting right in a caption — chiefly that Stanford has **no
per-frame fat label** (the published per-animal table cannot be joined to an
acquisition), so the latent plots are coloured by echogenicity as a
tissue-composition proxy.

---

## Setup

Requires **Python ≥ 3.11**.

```bash
poetry install -E dev      # local dev
poetry run pytest -q
poetry run ruff check .
```

```bash
pip install -r requirements.txt    # or, without the editable install
```

On the CUDA cluster two virtualenvs are pre-baked and should not be rebuilt:
`/opt/venvs/.venv` (PyTorch — training, eval, pytest, ruff) and
`/opt/venvs/.venv-zea` (zea + JAX — beamforming only). Mixing JAX-CUDA and
torch-CUDA in one environment is a known source of conflicts.

`zea` is pinned to `==0.1.4` exactly (known API instability between releases).
It picks a Keras backend at import; with no CUDA present it reuses the installed
torch as `KERAS_BACKEND=torch`, so CPU-only development needs no extra install.

### Tests

```bash
pytest -q      # expected: 456 passed, 6 skipped
```

The 6 skips are `test_stanford_mat.py`, which needs the local `data/rats/*.mat`
subset that is not present on the cluster.

---

## W&B logging

Each user authenticates with their own key. Once, on the cluster:

```bash
echo "WANDB_API_KEY=<your-personal-key-here>" > ~/.wandb_env
```

Get the key from [wandb.ai/authorize](https://wandb.ai/authorize).

---

## Key references

- **VICReg** (Bardes et al. 2022) — variance-invariance-covariance regularization; this project's primary objective
- **I-JEPA** (Assran et al. 2023) — joint-embedding predictive architecture
- **MAE** (He et al. 2022) — masked autoencoders
- **SimCLR** (Chen et al. 2020) — contrastive learning
- **DINO** (Caron et al. 2021) — self-distillation with Vision Transformers
- **EchoJEPA** — cardiac ultrasound foundation model
