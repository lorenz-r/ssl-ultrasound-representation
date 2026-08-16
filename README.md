# ssl-ultrasound-representation

Self-supervised latent space learning for ultrasound image representation (TUM SS26 practical).

Contact: Magdalena Wysocki (magdalena.wysocki@tum.de)

---

## Setup

Requires **Python >=3.11**.

**Poetry** (recommended):

```bash
poetry install
poetry shell
```

**pip**:

```bash
pip install -r requirements.txt
```

---

## Dataset and pipeline

As of `feature/beamformed-vicreg` (see **[PLAN.md](PLAN.md)** for the full design and
rationale), the project pretrains on the simulated **NVIDIA
[NV-Raw2Insights-US](https://huggingface.co/datasets/nvidia/NV-Raw2Insights-US)**
corpus (923 k-Wave-simulated frames, dense physics ground truth: sound-speed field,
cyst segmentation, phase error) and evaluates transfer on the real **Stanford
RF channel data** — 117 in-vivo rat acquisitions from
[figshare 28291985](https://doi.org/10.25452/figshare.plus.28291985)
(*Pulse-Echo Ultrasound Murine In Vivo Acquisitions*, Apache-2.0), built into a
corpus by `scripts/wi16_fetch_stanford_fsa.py`. This is a sim-pretrain /
real-probe structure, not a dataset switch — see PLAN.md §3.6.

The transfer corpus uses that source's **FSA** acquisitions — one transmit per
element, the same full-synthetic-aperture scheme as NVIDIA — rather than its
MultiFocal ones, so no REFoCUS decoding step is needed. The `data/rats/*.mat`
files referenced by `tests/test_stanford_mat.py` are a local phase-1 subset of
the MultiFocal acquisitions; those tests skip when it is absent.

The old Stanford-only MAE pipeline (`src/ssl_ultrasound_representation/`, the
`benbarkow/us-ssl-mf` HF dataset) has been retired; it lives in git history on
`main` (PLAN.md WI-1, §1).

### Module layout

```
src/ssl_us/
  beamform/   offline NVIDIA/Stanford -> beamformed store (separate concerns from training)
  data/       store reader, crops, splits, augmentation policies, transforms
  models/     ResNet-18, VICReg, MAE, cross-aperture arms
  train/      training loop scaffolding + instrumentation
  eval/       probes, clustering, embeddings, baselines, results matrix
  utils/      W&B monitoring
```

`scripts/wi0_probe.py` and `tests/test_nv_frame.py` are WI-0 (verifying the raw
NVIDIA frame against its documentation before anything downstream is built on
it); `src/ssl_us/eval/matrix.py` is WI-2 (the eval harness, runnable end-to-end
on the dataset's own `bmode` field with a random encoder, before any beamformer
exists — PLAN.md §4.6).

### zea

Offline beamforming (`src/ssl_us/beamform/`) uses
[`zea`](https://github.com/tue-bmd/zea) `==0.1.4`, pinned exactly (PLAN.md §0.7).
It is declared as a normal dependency of this project rather than a separate
virtualenv: it auto-detects an installed Keras backend, and since this project
already depends on `torch`, `KERAS_BACKEND=torch` reuses it with no separate
JAX/TensorFlow install needed for CPU-only development. On the training
cluster, prefer a dedicated `KERAS_BACKEND=jax` environment for the offline
beamforming step per PLAN.md §0.6, to keep JAX-CUDA away from torch-CUDA.

---

## 📊 Enabling Weights & Biases (W&B) Logging

To track your experiments on the cluster, you need to authenticate your personal Weights & Biases account. For security and compatibility with our shared containers, each user must set up a local environment file.

### Setup Instructions (Do this once)

**1. Get your API key:**
Log in to your W&B account and grab your personal API key from [wandb.ai/authorize](https://wandb.ai/authorize).

**2. Create your credentials file:**
Log into the cluster and run the following command to create a hidden `.wandb_env` file in your home directory. Be sure to replace `<your-personal-key-here>` with your actual key:

```bash
echo "WANDB_API_KEY=<your-personal-key-here>" > ~/.wandb_env
```
