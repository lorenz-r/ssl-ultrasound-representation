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

## Dataset

The processed dataset is hosted on HuggingFace at **[benbarkow/test-us-ssl](https://huggingface.co/datasets/benbarkow/us-ssl-mf)** (public).

It contains multi-focal ultrasound RF/IQ data from the Stanford Ultrasound RF Channel dataset, pre-processed and ready for self-supervised training (MAE).

### Array layout

| Axis | Dimension    | Size | Description                                                   |
| ---- | ------------ | ---- | ------------------------------------------------------------- |
| 0    | `frame`      | N    | One temporal frame per `.mat` cine (2 frames per source file) |
| 1    | `transducer` | 192  | Receive elements                                              |
| 2    | `scanline`   | 192  | Lateral scan positions                                        |
| 3    | `focal_zone` | 3    | Multi-focal transmit zones                                    |
| 4    | `iq`         | 2    | `0` = in-phase (I), `1` = quadrature (Q)                      |
| 5    | `sample`     | 215  | Fast-time depth samples                                       |

- **Dtype:** `float32`
- **Chunks:** `(1, 32, 32, 1, 2, 215)` — one frame on the leading axis, aligned for DataLoader workers
- **Compression:** Blosc-LZ4 inside a `zarr.ZipStore`

### Processing pipeline

1. Source: Verasonics Vantage `.mat` files with `MultiFocal` in the filename. Each file holds a 2-frame cine (`Resource.RcvBuffer.numFrames = 2`), giving two independent temporal snapshots per acquisition.
2. Sub-aperture combination (interleaved RX halves merged) + IQ demodulation via `pymust.rf2iq`.
3. Fast-time axis truncated at **860 samples** (signal beyond this is only zeros), then decimated by **4×** → **215 output samples** per signal.
4. Stacked into a single 6D zarr array stored as a `.zarr.zip` (Blosc-LZ4 compressed).

---

## Loading the dataset

All loading logic lives in `src/dataset.py`. The dataset is **downloaded once** to a local cache (default: `/tmp/hf_cache` (SLURM Cluster fast storage), override with `$HF_CACHE_DIR`).

### Quickstart — training splits

```python
from src.dataset import build_split_datasets

splits, meta = build_split_datasets(
    "benbarkow/test-us-ssl",
    split_strategy="file",   # "subject" for subject-level holdout
    cache_dir="/tmp/hf_cache",
)

train_ds = splits["train"]   # torch.utils.data.Dataset
val_ds   = splits["val"]
test_ds  = splits["test"]

print(len(train_ds))         # number of frames in this split
frame = train_ds[0]          # torch.Tensor shape (192, 192, 3, 2, 215)
```

Each item returned is a 5D float32 tensor `(transducers, scanlines, focal_zones, iq, samples)` — the leading frame axis has been indexed out.

### Split strategies

| Strategy    | Granularity   | Use case                                                                     |
| ----------- | ------------- | ---------------------------------------------------------------------------- |
| `"file"`    | `.mat` file   | Both frames of the same cine stay together; no within-file leakage           |
| `"subject"` | Subject (rat) | Strictest: no subject appears in two splits; better for generalization tests |

Default ratios: `train=0.7 / val=0.15 / test=0.15`. Override with `ratios=(0.8, 0.1, 0.1)`.

For subject-level control (useful when subject count is small), pass explicit lists:

```python
splits, meta = build_split_datasets(
    "benbarkow/test-us-ssl",
    split_strategy="subject",
    train_subjects=["rat1"],
    val_subjects=["rat2"],
    test_subjects=[],
)
```

### PyTorch DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
for batch in loader:
    # batch: (4, 192, 192, 3, 2, 215)
    ...
```

### Raw array access (notebooks / EDA)

```python
from src.dataset import load_dataset

arr, meta = load_dataset("benbarkow/test-us-ssl")
print(arr.shape)        # (N, 192, 192, 3, 2, 215)
print(meta["axes"])     # ['frame', 'transducer', 'scanline', 'focal_zone', 'iq', 'sample']

frame = arr[0]          # numpy ndarray, shape (192, 192, 3, 2, 215)
```

### Local dataset (no HF download)

```python
from src.dataset import build_split_datasets

splits, meta = build_split_datasets(
    local_zip="data/processed/ultrasonic_dataset.zarr.zip",
    local_metadata="data/processed/ultrasonic_dataset.metadata.json",
    split_strategy="file",
)
```

---

## Reproducing the dataset

If you have access to the raw Stanford RF `.mat` files, rebuild the dataset with:

```bash
poetry run python scripts/dataset_scripts/build_mae_dataset.py \
    --input_dir data/rats \
    --output_dir data/processed
```

Then push to HuggingFace:

```bash
poetry run python scripts/dataset_scripts/push_to_hf.py \
    --repo_id <your-hf-username>/ssl-ultrasound-rats
```

Pass `--dry-run` first to validate inputs and preview the generated README without any network writes.

---
