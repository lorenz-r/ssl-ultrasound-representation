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

### NV-Raw2Insights-US

The NVIDIA **NV-Raw2Insights-US** dataset contains 923 simulated full synthetic
aperture (FSA) acquisitions: 830 publisher training frames and 93 validation
frames. The raw Parquet samples have 180 transmit events, 180 receive elements,
and 1024 complex IQ samples.

Set the repository and dataset paths before running the commands below:

```bash
export REPO_ROOT=/path/to/ssl-ultrasound-representation
export NV_RAW_ROOT=/mnt/projects/drl4cvb/us_ssl/nv-us/NV-Raw2Insights-US
export NV_WORK=/tmp/${USER}/nv-raw2insights

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$NV_WORK"
```

#### Load raw Parquet samples

Use the dataset adapter for inspection or small experiments:

```python
from ssl_ultrasound_representation.dataset import build_nv_raw2insights_datasets

splits = build_nv_raw2insights_datasets(
    "/mnt/projects/drl4cvb/us_ssl/nv-us/NV-Raw2Insights-US",
    mode="mae",
    sample_stride=4,
)
frame = splits["train"][0]
print(frame.shape)  # (180 RX, 180 TX, 2 IQ, 256 depth samples)
```

Direct random access to the Parquet shards is too slow for training. Convert the
dataset once to an unzipped Zarr directory on the compute node's local SSD:

```bash
python scripts/dataset_scripts/build_nv_raw2insights_mae_dataset.py \
  --input-root "$NV_RAW_ROOT" \
  --output-store "$NV_WORK/nv_raw2insights_full.zarr" \
  --metadata "$NV_WORK/nv_raw2insights_full.metadata.json" \
  --beamform-params "$NV_WORK/nv_raw2insights_full.beamform_params.json" \
  --sample-stride 4 \
  --workers 4
```

This produces the array `rf_iq` with layout
`(frame, RX, TX, focal_zone, IQ, depth) = (923, 180, 180, 1, 2, 256)`.
For a conversion smoke test, add `--max-samples-per-split 2` and use a separate
output path.

Generate a manifest for the 830 publisher training frames and deterministic
scanline masks:

```bash
python scripts/generate_nv_complex_unet_sweep_inputs.py \
  --metadata "$NV_WORK/nv_raw2insights_full.metadata.json" \
  --output-dir "$NV_WORK/inputs" \
  --sample-sizes 830 \
  --mask-counts 12 36 60 90 120 150 179
```

#### Train the complex U-Net

The trainer counts optimizer steps. With all 830 training frames, 830 steps are
one frame epoch. The following command trains a 100-epoch zero-mask
reconstruction baseline:

```bash
python scripts/train_nv_complex_unet.py \
  --data-zarr "$NV_WORK/nv_raw2insights_full.zarr" \
  --metadata "$NV_WORK/nv_raw2insights_full.metadata.json" \
  --beamform-params "$NV_WORK/nv_raw2insights_full.beamform_params.json" \
  --train-manifest "$NV_WORK/inputs/n830_train_global_indices.txt" \
  --output-dir "$NV_WORK/runs/reconstruction" \
  --run-name nv-complex-unet-reconstruction \
  --mask-count 0 \
  --loss-region all \
  --steps 83000 \
  --warmup-steps 830 \
  --stream-depth-windows \
  --validate-at-start \
  --validate-every 830 \
  --checkpoint-policy best
```

W&B logging is online by default. Run `wandb login` first, or pass
`--wandb-mode disabled`. Keeping `--checkpoint-policy best` avoids retaining a
checkpoint at every stage.

For MAE-style training with a new random mask each step, replace the zero-mask
arguments with one of the following configurations:

```bash
# Mask 60 complete TX scanlines and compute loss only on those scanlines.
--masking-mode scanline --mask-count 60 --loss-region masked

# Mask 36 TX positions independently for every RX element.
--masking-mode receive_tube --mask-count 36 --loss-region masked
```

Append these arguments to the training command; remove `--mask-count 0` and
`--loss-region all`. To initialize MAE training from the reconstruction model,
also add `--init-checkpoint "$NV_WORK/runs/reconstruction/best.pt"`. Use
`--fixed-mask-file "$NV_WORK/inputs/k060_masked_tx.txt"` only for a fixed
scanline-mask experiment; omitting it is the standard dynamic-mask setting.

#### Complex U-Net design

The current implementation combines three architectural ideas:

- [U-Net](https://arxiv.org/abs/1505.04597) provides the symmetric
  encoder-decoder topology and high-resolution skip connections.
- [Deep Complex Networks](https://openreview.net/forum?id=H1T2hmZAb) provides
  the coupled real/imaginary convolution rule and complex covariance
  normalization used to preserve IQ phase information.
- [Deep Complex U-Net](https://arxiv.org/abs/1903.03107) demonstrates how these
  complex-valued operations can be combined with a U-Net encoder-decoder while
  retaining phase-sensitive information through the reconstruction path.

For NV-Raw2Insights-US, one axial depth slice is treated as a `180 x 180` TX/RX
grid. The model receives two complex channels: normalized IQ data and a binary
visibility mask whose imaginary component is zero. Every encoder and decoder
operation uses coupled complex convolutions; downsampling uses strided complex
convolution, upsampling uses complex transposed convolution, and same-scale
encoder features are concatenated through U-Net skip connections. The network
internally pads the `180 x 180` grid to a valid power-of-two multiple and crops
the output back to the original geometry. Its output is one complex channel
containing the reconstructed IQ response.

#### Beamform a trained checkpoint

Generate a full-aperture comparison for publisher validation frame 830:

```bash
python scripts/evaluate_nv_complex_unet_checkpoint.py \
  --checkpoint "$NV_WORK/runs/scanline-k60/best.pt" \
  --data-zarr "$NV_WORK/nv_raw2insights_full.zarr" \
  --beamform-params "$NV_WORK/nv_raw2insights_full.beamform_params.json" \
  --masking-mode scanline \
  --mask-file "$NV_WORK/inputs/k060_masked_tx.txt" \
  --global-frame 830 \
  --beamform-tx-step 1 \
  --output-dir "$NV_WORK/eval/scanline-k60"
```

The evaluation writes `beamforming_frame0.png` and `summary.json`. For a
receive-tube model, replace `--mask-file ...` with
`--masking-mode receive_tube --mask-count 36`.

### Rat multi-focal dataset

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
