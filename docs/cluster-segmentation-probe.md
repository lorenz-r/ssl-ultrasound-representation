# WI-19 — segmentation probe and mask figure, on the cluster

How to run the per-pixel cyst probe and produce the mask figure. This is the
operating doc; the reasoning lives in the module docstrings
(`src/ssl_us/eval/dense_probe.py` and `dense_features.py`) and the WI numbering
in [work-item-map.md](work-item-map.md).

## What this produces and why it is not already in the matrix

`nvidia_matrix.py` already probes the segmentation map, but only after reducing
it to a scalar: `contains_cyst` asks whether a crop contains a cyst,
`cyst_fraction_crop` asks how much. Neither asks **where**. WI-19 asks that, and
gives two artefacts:

| File | What it is |
|------|------------|
| `results.json` | full record — per-encoder Dice with a frame-level CI, per-frame Dice, the run's geometry, and the exclusions applied |
| `dice_table.md` | the same table, readable |
| `masks.png` | B-mode, ground truth, then one predicted mask per encoder, on whole frames |

> **Read every trained arm against `trivial_dense`.** Cysts are anechoic, so a
> per-cell brightness threshold already segments them competently — in the
> synthetic end-to-end test it scores Dice 0.99. Both the number and the picture
> will look respectable for *every* encoder in the table. The finding is the
> **gap** between a trained arm and that threshold. A run whose trained arms do
> not clear it has reported that darkness explains the masks, which is a result
> and should be presented as one, not quietly dropped.

## Which instance

**`24g`.** It is a GPU job — nine encoders' forward passes over the crops, then
again over ~156 sliding windows per figure frame — and the peak host memory is
around 2.5 GiB at the default settings, above the `data` instance's 4 GiB cgroup
once the crops are resident too. `12g` would fit the model but there is no
reason to prefer it; nothing here needs gradients, and the bottleneck is the
probe's CPU-side logistic fits, not VRAM.

```bash
cd /mnt/home/benbarkow/ssl-ultrasound-representation
source /opt/venvs/.venv/bin/activate   # the PyTorch venv, not .venv-zea
```

## Prerequisites

1. **The per-pixel label maps.** Unlike `nvidia_matrix.py`, which degrades to
   `not_run` without them, this script hard-fails: the map *is* the target.

   ```bash
   ls /mnt/home/benbarkow/stores/wi8_label_maps/manifest.json
   # missing? -> python scripts/data/fetch_nvidia_label_maps.py   (data instance, ~34 min)
   ```

2. **Trained checkpoints** under `/mnt/home/benbarkow/checkpoints`. Arms are
   registered only when their checkpoint exists, exactly as in the matrix
   columns, and skipped arms are printed. The three baselines
   (`trivial_dense`, `pca_raw_dense`, `random_resnet18`) need no checkpoint, so
   the script produces a usable column even with none.

## Running it

```bash
# every K that has a trained pair, validation split
bash scripts/cluster/eval_segmentation.sh

# one column, by hand
python -u scripts/eval/nvidia_segmentation.py \
  --k 4 --split validation \
  --store /mnt/home/benbarkow/stores/wi8_fsa_k16.zarr \
  --label-maps /mnt/home/benbarkow/stores/wi8_label_maps \
  --checkpoint-root /mnt/home/benbarkow/checkpoints \
  --out outputs/nvidia/k4_validation_segmentation
```

The wrapper honours `KS`, `SPLIT` and `CROPS_PER_FRAME` from the environment:

```bash
KS=4 CROPS_PER_FRAME=16 bash scripts/cluster/eval_segmentation.sh
```

### Smoke test first

Two frames, no figure, on CPU — finishes in under a minute and catches a bad
path or a stale label-map directory before an hour of extraction:

```bash
python -u scripts/eval/nvidia_segmentation.py \
  --k 4 --limit-frames 4 --crops-per-frame 4 \
  --no-figure --n-boot 50 --device cpu \
  --out /tmp/wi19_smoke
```

## Cost

K=4, validation (93 frames), `--crops-per-frame 8`. Sizes are derived from
the array shapes; wall clock is an estimate from the fit counts and has not
been measured on the cluster yet — record the real number here after the
first run.

| | |
|---|---|
| crops | 744 |
| dense readout | 16x16 cells, 8 px per cell at a 128 px crop |
| feature maps, per crop, all nine arms | ~2.3 MiB (fp16) |
| feature maps, total | ~1.7 GiB |
| crop materialization | ~372 MiB |
| peak host memory | ~2.5 GiB |
| sliding windows per figure frame | 156 at stride 32 |
| wall clock | ~30-60 min, dominated by the probe's nested CV |

**Memory scales with (crops x channels).** `--frame-chunk` does *not* bound it —
chunking caps only the crop materialization, the same limitation
`nvidia_matrix.py` has. Use `--crops-per-frame` to control the feature
footprint. Scaling from the table above:

- `--split validation --crops-per-frame 16` -> ~3.3 GiB
- `--split train --crops-per-frame 8` -> ~15 GiB, i.e. do not, without dropping
  crops per frame to 2-3 first

**Runtime is the logistic fits, not the GPU.** Each encoder runs 5 outer folds x
5 penalties x 3 inner folds on up to `--max-fit-cells` rows. `--max-fit-cells
50000` cuts the probe time roughly 4x at some cost in fit stability; scoring
always uses every held-out cell regardless.

## What the protocol does, and the three places it deviates from PLAN.md 4.4

Unchanged: `GroupKFold` over `frame_id`, PCA capacity control to 64 dimensions,
and the exclusion of crops from the 75 empty-`segmentation_map` frames
(PLAN.md 2.6 — an empty mask is an unknown mix of true negatives and unlabelled
edge-touching cysts, and as a *dense* target it would teach the head to erase
real cysts).

Deviating, deliberately:

1. **The bootstrap resamples frames, not rows.** A row here is a cell, not a
   crop; resampling rows over ~190k cells would report a CI roughly two orders
   of magnitude too tight.
2. **The penalty grid is 5 values, not 15, and fits are subsampled.** Nested CV
   at the full grid does not finish on this many rows.
3. **`class_weight="balanced"` is on.** Cyst pixels are a small minority and an
   unweighted head at a 0.5 threshold collapses to all-background for every arm.
   Applied identically to every encoder including the baselines.

Because of (2) and (3), **a dense cell is not interchangeable with a pooled
matrix cell.** They are reported in separate tables and should stay that way.

`dice` is scored at full pixel resolution after upsampling the 16x16 readout;
`dice_cell` in the same row is the score on the grid the head actually predicts
on. The gap between them is how much is lost to the coarse readout rather than
to the representation.

## The figure

Frames are chosen **by rule, not by eye**: the median-Dice and worst-Dice frame
of `--reference-encoder` (default `vicreg_A`), among frames with a non-empty
mask, worst last. One hand-picked frame is an anecdote; the worst frame is what
keeps a presentation honest. The head that renders them is fit on the other
frames of the split, so the picture is out-of-sample.

Columns are ordered baselines-first (`FIGURE_ORDER` in the script) so a reader
meets `trivial_dense` before `vicreg_A`. The ground-truth contour is drawn over
every prediction panel, so over- and under-segmentation are separable without
comparing panels across the page. The Dice printed under each panel is
whole-frame and is the same quantity as the table's, computed on that frame.

Useful flags:

- `--figure-frames 3` — more rows, drawn evenly across the Dice ranking.
- `--window-stride 16` — finer stitching, ~4x the windows. Must be `<=` the crop
  size; a larger stride leaves uncovered pixels that score as confident
  background, and the script refuses it rather than rendering a checkerboard.
- `--no-figure` — table only.
- `--reference-encoder trivial_dense` — pick figure frames by where the
  *baseline* struggles instead, which is the more interesting slide when the
  trained arms are close to it.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `FileNotFoundError: no manifest at .../wi8_label_maps` | label maps not fetched — see Prerequisites |
| `... holds N of 923 frames` | partial fetch; unwritten rows read back as zeros, i.e. as empty masks. Finish it |
| `fewer than 2 frames with a non-empty mask` | `--limit-frames` too small, or the split is all empty-mask frames |
| every arm at Dice ~0.0 | the head collapsed to background; check `positive_rate` in `results.json` |
| every arm within noise of `trivial_dense` | not a bug. That is the finding — report it |
| `window stride N exceeds the crop size` | lower `--window-stride`; see above |
| OOM during extraction | lower `--crops-per-frame`, not `--frame-chunk` |
