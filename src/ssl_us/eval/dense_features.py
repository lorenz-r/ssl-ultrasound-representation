"""Pre-pool feature maps and whole-frame inference for the segmentation probe (WI-19).

Every encoder in `store_features.py` returns ``(N, D)``: `ResNetBackbone.forward`
is torchvision's ResNet-18 with ``fc=Identity``, so global average pooling has
already collapsed the spatial grid, and the three ViT arms mean over patch
tokens. Those vectors can say *whether* a crop contains a cyst — `contains_cyst`
and `cyst_fraction_crop` are matrix cells already — but not *where* it is. This
module is the pre-pool path, ``(N, 2K, size, size) -> (N, C, g, g)``, plus the
sliding window that turns a crop-shaped probe into a whole-frame picture.

**One grid for every arm.** `DENSE_GRID` is 16 and *every* extractor resamples
to it. A ResNet hypercolumn is natively 16/8/4 across layers 2-4 and a ViT token
grid is `img_size // patch_size`, which `build_backbone` fixes at 16 for every K
— so without a common target the columns of the figure would differ in
resolution as well as in weights, and the coarser panel would look worse for a
reason that is not its representation. Resampling is bilinear, done once, on the
feature map rather than on the prediction.

**Why a hypercolumn and not `layer4`.** `layer4` is stride 32: a 4x4 map for a
128 px crop, which is 32 px per cell against cysts a few mm across. Layers 2-4
concatenated (128 + 256 + 512 = 896 channels) keep the semantic depth of the
last block and recover the spatial precision of the earlier ones. The channel
axis is then PCA-reduced to `probes.CAPACITY_DIMS`, exactly as the pooled probe
reduces its 512-d and 384-d vectors, so 896 hypercolumn channels do not buy a
score on width alone.

**The baselines are the point.** Cysts are anechoic, so a per-pixel brightness
threshold already segments them well. `trivial_dense` is that threshold, fit as
a linear head on two numbers per cell; `pca_raw_dense` is the local raw-pixel
basis. A trained arm that does not clear both has not learned to localise
anatomy — it has learned to report darkness, which the input supplies for free.
Reading a dense column without them is the one way this metric misleads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import transforms as tf
from .encoders import IJEPA_TEMPLATE, MAE_TEMPLATES, VIT_TEMPLATE, find_checkpoint
from .store_features import (
    IJEPACheckpointEncoder,
    MAECheckpointEncoder,
    StoreCropSpec,
    VICRegCheckpointEncoder,
    sample_store_crops,
)

# The common spatial grid every arm is resampled to. 16 is the ViT arms' native
# token grid at every K (`vicreg.build_backbone` derives `patch_size` as
# `crop_size // 16` precisely to hold the token count fixed), so this choice
# resamples the CNNs onto the transformers' grid rather than the reverse — the
# ViT columns are exact and only the hypercolumns interpolate.
DENSE_GRID = 16

# ResNet blocks tapped for the hypercolumn, shallowest first. `layer1` is
# excluded deliberately: at stride 4 it is 64 channels of near-raw edge
# response, which pushes the column count up by 64 while adding little the
# `pca_raw_dense` baseline does not already supply.
HYPERCOLUMN_LAYERS = ("layer2", "layer3", "layer4")

EPS_DB = 1e-12


@dataclass(frozen=True)
class DenseSpec:
    """Geometry of the dense readout, shared by every encoder in a run."""

    grid: int = DENSE_GRID
    layers: tuple[str, ...] = HYPERCOLUMN_LAYERS
    batch_size: int = 64

    def cell_size(self, crop_size: int) -> int:
        if crop_size % self.grid:
            raise ValueError(
                f"crop_size {crop_size} is not divisible by the dense grid {self.grid} — "
                "the cell-pooled baselines and the mask targets would not align"
            )
        return crop_size // self.grid


# ── envelope helpers ────────────────────────────────────────────────────────


def crop_envelope(crops: np.ndarray, k: int) -> np.ndarray:
    """``(N, 2k, H, W) -> (N, H, W)`` incoherent envelope, averaged over K.

    Same convention as `store_features.crop_echogenicity_db`, and for the same
    reason: the sectors are receive sub-apertures of one set of scatterers, so
    a coherent re-sum across them would measure inter-sector phase agreement —
    an aberration-sensitive quantity — on top of the brightness wanted here.
    The channel layout is I-block then Q-block (`sample_store_crops` writes it
    that way), so sector `s` is channels `s` and `k + s`.
    """
    if crops.shape[1] != 2 * k:
        raise ValueError(f"expected {2 * k} channels for K={k}, got {crops.shape[1]}")
    i = crops[:, :k].astype(np.float32)
    q = crops[:, k:].astype(np.float32)
    return np.sqrt(i * i + q * q).mean(1)


def log_envelope(crops: np.ndarray, k: int) -> np.ndarray:
    """`crop_envelope` in dB on the store's own raw-sum scale."""
    return 20.0 * np.log10(crop_envelope(crops, k) + EPS_DB)


def cell_pool(x: np.ndarray, grid: int) -> np.ndarray:
    """``(N, H, W) -> (N, grid, grid, H//grid, W//grid)`` — the per-cell patches.

    Kept as the un-reduced view so callers can take a mean, a std, or the whole
    flattened patch off one reshape.
    """
    n, h, w = x.shape
    ch, cw = h // grid, w // grid
    return x.reshape(n, grid, ch, grid, cw).transpose(0, 1, 3, 2, 4)


# ── pre-pool extractors ─────────────────────────────────────────────────────


class ResNetHypercolumn:
    """``(N, 2K, H, W) -> (N, sum(channels), g, g)`` from a torchvision ResNet.

    Runs the stem and blocks by hand rather than calling `forward`, because
    `forward` ends in `avgpool` + `fc` and there is no argument that stops it.
    Takes the *module* — `ResNetBackbone.net` — so the checkpoint-reading logic
    stays in `store_features.py` and this class never learns about run configs.
    """

    def __init__(self, resnet, spec: DenseSpec, device: str = "cuda"):
        self.net = resnet.to(device).eval()
        self.spec = spec
        self.device = device
        self.dim = sum(
            {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}[name]
            for name in spec.layers
        )

    @torch.no_grad()
    def __call__(self, crops: np.ndarray) -> np.ndarray:
        g = self.spec.grid
        out = []
        for i in range(0, len(crops), self.spec.batch_size):
            x = torch.from_numpy(
                np.ascontiguousarray(crops[i : i + self.spec.batch_size])
            ).float().to(self.device)
            x = self.net.maxpool(self.net.relu(self.net.bn1(self.net.conv1(x))))
            taps = []
            for name in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(self.net, name)(x)
                if name in self.spec.layers:
                    taps.append(x)
            resampled = [
                t if t.shape[-2:] == (g, g)
                else F.interpolate(t, size=(g, g), mode="bilinear", align_corners=False)
                for t in taps
            ]
            out.append(torch.cat(resampled, dim=1).float().cpu().numpy().astype(np.float16))
        return np.concatenate(out)


class TokenMap:
    """``(N, 2K, H, W) -> (N, D, g, g)`` from any callable returning patch tokens.

    `token_fn` is `MAE2D.encode_tokens`, `IJEPA.encode_tokens`, or a ViT
    backbone's encoder — all three emit ``(B, N, D)`` in row-major patch order
    (`mae2d.PatchEmbed` unfolds rows then columns), so the reshape below is the
    inverse of the tokenization rather than a guess about it.
    """

    def __init__(self, token_fn, spec: DenseSpec, device: str = "cuda"):
        self.token_fn = token_fn
        self.spec = spec
        self.device = device
        self.dim: int | None = None

    @torch.no_grad()
    def __call__(self, crops: np.ndarray) -> np.ndarray:
        g = self.spec.grid
        out = []
        for i in range(0, len(crops), self.spec.batch_size):
            x = torch.from_numpy(
                np.ascontiguousarray(crops[i : i + self.spec.batch_size])
            ).float().to(self.device)
            tokens = self.token_fn(x)  # (B, N, D)
            b, n, d = tokens.shape
            g0 = int(round(math.sqrt(n)))
            if g0 * g0 != n:
                raise ValueError(f"{n} tokens is not a square grid — cannot map to (g, g)")
            m = tokens.transpose(1, 2).reshape(b, d, g0, g0)
            if g0 != g:
                m = F.interpolate(m, size=(g, g), mode="bilinear", align_corners=False)
            self.dim = d
            out.append(m.float().cpu().numpy().astype(np.float16))
        return np.concatenate(out)


class TrivialDense:
    """Mean and std of the cell's log-envelope. Two numbers per cell.

    The dense counterpart of `features.trivial_features`, and the baseline that
    decides whether a dense column means anything: a linear head on these two
    channels *is* a per-cell brightness threshold, which on anechoic cysts is
    already a competent segmenter. Every trained arm is read against it.
    """

    def __init__(self, k: int, spec: DenseSpec):
        self.k = k
        self.spec = spec
        self.dim = 2

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        cells = cell_pool(log_envelope(crops, self.k), self.spec.grid)
        flat = cells.reshape(*cells.shape[:3], -1)
        return np.stack([flat.mean(-1), flat.std(-1)], axis=1).astype(np.float16)


class PcaRawDense:
    """PCA over the cell's flattened log-envelope patch — the local raw-pixel basis.

    ⚠ Fitted rather than applied, like `features.pca_features`: it fits on
    whatever array it is handed, so under chunked extraction it fits one basis
    per chunk. `scripts/eval/nvidia_segmentation.py` warns when that happens and
    records it, the same way `nvidia_matrix.py` does for `pca_raw`.
    """

    def __init__(self, k: int, spec: DenseSpec, dims: int = 64, seed: int = 0):
        self.k = k
        self.spec = spec
        self.dims = dims
        self.seed = seed
        self.dim = dims

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        g = self.spec.grid
        cells = cell_pool(log_envelope(crops, self.k), g)
        n = len(crops)
        flat = cells.reshape(n * g * g, -1).astype(np.float32)
        dims = min(self.dims, flat.shape[0], flat.shape[1])
        z = PCA(n_components=dims, random_state=self.seed).fit_transform(flat)
        return z.reshape(n, g, g, dims).transpose(0, 3, 1, 2).astype(np.float16)


# ── encoder construction ────────────────────────────────────────────────────


def _resnet_module(pooled) -> torch.nn.Module | None:
    """The torchvision ResNet inside a pooled encoder, or None if it is a ViT."""
    inner = getattr(pooled, "model", None) or getattr(pooled, "net", None)
    inner = getattr(inner, "net", inner)  # ResNetBackbone.net / ViTBackbone.net
    return inner if hasattr(inner, "layer4") else None


def dense_from_pooled(pooled, spec: DenseSpec, device: str):
    """The pre-pool extractor for an already-constructed pooled encoder.

    Deliberately built *from* the pooled encoder rather than beside it: the
    checkpoint is read once, by the class that knows its `run_config.json`, and
    the dense and pooled readings are then provably the same weights.
    """
    if isinstance(pooled, MAECheckpointEncoder):
        return TokenMap(pooled.model.encode_tokens, spec, device)
    if isinstance(pooled, IJEPACheckpointEncoder):
        return TokenMap(
            lambda x: pooled.model.encode_tokens(x, encoder=pooled.encoder), spec, device
        )
    resnet = _resnet_module(pooled)
    if resnet is not None:
        return ResNetHypercolumn(resnet, spec, device)
    vit = getattr(pooled, "model", None) or getattr(pooled, "net", None)
    vit = getattr(vit, "net", vit)
    if hasattr(vit, "patch_embed"):
        return TokenMap(vit, spec, device)
    raise TypeError(f"no pre-pool path known for {type(pooled).__name__}")


def build_dense_encoders(args, k: int, spec: DenseSpec, tag: str = "wi19") -> dict:
    """Encoder name -> pre-pool extractor, mirroring `encoders.build_encoders`.

    Same checkpoint-discovery constants, same arm names, same "registered only
    when trained" rule — so a dense column and a pooled column disagree about
    which arms exist only when the checkpoints on disk do. `args` supplies
    `checkpoint_root`, `device` and `seed`.
    """
    from ..models.resnet import ResNetBackbone

    in_channels = 2 * k
    device = args.device
    out: dict = {}

    def register(name: str, pooled) -> None:
        out[name] = dense_from_pooled(pooled, spec, device)

    for policy in ("A", "B"):
        ckpt = find_checkpoint(args.checkpoint_root, k, policy)
        if ckpt is None:
            print(f"[{tag}] policy {policy}: no K={k} checkpoint found — skipping")
            continue
        print(f"[{tag}] policy {policy}: {ckpt}")
        register(
            f"vicreg_{policy}",
            VICRegCheckpointEncoder(ckpt, in_channels=in_channels, device=device),
        )

    vit_ckpt = Path(args.checkpoint_root) / VIT_TEMPLATE.format(k=k) / "last.pt"
    if vit_ckpt.exists():
        print(f"[{tag}] vicreg_vit_A: {vit_ckpt}")
        register(
            "vicreg_vit_A",
            VICRegCheckpointEncoder(vit_ckpt, in_channels=in_channels, device=device),
        )
    else:
        print(f"[{tag}] no vicreg_vit checkpoint for K={k} — skipping")

    for template, name in MAE_TEMPLATES:
        mae_ckpt = Path(args.checkpoint_root) / template.format(k=k) / "last.pt"
        if mae_ckpt.exists():
            print(f"[{tag}] {name}: {mae_ckpt}")
            register(name, MAECheckpointEncoder(mae_ckpt, in_channels=in_channels, device=device))
        else:
            print(f"[{tag}] no {name} checkpoint for K={k} — skipping")

    ijepa_ckpt = Path(args.checkpoint_root) / IJEPA_TEMPLATE.format(k=k) / "last.pt"
    if ijepa_ckpt.exists():
        print(f"[{tag}] ijepa: {ijepa_ckpt}")
        register(
            "ijepa", IJEPACheckpointEncoder(ijepa_ckpt, in_channels=in_channels, device=device)
        )
        register(
            "random_vit",
            IJEPACheckpointEncoder(
                ijepa_ckpt, in_channels=in_channels, device=device,
                random_init=True, seed=args.seed,
            ),
        )
    else:
        print(f"[{tag}] no ijepa checkpoint for K={k} — skipping")

    torch.manual_seed(args.seed)
    out["random_resnet18"] = ResNetHypercolumn(
        ResNetBackbone(in_channels, seed=args.seed).net, spec, device
    )
    out["pca_raw_dense"] = PcaRawDense(k, spec, seed=args.seed)
    out["trivial_dense"] = TrivialDense(k, spec)
    return out


# ── corpus extraction ───────────────────────────────────────────────────────


@dataclass
class DenseBatch:
    """Feature maps, their per-pixel mask targets, and the frame each came from."""

    maps: dict[str, np.ndarray]  # name -> (N, C, g, g) float16
    seg: np.ndarray  # (N, size, size) uint8 — the per-pixel target
    frame_id: np.ndarray  # (N,) str
    origins: np.ndarray  # (N, 2) int
    segmentation_empty: np.ndarray  # (N,) bool

    def __len__(self) -> int:
        return len(self.frame_id)


def extract_dense_features(
    store_path: str | Path,
    frame_ids: list[str],
    spec: StoreCropSpec,
    encoders: dict,
    label_maps,
    *,
    dense_spec: DenseSpec,
    frame_chunk: int = 0,
    tag: str = "wi19",
) -> DenseBatch:
    """Dense features for every encoder, a chunk of frames at a time.

    Mirrors `store_features.extract_features`, including its reason for living
    in the package rather than the script, and adds the per-pixel target: the
    crops and the masks come off the *same* `origins`, so a mask can never be
    paired with a crop it was not cut from.

    Feature maps are stored `float16`. At the default grid a K=4 hypercolumn is
    896 x 16 x 16 per crop, i.e. 459 kB in fp32 and 229 kB in fp16; over a
    93-frame validation split at 8 crops per frame that is the difference
    between 342 MB and 171 MB *per encoder*, and there are nine of them. The
    probe casts back to fp32 before PCA, where the precision actually matters.
    """
    chunk = frame_chunk or len(frame_ids)
    parts: list[tuple] = []
    acc: dict[str, list[np.ndarray]] = {name: [] for name in encoders}
    for start in range(0, len(frame_ids), chunk):
        sub = frame_ids[start : start + chunk]
        b = sample_store_crops(
            store_path, frame_ids, spec, label_maps=label_maps, only_frames=list(sub)
        )
        for name, fn in encoders.items():
            acc[name].append(fn(b.images))
        seg = label_maps.crop_masks(list(b.frame_id), b.origins, spec.crop_size)
        b.images = np.empty((0,), dtype=np.float32)
        parts.append((b.frame_id, b.origins, seg, b.labels["segmentation_empty"]))
        if chunk < len(frame_ids):
            print(
                f"[{tag}] dense features {min(start + chunk, len(frame_ids))}/"
                f"{len(frame_ids)} frames",
                flush=True,
            )
    return DenseBatch(
        maps={name: np.concatenate(v) for name, v in acc.items()},
        seg=np.concatenate([p[2] for p in parts]),
        frame_id=np.concatenate([p[0] for p in parts]).astype(str),
        origins=np.concatenate([p[1] for p in parts]),
        segmentation_empty=np.concatenate([p[3] for p in parts]).astype(bool),
    )


# ── whole-frame sliding window ──────────────────────────────────────────────


def reduced_frame(store, frame_id: str, k: int, intensity_scale: float = 1.0) -> np.ndarray:
    """One whole frame in the encoder's layout, ``(2k, nz, nx)``.

    Reducing the K-ladder for the *whole* frame and cutting crops afterwards is
    exactly equivalent to `sample_store_crops`' cut-then-reduce, because
    `reduce_and_normalize` is elementwise in (z, x): it sums along the sector
    axis and divides by the co-reduced occupancy at the same pixel. Nothing
    crosses a crop boundary, so the sliding window costs one frame read rather
    than one per window — which is what makes a 132-window frame affordable.
    """
    frame = store.read_frame(store.index_of(frame_id))  # (2, Z, X, K_store)
    occ = np.asarray(store.occupancy)
    v = tf.reduce_and_normalize(frame, occ, target_k=k)  # (2, Z, X, k)
    if intensity_scale != 1.0:
        v = v * intensity_scale
    return np.moveaxis(v, -1, 1).reshape(2 * k, *v.shape[1:3])


def window_origins(nz: int, nx: int, size: int, stride: int) -> np.ndarray:
    """Top-left origins tiling ``(nz, nx)``, with the last row/column flush to the edge.

    Snapping the final window to the border rather than dropping it is what
    keeps the accumulator from having an uncovered strip on the bottom and
    right whenever `size` does not divide the frame — 507 and 456 are not
    multiples of 128, so this is the normal case, not an edge case.

    ⚠ ``stride > size`` is refused rather than tolerated. It leaves gaps between
    windows, and an uncovered pixel comes out of the accumulator as probability
    0 — indistinguishable from a confident background prediction. The figure
    renders as a checkerboard and every Dice in the table is depressed by an
    amount that has nothing to do with the representation.
    """
    if stride > size:
        raise ValueError(
            f"window stride {stride} exceeds the crop size {size}: the windows would "
            "not cover the frame, and the uncovered pixels would score as confident "
            "background. Use a stride <= the crop size."
        )
    rows = list(range(0, max(nz - size, 0) + 1, stride))
    cols = list(range(0, max(nx - size, 0) + 1, stride))
    if rows[-1] != nz - size:
        rows.append(nz - size)
    if cols[-1] != nx - size:
        cols.append(nx - size)
    return np.array([(r, c) for r in rows for c in cols], dtype=int)


def whole_frame_probability(
    frame: np.ndarray,
    extractor,
    head,
    *,
    size: int,
    stride: int,
    batch: int = 32,
) -> np.ndarray:
    """``(2k, nz, nx) -> (nz, nx)`` cyst probability, by overlap-averaged windows.

    Each window is encoded, scored by `head` at the dense grid, bilinearly
    upsampled back to `size`, and accumulated with a count map. Averaging
    probabilities over overlapping windows — rather than stitching disjoint
    tiles — is what removes the window seams that would otherwise read as
    structure in the figure.

    `head` is a fitted `dense_probe.DenseHead`; it must be the *same* object
    used for the metric, or the picture illustrates a different model than the
    number beside it.
    """
    _, nz, nx = frame.shape
    origins = window_origins(nz, nx, size, stride)
    total = np.zeros((nz, nx), dtype=np.float64)
    count = np.zeros((nz, nx), dtype=np.float64)
    for start in range(0, len(origins), batch):
        block = origins[start : start + batch]
        crops = np.stack([frame[:, r : r + size, c : c + size] for r, c in block])
        prob = head.predict_proba_maps(extractor(crops))  # (B, g, g)
        up = (
            F.interpolate(
                torch.from_numpy(prob).float()[:, None],
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            .numpy()
            .astype(np.float64)
        )
        for (r, c), p in zip(block, up, strict=True):
            total[r : r + size, c : c + size] += p
            count[r : r + size, c : c + size] += 1.0
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def frame_bmode_db(frame: np.ndarray, k: int) -> np.ndarray:
    """Log-envelope of a whole reduced frame, for the figure's B-mode panel."""
    return log_envelope(frame[None], k)[0]
