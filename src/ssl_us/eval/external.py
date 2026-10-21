"""External frozen baselines — PLAN.md §4.4 baseline 5, the natural-image control.

These answer "does any of this beat an off-the-shelf encoder?", which §4.4 lists
and nothing in the project had implemented. They are **ceilings, not the
answer**: §4.4's reporting rule keeps them in a visually separate block from the
internal arms, and every claim about signal stage cites the internal
B-mode-vs-sub-aperture row instead.

**They see log-compressed B-mode, and only that.** A ViT pre-trained on natural
images cannot consume a 2-channel complex IQ sector stack, so §4.4 fixes the
input for baselines 5-8: *"Log-compressed B-mode, single channel replicated
to 3 … All run on the K = 1 log-envelope stage."* :func:`bmode_from_crops`
implements exactly that.

**Which model — and the gate, which has a legitimate way around it.** §4.4 names
frozen **DINOv3**, notes `facebook/dinov3-*` is `gated: manual`, and falls back
to `facebook/dinov2-small`. Measured 2026-08-19: every `facebook/dinov3-*` repo
still returns *"Access … is restricted and you are not in the authorized
list"* — authenticated as the project's own HF account, so it is a genuine
account-level gate and not a missing login.

That gate is Meta's **distribution** gate, not a use restriction. The DINOv3
License itself grants "a non-exclusive, worldwide, non-transferable and
royalty-free limited license … to use, reproduce, distribute, copy, create
derivative works of, and make modifications to the DINO Materials", and clause
1.b.i permits redistribution provided the agreement travels with it. The `timm`
organisation on HuggingFace redistributes the same weights **ungated, with
`LICENSE.md` included**, which is exactly what that clause requires. So
:data:`DINOV3_TIMM` is tried first and normally wins, and the project gets the
model §4.4 actually asked for rather than the fallback.

⚠ **One obligation follows the weights into the write-up.** Clause 1.b.ii: *"If
you submit for publication the results of research you perform on, using, or
otherwise in connection with DINO Materials, you must acknowledge the use of
DINO Materials in your publication."* Whichever model id this reports must be
named in the report.

⚠ **Three traps this module exists to avoid**, each of which silently produces a
plausible number:

1. **Per-image dB normalisation would delete the `echogenicity` axis.** Display
   B-mode is conventionally normalised to its own max, and doing that per crop
   makes every crop's brightness identical by construction — turning a probe
   target into a constant. The reference level here is **one percentile over
   the whole corpus** (:func:`bmode_from_crops`'s `reference`), so relative
   brightness between crops survives. This is the same failure the cross-corpus
   amplitude match avoids (`store_features.corpus_intensity_scale`), in a
   different coordinate.
2. **Reducing a K-sector crop to K=1 by averaging its sectors is not the K=1
   stage.** Store crops arrive occupancy-*normalised* at their target K, so
   averaging sector means is not the same as summing raw sums and dividing by
   the total occupancy (PLAN.md §3.2's whole point). The caller must re-sample
   at `k=1` instead; origins do not depend on `k`, so the crops are the same
   regions. :func:`assert_same_crops` enforces the alignment.
3. **DINOv3 ViTs carry register tokens.** `vit_small_patch16_dinov3` emits 261
   tokens for a 256 px input: 1 CLS + **4 registers** + 256 patches. Mean-pooling
   from index 1 would average four non-spatial registers into the patch
   descriptor. The prefix count is read from the model
   (`num_prefix_tokens`), never assumed.
"""

from __future__ import annotations

import numpy as np
import torch

# Ungated redistribution of the DINOv3 weights, under the DINOv3 License with
# LICENSE.md shipped alongside — tried first, so §4.4 gets DINOv3 rather than
# its fallback. ViT-S/16 matches dinov2-small's scale, keeping the comparison
# between the two a comparison of pre-training rather than of capacity.
DINOV3_TIMM = "vit_small_patch16_dinov3.lvd1689m"
# The official repos, in case the gated-access request in CLUSTER_HANDOFF.md's
# manual-action list ever clears. Kept so the path is one line from working.
DINOV3_HF = ("facebook/dinov3-vits16-pretrain-lvd1689m",)
DINOV2_FALLBACK = "facebook/dinov2-small"

# Standard B-mode display range. Wide enough to keep speckle texture, narrow
# enough that the quantisation still resolves it.
DYNAMIC_RANGE_DB = 60.0
# The corpus-level 0 dB reference. A high percentile rather than the max, so a
# single specular reflector cannot set the scale for every crop in the corpus.
REFERENCE_PERCENTILE = 99.0
# ImageNet statistics — what both DINOv2 and DINOv3 were normalised with
# (timm's `resolve_data_config` reports exactly these for the DINOv3 weights).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def assert_same_crops(a, b) -> None:
    """Raise unless two batches are the same crops in the same order.

    The external baselines are encoded from a `k=1` batch but probed against the
    *main* batch's labels, so that every encoder in the table is scored against
    one definition of every target. That is only valid if the two batches align
    row for row.
    """
    if len(a) != len(b):
        raise ValueError(f"crop counts differ: {len(a)} vs {len(b)}")
    if not np.array_equal(np.asarray(a.frame_id), np.asarray(b.frame_id)):
        raise ValueError("frame_id order differs between the K=1 and main batches")
    if a.origins is not None and b.origins is not None:
        if not np.array_equal(a.origins, b.origins):
            raise ValueError(
                "crop origins differ between the K=1 and main batches — the "
                "external baseline would be encoding different pixels than the "
                "labels it is probed against describe"
            )


def bmode_from_crops(
    crops: np.ndarray,
    reference: float | None = None,
    dynamic_range_db: float = DYNAMIC_RANGE_DB,
) -> tuple[np.ndarray, float]:
    """``(N, 2, H, W)`` K=1 I/Q crops -> ``(N, H, W)`` B-mode in ``[0, 1]``.

    Args:
        crops: K=1 crops, channel 0 real and channel 1 imaginary.
        reference: the 0 dB level. Computed once over `crops` when omitted, so
            it is a **corpus** constant — never per image (see trap 1).
        dynamic_range_db: displayed range below `reference`.

    Returns:
        `(images, reference)` — the reference is returned so a caller can reuse
        one corpus's level for a later pass and record it in its results.
    """
    env = np.sqrt(
        crops[:, 0].astype(np.float64) ** 2 + crops[:, 1].astype(np.float64) ** 2
    )
    if reference is None:
        reference = float(np.percentile(env, REFERENCE_PERCENTILE))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f"non-positive B-mode reference level {reference!r}")
    db = 20.0 * np.log10(np.maximum(env, 1e-12) / reference)
    img = (db + dynamic_range_db) / dynamic_range_db
    return np.clip(img, 0.0, 1.0).astype(np.float32), reference


class _TimmBackbone:
    """timm ViT — used for the ungated DINOv3 weights."""

    def __init__(self, model_id: str, device: str):
        import timm

        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)
        self.model = self.model.to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.input_size = int(cfg["input_size"][-1])
        # 1 CLS + N registers for DINOv3; read, never assumed (trap 3).
        self.n_prefix = int(getattr(self.model, "num_prefix_tokens", 1))

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)


class _HfBackbone:
    """HuggingFace `AutoModel` — used for DINOv2, and for DINOv3 if ungated."""

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.input_size = 224
        self.n_prefix = 1

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).last_hidden_state


def load_backbone(model_id: str | None, device: str) -> tuple[object, str, dict]:
    """First loadable backbone, preferring DINOv3. Returns `(backbone, id, skipped)`."""
    if model_id:
        candidates = [(model_id, _TimmBackbone if "/" not in model_id else _HfBackbone)]
    else:
        candidates = [
            (DINOV3_TIMM, _TimmBackbone),
            *[(m, _HfBackbone) for m in DINOV3_HF],
            (DINOV2_FALLBACK, _HfBackbone),
        ]
    skipped: dict[str, str] = {}
    for candidate, cls in candidates:
        try:
            return cls(candidate, device), candidate, skipped
        except Exception as exc:  # gated repo, missing weights, no network
            skipped[candidate] = f"{type(exc).__name__}: {str(exc)[:120]}"
    raise RuntimeError(f"no external backbone could be loaded: {skipped}")


class FrozenViTEncoder:
    """Frozen natural-image ViT over log-compressed B-mode (PLAN.md §4.4 #5).

    Features are `CLS` concatenated with the mean of the **patch** tokens, which
    is the combination DINOv2's own linear-evaluation protocol reports as
    strongest. Width does not affect fairness: `probes.capacity_control`
    PCA-projects every encoder to 64 dimensions before fitting, so this cannot
    win on dimensionality. Giving the ceiling its best form is the right
    disposition for a baseline whose job is to be hard to beat.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        batch_size: int = 64,
        reference: float | None = None,
    ):
        self.backbone, self.model_id, self.skipped = load_backbone(model_id, device)
        if self.skipped:
            print(f"[external] skipped {list(self.skipped)} -> using {self.model_id}")
        print(f"[external] {self.model_id} @ {self.backbone.input_size}px, "
              f"{self.backbone.n_prefix} prefix token(s)")
        self.device = device
        self.batch_size = batch_size
        self.reference = reference

    @torch.no_grad()
    def __call__(self, crops: np.ndarray) -> np.ndarray:
        images, self.reference = bmode_from_crops(crops, reference=self.reference)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        size = self.backbone.input_size
        n_prefix = self.backbone.n_prefix
        out = []
        for start in range(0, len(images), self.batch_size):
            chunk = torch.from_numpy(images[start : start + self.batch_size])
            x = chunk.unsqueeze(1).to(self.device)  # (B, 1, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(size, size), mode="bicubic", align_corners=False
            ).clamp_(0.0, 1.0)
            x = (x.repeat(1, 3, 1, 1) - mean) / std  # grey -> 3 channels, §4.4
            tokens = self.backbone.tokens(x)
            feats = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(dim=1)], dim=1)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out)
