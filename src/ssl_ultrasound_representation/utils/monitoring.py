"""
Weights & Biases monitoring for MAE training.

`TrainingMonitor` centralises all experiment tracking:
  * scalar metrics per epoch (losses, learning rate, gradient norm)
  * periodic beamformed reconstruction snapshots: model output vs. ground
    truth B-mode (PyMUST DAS), plus their dB difference

Everything degrades gracefully: if wandb is unavailable / disabled or the
beamform geometry file is missing, training proceeds without logging.

Authentication: run `wandb login` once, or set WANDB_API_KEY (preferred on
the cluster). Never commit the key.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..eval.metrics import compute_all_metrics
from .beamform import ZoneBeamformer

# wandb caches a copy of every uploaded artifact (each checkpoint is the full
# model+optimizer state) under WANDB_CACHE_DIR and stages artifacts under
# WANDB_DATA_DIR; both default to ~/.cache/wandb. In the enroot training
# container ~ lives on a RAM-backed overlay, so that cache silently grows to
# many GB of resident RAM that survive the run and eat into the OOM budget.
# Default both to local scratch (/tmp is ext4 here, like HF_CACHE_DIR); an
# explicitly set env var still wins.
DEFAULT_WANDB_CACHE_DIR = "/tmp/wandb-cache"
DEFAULT_WANDB_DATA_DIR = "/tmp/wandb-data"


class TrainingMonitor:
    def __init__(
        self,
        config: dict,
        enabled: bool = True,
        project: str = "ssl-ultrasound-mae",
        run_name: str | None = None,
        beamform_params: str = "configs/beamform_params.json",
        focal_zone: int = 0,
    ):
        self.run = None
        self._wandb = None
        if enabled:
            # Keep wandb's artifact cache/staging off the RAM-backed container
            # overlay (~/.cache/wandb) — set before importing wandb so it picks
            # these up. setdefault keeps any explicit env override authoritative.
            os.environ.setdefault("WANDB_CACHE_DIR", DEFAULT_WANDB_CACHE_DIR)
            os.environ.setdefault("WANDB_DATA_DIR", DEFAULT_WANDB_DATA_DIR)
            for var in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
                Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
            try:
                import wandb

                self.run = wandb.init(project=project, name=run_name, config=config)
                self._wandb = wandb
            except Exception as exc:  # offline cluster node, bad key, ...
                warnings.warn(f"wandb disabled ({exc}); training continues unlogged.")

        self._beamformer: ZoneBeamformer | None = None
        if Path(beamform_params).exists():
            self._beamformer = ZoneBeamformer(beamform_params, focal_zone=focal_zone)
        else:
            warnings.warn(
                f"{beamform_params} not found — reconstruction snapshots disabled. "
                "Generate it with scripts/dataset_scripts/export_beamform_params.py."
            )

        # Fixed vis frames (set via set_vis_frames); GT B-mode cached lazily.
        self._vis_frames: torch.Tensor | None = None
        self._vis_labels: list[str] = []
        self._gt_db: list[np.ndarray] | None = None

    @property
    def enabled(self) -> bool:
        return self.run is not None

    # ── scalars ──────────────────────────────────────────────────────────────
    def log_epoch(self, epoch: int, metrics: dict) -> None:
        if self.run is not None:
            self.run.log({"epoch": epoch, **metrics}, step=epoch)

    # ── reconstruction snapshots ─────────────────────────────────────────────
    def set_vis_frames(self, frames: torch.Tensor, labels: list[str]) -> None:
        """Fix the frames used for every snapshot so epochs stay comparable.

        Args:
            frames: (N, transducer, scanline, 2, sample) — already preprocessed
                exactly like training inputs (focal zone selected, normalized).
            labels: one short identifier per frame (e.g. "val[0]").
        """
        self._vis_frames = frames
        self._vis_labels = labels

    def log_reconstructions(
        self, epoch: int, model: torch.nn.Module, pos_embed: torch.Tensor, device
    ) -> None:
        if self.run is None or self._beamformer is None or self._vis_frames is None:
            return

        model.eval()
        with torch.no_grad():
            x = self._vis_frames.to(device)
            pred, _, _ = model(x, pos_embed)
            recon = model.unpatchify(pred).cpu().numpy()
        gt = self._vis_frames.numpy()

        images = []
        for i, label in enumerate(self._vis_labels):
            gt_db, rec_db = self._beamformer.bmode_pair(gt[i], recon[i])
            images.append(
                self._wandb.Image(
                    self._comparison_figure(gt_db, rec_db, f"{label} — epoch {epoch}"),
                    caption=label,
                )
            )
            plt.close("all")
        self.run.log({"reconstructions": images}, step=epoch)

    def _comparison_figure(self, gt_db, rec_db, title: str):
        bf = self._beamformer
        extent = [
            bf.x_grid[0] * 1e3,
            bf.x_grid[-1] * 1e3,
            bf.z_grid[-1] * 1e3,
            bf.z_grid[0] * 1e3,
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        for ax, img, name, kw in (
            (axes[0], gt_db, "ground truth", dict(cmap="gray", vmin=-50, vmax=0)),
            (axes[1], rec_db, "reconstruction", dict(cmap="gray", vmin=-50, vmax=0)),
            (
                axes[2],
                rec_db - gt_db,
                "difference (dB)",
                dict(cmap="coolwarm", vmin=-20, vmax=20),
            ),
        ):
            im = ax.imshow(img, extent=extent, aspect="equal", **kw)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Lateral (mm)", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[0].set_ylabel("Depth (mm)", fontsize=8)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        return fig

    # ── physics metrics ──────────────────────────────────────────────────────
    def log_physics_metrics(
        self, epoch: int, model: torch.nn.Module, pos_embed: torch.Tensor, device
    ) -> None:
        """Log physics-aware IQ reconstruction metrics (see `eval.metrics`) on
        the fixed visualisation frames: complex correlation, centre-frequency
        shift, spectral coherence, SLSC error, coherence-factor error. Unlike
        `log_reconstructions` this needs no beamform geometry, so it still runs
        when that's unavailable.
        """
        if self.run is None or self._vis_frames is None:
            return

        model.eval()
        with torch.no_grad():
            x = self._vis_frames.to(device)
            pred, _, _ = model(x, pos_embed)
            recon = model.unpatchify(pred).cpu().numpy()
        gt = self._vis_frames.numpy()

        n = gt.shape[0]
        sums: dict[str, float] = {}
        for i in range(n):
            for k, v in compute_all_metrics(gt[i], recon[i]).items():
                sums[k] = sums.get(k, 0.0) + v
        self.run.log({f"physics/val/{k}": v / n for k, v in sums.items()}, step=epoch)

    # ── checkpoints ──────────────────────────────────────────────────────────
    def log_checkpoint(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict | None = None,
        keep_last: int | None = None,
    ) -> None:
        """Upload a checkpoint file as a versioned wandb model artifact.

        No-op if wandb is disabled. The artifact defaults to the run's own name
        so it can be identified at a glance in the wandb UI. `aliases` (e.g.
        ["best", "epoch-120"]) make a version easy to fetch later; the latest
        logged version is always reachable as "<name>:latest". `metadata` is
        stored on the artifact for provenance (epoch, losses, model config) so
        the eval suite can rebuild the model without the training config on hand.

        `keep_last` prunes the artifact collection after the upload commits,
        deleting all but the `keep_last` newest versions. Each checkpoint is the
        full model+optimizer state, so without pruning a long run accumulates one
        copy per val improvement. `None` (default) keeps every version.
        """
        if self.run is None:
            return
        # wandb artifact names only allow [A-Za-z0-9._-]; run names can contain
        # other characters (e.g. spaces), so sanitize when falling back to it.
        if name is None:
            name = re.sub(r"[^A-Za-z0-9._-]+", "-", self.run.name)
        artifact = self._wandb.Artifact(
            name=name, type="model", metadata=metadata or {}
        )
        artifact.add_file(str(path))
        logged = self.run.log_artifact(artifact, aliases=aliases or ["latest"])
        if keep_last is not None:
            # Must wait for the server to commit this version before we can list
            # and prune; otherwise the just-logged "best" could be missed or the
            # collection enumerated stale.
            logged.wait()
            self._prune_checkpoints(name, keep_last)

    def _prune_checkpoints(self, name: str, keep_last: int) -> None:
        """Delete all but the `keep_last` newest versions of the `name` model
        artifact. Degrades to a warning if the public API is unavailable so a
        pruning hiccup never aborts training."""
        try:
            api = self._wandb.Api()
            collection = f"{self.run.entity}/{self.run.project}/{name}"
            versions = list(api.artifacts(type_name="model", name=collection))
        except Exception as exc:  # noqa: BLE001 — never fail training on cleanup
            warnings.warn(f"checkpoint pruning skipped ({exc})")
            return

        def _version_num(art) -> int:
            try:
                return int(str(art.version).lstrip("v"))
            except (ValueError, AttributeError):
                return -1

        versions.sort(key=_version_num, reverse=True)  # newest first
        for art in versions[keep_last:]:
            try:
                # delete_aliases drops the per-epoch alias each version carries;
                # "best"/"latest" live on the newest (kept) version, so they move
                # with the collection rather than getting deleted here.
                art.delete(delete_aliases=True)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"could not delete artifact {art.version} ({exc})")

    # ── lifecycle ────────────────────────────────────────────────────────────
    def finish(self, summary: dict | None = None) -> None:
        if self.run is not None:
            if summary:
                for k, v in summary.items():
                    self.run.summary[k] = v
            self.run.finish()
