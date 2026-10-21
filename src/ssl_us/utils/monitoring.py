"""Weights & Biases monitoring, ported from the old repo (PLAN.md 1.2 / 5.1).

`TrainingMonitor` centralises scalar logging, arbitrary-figure logging, and
checkpoint-artifact upload with pruning. Everything degrades gracefully: if
wandb is unavailable or disabled, training proceeds unlogged.

The old version's reconstruction-snapshot hook was coupled to the Stanford
`ZoneBeamformer` and a fixed `(N, transducer, scanline, 2, sample)` tensor
layout — both gone with the old data path. It is replaced here by
:meth:`log_figure`, a data-agnostic hook: callers (e.g. the VICReg/MAE
instrumentation in `train/instrument.py`) build whatever matplotlib figure is
appropriate for the current arm and hand it over to be logged and closed.

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

# wandb caches a copy of every uploaded artifact (each checkpoint is the full
# model+optimizer state) under WANDB_CACHE_DIR and stages artifacts under
# WANDB_DATA_DIR; both default to ~/.cache/wandb. In the enroot training
# container ~ lives on a RAM-backed overlay, so that cache silently grows to
# many GB of resident RAM that survive the run and eat into the OOM budget.
# Default both to local scratch (/tmp is ext4 there); an explicitly set env
# var still wins.
DEFAULT_WANDB_CACHE_DIR = "/tmp/wandb-cache"
DEFAULT_WANDB_DATA_DIR = "/tmp/wandb-data"


class TrainingMonitor:
    def __init__(
        self,
        config: dict,
        enabled: bool = True,
        project: str = "ssl-us",
        run_name: str | None = None,
    ):
        self.run = None
        self._wandb = None
        if enabled:
            # Keep wandb's artifact cache/staging off the RAM-backed container
            # overlay (~/.cache/wandb) — set before importing wandb so it
            # picks these up. setdefault keeps any explicit env override
            # authoritative.
            os.environ.setdefault("WANDB_CACHE_DIR", DEFAULT_WANDB_CACHE_DIR)
            os.environ.setdefault("WANDB_DATA_DIR", DEFAULT_WANDB_DATA_DIR)
            for var in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
                Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
            try:
                import wandb

                self.run = wandb.init(project=project, name=run_name, config=config)
                self._wandb = wandb
            except Exception as exc:  # offline cluster node, bad key, ...
                warnings.warn(f"wandb disabled ({exc}); training continues unlogged.", stacklevel=2)

    @property
    def enabled(self) -> bool:
        return self.run is not None

    # ── scalars ──────────────────────────────────────────────────────────────
    def log_epoch(self, epoch: int, metrics: dict) -> None:
        if self.run is not None:
            self.run.log({"epoch": epoch, **metrics}, step=epoch)

    # ── figures / images ─────────────────────────────────────────────────────
    def log_figure(self, epoch: int, key: str, fig, *, caption: str | None = None) -> None:
        """Log one matplotlib figure under `key` and close it.

        Generic replacement for the old beamform-specific reconstruction
        snapshot: callers build whatever figure suits the current arm (VICReg
        singular-value spectrum, MAE reconstruction grid, cross-aperture
        prediction vs target, ...) and hand it straight to this method.
        """
        if self.run is None:
            plt.close(fig)
            return
        self.run.log({key: self._wandb.Image(fig, caption=caption)}, step=epoch)
        plt.close(fig)

    def log_histogram(self, epoch: int, key: str, values) -> None:
        if self.run is None:
            return
        self.run.log({key: self._wandb.Histogram(values)}, step=epoch)

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
        artifact = self._wandb.Artifact(name=name, type="model", metadata=metadata or {})
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
            warnings.warn(f"checkpoint pruning skipped ({exc})", stacklevel=2)
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
                warnings.warn(f"could not delete artifact {art.version} ({exc})", stacklevel=2)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def finish(self, summary: dict | None = None) -> None:
        if self.run is not None:
            if summary:
                for k, v in summary.items():
                    self.run.summary[k] = v
            self.run.finish()
