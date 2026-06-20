"""
DAS beamforming of processed zarr IQ frames (PyMUST), for reconstruction
monitoring during training.

Uses the acquisition geometry exported by
`scripts/dataset_scripts/export_beamform_params.py`. The per-scanline DAS
matrices depend only on the (fixed) imaging grid and geometry, so they are
built once and cached — beamforming a frame afterwards is a handful of sparse
matrix-vector products (fast enough to run every few epochs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pymust


class ZoneBeamformer:
    """
    DAS-beamform single-focal-zone IQ frames `(transducer, scanline, iq, sample)`
    into log-compressed B-mode images, mirroring the strip-compounding scheme of
    `scripts/dataset_scripts/visualize_beamformed_pymust.py`.
    """

    def __init__(
        self,
        params_path: str | os.PathLike,
        focal_zone: int = 0,
        nx: int = 192,
        nz: int = 256,
        strip_halfwidth_m: float = 1.5e-3,
        fnumber: float = 1.5,
    ):
        p = json.loads(Path(params_path).read_text())
        self.focal_zone = focal_zone
        self.n_elem = p["n_elements"]
        self.n_samples = p["n_samples"]
        self.origins = np.array(p["scanline_origins_m"])
        self.strip_halfwidth_m = strip_halfwidth_m

        delays = p["tx_delays_s"][focal_zone]  # [scanline][element], None = inactive
        self.tx_delays = np.array(
            [[np.nan if d is None else d for d in row] for row in delays]
        )

        param = pymust.utils.Param()
        param.fs = p["fs"]
        param.fc = p["fc"]
        param.c = p["c"]
        param.pitch = p["pitch"]
        param.Nelements = p["n_elements"]
        param.t0 = np.array([[p["t0"]]])
        param.fnumber = fnumber
        self.param = param

        self.x_grid = np.linspace(self.origins[0], self.origins[-1], nx)
        self.z_grid = np.linspace(p["start_depth_m"], p["data_depth_limit_m"], nz)
        self.nx, self.nz = nx, nz

        self._plans: list[tuple[np.ndarray, object, np.ndarray]] | None = None

    def _build_plans(self) -> None:
        """Precompute (columns, DAS matrix, lateral weights) per scanline."""
        # PyMUST convention: imaginary shape flags complex (IQ) input
        iq_shape = 1j * np.array([self.n_samples, self.n_elem])
        plans = []
        for sl, ox in enumerate(self.origins):
            cols = np.where(np.abs(self.x_grid - ox) <= self.strip_halfwidth_m)[0]
            if cols.size == 0:
                plans.append((cols, None, np.empty(0)))
                continue
            xs = self.x_grid[cols]
            X, Z = np.meshgrid(xs, self.z_grid)
            M = pymust.dasmtx(iq_shape, X, Z, self.tx_delays[sl], self.param)
            w = (
                0.5 * (1.0 + np.cos(np.pi * (xs - ox) / self.strip_halfwidth_m))
            ).astype(np.float32)
            plans.append((cols, M, w))
        self._plans = plans

    def beamform(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: (transducer, scanline, 2, sample) float array, single focal
                zone (any global scaling is fine — DAS is linear).

        Returns:
            Complex image (nz, nx).
        """
        if self._plans is None:
            self._build_plans()
        assert self._plans is not None

        img = np.zeros((self.nz, self.nx), dtype=np.complex64)
        wts = np.zeros((self.nz, self.nx), dtype=np.float32)
        for sl, (cols, M, w) in enumerate(self._plans):
            if M is None:
                continue
            iq = (frame[:, sl, 0] + 1j * frame[:, sl, 1]).T  # (sample, elem)
            bf = (M @ iq.flatten(order="F")).reshape((self.nz, cols.size), order="F")
            img[:, cols] += bf * w[np.newaxis, :]
            wts[:, cols] += w[np.newaxis, :]
        np.divide(img, wts, out=img, where=wts > 0)
        return img

    def bmode_pair(
        self, gt_frame: np.ndarray, recon_frame: np.ndarray, db_range: float = 50.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Beamform a ground-truth / reconstruction pair into comparable B-mode
        images: the TGC gain and the 0 dB reference are computed from the
        ground truth and applied to both, so amplitude errors in the
        reconstruction stay visible instead of being normalised away.

        Returns (gt_db, recon_db), each (nz, nx) in [-db_range, 0].
        """
        env_gt = np.abs(self.beamform(gt_frame))
        env_rec = np.abs(self.beamform(recon_frame))

        _, C = pymust.tgc(env_gt)
        env_gt = np.maximum(env_gt * C, 1e-9)
        env_rec = np.maximum(env_rec * C, 1e-9)

        ref = env_gt.max()
        gt_db = np.clip(20.0 * np.log10(env_gt / ref), -db_range, 0)
        rec_db = np.clip(20.0 * np.log10(env_rec / ref), -db_range, 0)
        return gt_db, rec_db
