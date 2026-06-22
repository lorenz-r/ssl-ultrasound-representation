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

    def bmode(
        self,
        frame: np.ndarray,
        *,
        gain: np.ndarray | None = None,
        ref: float | None = None,
        db_range: float = 50.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Beamform one frame into a log-compressed B-mode image (nz, nx) in
        [-db_range, 0].

        `gain` (a TGC curve) and `ref` (the 0 dB envelope level) default to being
        computed from this frame; pass the values returned for an *anchor* frame
        to put another frame (e.g. a reconstruction) on exactly that scale, so
        amplitude errors stay visible instead of being normalised away.

        Returns `(db_image, gain, ref)` — feed `gain`/`ref` back in for the paired
        frame, or cache them for a fixed anchor across epochs.
        """
        env = np.abs(self.beamform(frame))
        if gain is None:
            _, gain = pymust.tgc(env)
        env = np.maximum(env * gain, 1e-9)
        if ref is None:
            ref = float(env.max())
        db = np.clip(20.0 * np.log10(env / ref), -db_range, 0)
        return db, gain, ref

    def bmode_pair(
        self, gt_frame: np.ndarray, recon_frame: np.ndarray, db_range: float = 50.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Beamform a ground-truth / reconstruction pair into comparable B-mode
        images: the TGC gain and the 0 dB reference are computed from the ground
        truth and applied to both. Thin wrapper over `bmode`.

        Returns (gt_db, recon_db), each (nz, nx) in [-db_range, 0].
        """
        gt_db, gain, ref = self.bmode(gt_frame, db_range=db_range)
        rec_db, _, _ = self.bmode(recon_frame, gain=gain, ref=ref, db_range=db_range)
        return gt_db, rec_db
