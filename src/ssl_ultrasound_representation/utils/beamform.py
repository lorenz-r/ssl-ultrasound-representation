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
        tx_step: int | None = None,
    ):
        p = json.loads(Path(params_path).read_text())
        self.acquisition_type = p.get("acquisition_type", "multifocal")
        self.beamforming_mode = p.get("beamforming_mode", "strip_compound")
        self.tx_step = max(
            int(tx_step if tx_step is not None else p.get("default_tx_step", 1)), 1
        )
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
        self._full_XZ: tuple[np.ndarray, np.ndarray] | None = None

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

    def _full_grid(self) -> tuple[np.ndarray, np.ndarray]:
        if self._full_XZ is None:
            self._full_XZ = np.meshgrid(self.x_grid, self.z_grid)
        return self._full_XZ

    def _beamform_fsa_pair(
        self, gt_frame: np.ndarray, recon_frame: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Synthetic-aperture FSA beamforming.

        FSA stores one TX event on the zarr scanline axis. Each TX contributes
        to the full image grid, so caching all full-grid DAS matrices would be
        memory-heavy; build each TX matrix, apply it to the requested frame(s),
        and immediately release it.
        """
        X, Z = self._full_grid()
        iq_shape = 1j * np.array([self.n_samples, self.n_elem])
        gt_img = np.zeros((self.nz, self.nx), dtype=np.complex64)
        rec_img = (
            np.zeros((self.nz, self.nx), dtype=np.complex64)
            if recon_frame is not None
            else None
        )
        used = 0

        for sl in range(0, min(len(self.tx_delays), gt_frame.shape[1]), self.tx_step):
            delays = self.tx_delays[sl]
            if np.all(np.isnan(delays)):
                continue
            M = pymust.dasmtx(iq_shape, X, Z, delays, self.param)

            iq_gt = (gt_frame[:, sl, 0] + 1j * gt_frame[:, sl, 1]).T
            gt_img += (M @ iq_gt.flatten(order="F")).reshape(
                (self.nz, self.nx), order="F"
            )

            if recon_frame is not None and rec_img is not None:
                iq_rec = (recon_frame[:, sl, 0] + 1j * recon_frame[:, sl, 1]).T
                rec_img += (M @ iq_rec.flatten(order="F")).reshape(
                    (self.nz, self.nx), order="F"
                )
            used += 1

        if used:
            gt_img /= used
            if rec_img is not None:
                rec_img /= used
        return gt_img, rec_img

    def _beamform_fsa_many(
        self, gt_frame: np.ndarray, recon_frames: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Beamform several FSA reconstructions while building each DAS matrix once."""
        X, Z = self._full_grid()
        iq_shape = 1j * np.array([self.n_samples, self.n_elem])
        gt_img = np.zeros((self.nz, self.nx), dtype=np.complex64)
        recon_images = {
            name: np.zeros((self.nz, self.nx), dtype=np.complex64)
            for name in recon_frames
        }
        used = 0
        for sl in range(0, min(len(self.tx_delays), gt_frame.shape[1]), self.tx_step):
            delays = self.tx_delays[sl]
            if np.all(np.isnan(delays)):
                continue
            matrix = pymust.dasmtx(iq_shape, X, Z, delays, self.param)
            gt_iq = (gt_frame[:, sl, 0] + 1j * gt_frame[:, sl, 1]).T
            gt_img += (matrix @ gt_iq.flatten(order="F")).reshape(
                (self.nz, self.nx), order="F"
            )
            for name, frame in recon_frames.items():
                iq = (frame[:, sl, 0] + 1j * frame[:, sl, 1]).T
                recon_images[name] += (matrix @ iq.flatten(order="F")).reshape(
                    (self.nz, self.nx), order="F"
                )
            used += 1
        if used:
            gt_img /= used
            for image in recon_images.values():
                image /= used
        return gt_img, recon_images

    def beamform(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: (transducer, scanline, 2, sample) float array, single focal
                zone (any global scaling is fine — DAS is linear).

        Returns:
            Complex image (nz, nx).
        """
        if self.acquisition_type == "fsa" or self.beamforming_mode == "synthetic_aperture":
            img, _ = self._beamform_fsa_pair(frame)
            return img

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
        if self.acquisition_type == "fsa" or self.beamforming_mode == "synthetic_aperture":
            gt_img, rec_img = self._beamform_fsa_pair(gt_frame, recon_frame)
            assert rec_img is not None
            env_gt = np.abs(gt_img)
            env_rec = np.abs(rec_img)
        else:
            env_gt = np.abs(self.beamform(gt_frame))
            env_rec = np.abs(self.beamform(recon_frame))

        _, C = pymust.tgc(env_gt)
        env_gt = np.maximum(env_gt * C, 1e-9)
        env_rec = np.maximum(env_rec * C, 1e-9)

        ref = env_gt.max()
        gt_db = np.clip(20.0 * np.log10(env_gt / ref), -db_range, 0)
        rec_db = np.clip(20.0 * np.log10(env_rec / ref), -db_range, 0)
        return gt_db, rec_db

    def bmode_many(
        self,
        gt_frame: np.ndarray,
        recon_frames: dict[str, np.ndarray],
        db_range: float = 50.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Beamform many outputs using one common ground-truth TGC/reference."""
        if self.acquisition_type == "fsa" or self.beamforming_mode == "synthetic_aperture":
            gt_img, recon_images = self._beamform_fsa_many(gt_frame, recon_frames)
        else:
            gt_img = self.beamform(gt_frame)
            recon_images = {
                name: self.beamform(frame) for name, frame in recon_frames.items()
            }
        env_gt = np.abs(gt_img)
        _, gain = pymust.tgc(env_gt)
        env_gt = np.maximum(env_gt * gain, 1e-9)
        reference = env_gt.max()
        gt_db = np.clip(20.0 * np.log10(env_gt / reference), -db_range, 0)
        recon_db = {
            name: np.clip(
                20.0 * np.log10(np.maximum(np.abs(image) * gain, 1e-9) / reference),
                -db_range,
                0,
            )
            for name, image in recon_images.items()
        }
        return gt_db, recon_db
