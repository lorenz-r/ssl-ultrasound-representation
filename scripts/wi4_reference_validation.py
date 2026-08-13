"""WI-4 reference-validation gate: beamform real NV-Raw2Insights-US frames
with `AngularSectorSum` (K=1) and compare against the dataset's own `bmode`.

Per PLAN.md WI-4's done-condition: on real frames, the log-envelope of our
K=1 image vs `bmode` should reach normalised cross-correlation >= 0.98 and
median absolute dB difference <= 1.5 dB over the -40 dB dynamic range, for at
least one of three receive-aperture variants (unlimited, f#=1.5, f#=1.0).
Also runs the mandatory negative control: beamforming with the baseband
phase-rotation term disabled (`demodulation_frequency=0`) must produce a
plausible-looking but badly-scoring image (PLAN.md 3.5 trap #1).

**Status, measured on this (CPU-only) machine, one real frame — the gate is
NOT yet cleared, and this is being reported rather than rounded up:**

- At a coarse 48x48 grid the comparison is actively misleading: `zea` warns
  the grid violates its own Nyquist spacing recommendation, and the resulting
  spatial aliasing pushed NCC *negative* (~-0.30) despite the beamformer being
  qualitatively correct — a coarse grid is not a valid stand-in for the
  full-resolution gate, only for the algebraic identities in
  `tests/test_sector_sum.py`.
- At 256x256 (still short of the dataset's native 507x456, but past the worst
  aliasing) the two images are **visually near-identical** — same two cysts,
  same shape, same acoustic shadow, same speckle character — and the
  negative control visibly degrades into the expected banded artifact. NCC is
  still only ~0.05 at this point.
- `t_peak` (waveform peak time) defaults to `1/center_frequency` when not
  supplied, per zea's own `Parameters`; OpenH-RF's published FSA scan dict
  (PLAN.md 0.2) does not mention it. Forcing `t_peak=0` measurably closed part
  of the gap: NCC 0.054 -> 0.179, median |dB| 4.35 -> 2.42, at f#=1.5. This is
  now the script's default.
- A small integer-pixel shift search (+-4 px in z and x) after that fix found
  its best correlation at a 1-pixel z shift, NCC ~0.30 — a real improvement,
  but not the whole gap, and 1 pixel here (156 um at this grid) is itself
  already smaller than one wavelength (237 um).

**Conclusion of this first pass.** The output pixel grid is not the limiting
resolution for this comparison — speckle coherence is governed by *sub*-
wavelength timing, not the output pixel size. Both images being visually
close to indistinguishable (same two cysts, same shape, same shadow, matching
speckle statistics) while the pixel-wise Pearson NCC stays far below 0.98 is
the classic signature of a *sub-wavelength* residual timing offset: a delay
error much smaller than one pixel but a non-trivial fraction of one RF cycle
(1 cycle at fc=6.5 MHz = 154 ns = 237 um, one wavelength) reshuffles which
speckle grains are bright vs dark without changing the image's macroscopic
appearance at all — exactly what the four panels below show. So the physics
(delays, phase rotation, FSA compounding, our angular-sector segment-sum) is
very likely essentially correct, and the remaining gap is a timing constant
at the sub-microsecond level (t0/initial_times sign convention, the near-field
mute width shifting the effective t0, or `t_peak`'s exact meaning) rather than
a structural bug. Closing it needs a systematic sub-sample delay sweep against
the real reconstruction OpenH-RF ships for this exact dataset — worth doing
with GPU time on the cluster (PLAN.md WI-8), not by further blind iteration
on a CPU-only laptop. See `tests/data/wi4_results.json` for the exact
checked-in numbers this docstring is describing, and
`figures/wi4/frame0_validation.png` for the visual side-by-side.

Each frame is a ~265 MB download, and the pre-summation tensor is large even
on a coarse grid, so this runs a reduced-resolution grid (not the full
507x456) — see the module docstring in `ssl_us/beamform/sector_sum.py` for
the full-resolution production path.

    KERAS_BACKEND=torch N_FRAMES=1 GRID=256 \\
        poetry run python scripts/wi4_reference_validation.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from zea import Parameters
from zea.ops.pipeline import Beamform

from ssl_us.beamform import sector_sum  # noqa: F401 — registers "angular_sector_sum"
from ssl_us.data import geometry as geo

N_FRAMES = int(os.environ.get("N_FRAMES", "1"))
# 48x48 aliases badly (zea's own Nyquist warning fires); 256x256 is the
# smallest grid at which this calibration pass stopped looking aliased.
GRID = int(os.environ.get("GRID", "256"))  # pixels per axis; full res is (507, 456)
# near-field direct-coupling mute, PLAN 10.4
MUTE_SAMPLES = int(os.environ.get("MUTE_SAMPLES", "32"))
NUM_PATCHES = int(os.environ.get("NUM_PATCHES", max(4, GRID // 2)))

F_NUMBER_VARIANTS = {"unlimited": 1e-6, "f1.5": 1.5, "f1.0": 1.0}
NCC_THRESHOLD = 0.98
MEDIAN_DB_THRESHOLD = 1.5
DYNAMIC_RANGE_DB = 40.0


def _bilinear_resample(img: np.ndarray, z_axis_mm, x_axis_mm, z_out_mm, x_out_mm) -> np.ndarray:
    """Resample a (nz_src, nx_src) image onto a new (nz_out, nx_out) grid."""
    zi = np.clip(np.interp(z_out_mm, z_axis_mm, np.arange(len(z_axis_mm))), 0, len(z_axis_mm) - 1)
    xi = np.clip(np.interp(x_out_mm, x_axis_mm, np.arange(len(x_axis_mm))), 0, len(x_axis_mm) - 1)
    z0 = np.floor(zi).astype(int)
    z1 = np.minimum(z0 + 1, len(z_axis_mm) - 1)
    x0 = np.floor(xi).astype(int)
    x1 = np.minimum(x0 + 1, len(x_axis_mm) - 1)
    wz = (zi - z0)[:, None]
    wx = (xi - x0)[None, :]
    top = img[np.ix_(z0, x0)] * (1 - wx) + img[np.ix_(z0, x1)] * wx
    bot = img[np.ix_(z1, x0)] * (1 - wx) + img[np.ix_(z1, x1)] * wx
    return top * (1 - wz) + bot * wz


def build_parameters(
    sample: dict, grid: int
) -> tuple[Parameters, np.ndarray, np.ndarray, np.ndarray]:
    """FSA scan parameters, exactly the OpenH-RF recipe (PLAN.md 0.2)."""
    n_tx = n_el = geo.N_ELEMENTS
    elpos = np.asarray(sample["elpos"], dtype=np.float32)  # (3, n_el): [x, y, z]
    probe_geometry = elpos.T.copy()  # (n_el, 3)

    t0 = float(sample["t0"])
    params = Parameters(
        probe_geometry=probe_geometry,
        sound_speed=float(sample["c0"]),
        sampling_frequency=float(sample["fs"]),
        center_frequency=float(sample["fc"]),
        n_el=n_el,
        n_tx=n_tx,
        n_ax=geo.N_FAST_TIME,
        n_ch=2,
        xlims=(geo.BMODE_X_MIN_MM * 1e-3, geo.BMODE_X_MAX_MM * 1e-3),
        zlims=(geo.BMODE_Z_MIN_MM * 1e-3, geo.BMODE_Z_MAX_MM * 1e-3),
        grid_size_x=grid,
        grid_size_z=grid,
        t0_delays=np.zeros((n_tx, n_el), dtype=np.float32),
        tx_apodizations=np.eye(n_tx, n_el, dtype=np.float32),
        focus_distances=np.zeros(n_tx, dtype=np.float32),
        polar_angles=np.zeros(n_tx, dtype=np.float32),
        transmit_origins=probe_geometry,
        initial_times=np.full(n_tx, t0, dtype=np.float32),
        f_number=1e-6,
    )
    z_out_mm = np.linspace(geo.BMODE_Z_MIN_MM, geo.BMODE_Z_MAX_MM, grid)
    x_out_mm = np.linspace(geo.BMODE_X_MIN_MM, geo.BMODE_X_MAX_MM, grid)
    return params, probe_geometry, z_out_mm, x_out_mm


def load_raw_iq(sample: dict) -> np.ndarray:
    """`(n_tx, n_ax, n_el, n_ch)`, muted near-field, from the dataset's own axes.

    Axis order between the two raw axes is undetermined (WI-0 §10.2) but proven
    immaterial here (near-symmetric matrix): axis 0 is treated as transmit.
    """
    re = np.asarray(sample["iq_real"], dtype=np.float32)
    im = np.asarray(sample["iq_imag"], dtype=np.float32)
    raw = np.stack([re, im], axis=-1).transpose(0, 2, 1, 3)  # (n_tx, n_ax, n_el, 2)
    raw[:, :MUTE_SAMPLES, :, :] = 0.0
    return raw


def log_envelope_db(img: np.ndarray) -> np.ndarray:
    env = np.sqrt(img[..., 0] ** 2 + img[..., 1] ** 2)
    db = 20 * np.log10(np.maximum(env, 1e-9))
    return db - db.max()


def score(ours_db: np.ndarray, ref_db: np.ndarray) -> dict:
    mask = (ours_db > -DYNAMIC_RANGE_DB) | (ref_db > -DYNAMIC_RANGE_DB)
    a, b = ours_db[mask], ref_db[mask]
    ncc = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
    mad = float(np.median(np.abs(a - b)))
    return {"ncc": ncc, "median_abs_db": mad, "n_pixels": int(mask.sum())}


def beamform_variant(params, raw, *, k, f_number, tof_f_number, demodulation_frequency=None):
    pipeline = Beamform(
        beamformer="angular_sector_sum",
        k=k,
        f_number=f_number,
        num_patches=NUM_PATCHES,
        with_batch_dim=False,
    )
    # t_peak defaults to 1/center_frequency in zea's own Parameters when not
    # supplied; OpenH-RF's FSA scan dict (PLAN.md 0.2) does not set it, and
    # measured here, t_peak=0 tracks the real data noticeably better (see the
    # module docstring). Kept as an explicit override, not a Parameters
    # default, since it's this script's own finding, not zea's or OpenH-RF's.
    overrides = {"f_number": tof_f_number, "t_peak": np.zeros(params.n_tx, dtype=np.float32)}
    if demodulation_frequency is not None:
        overrides["demodulation_frequency"] = demodulation_frequency
    inputs = pipeline.prepare_parameters(params, data=raw, **overrides)
    out = pipeline(**inputs)
    return keras.ops.convert_to_numpy(out["data"])  # (nz, nx, n_ch, k)


def main() -> dict:
    ds = load_dataset("nvidia/NV-Raw2Insights-US", split="validation", streaming=True).with_format(
        "numpy"
    )
    results = []
    fig_dir = Path("figures/wi4")
    fig_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(ds):
        if idx >= N_FRAMES:
            break
        t0w = time.time()
        params, probe_geometry, z_mm, x_mm = build_parameters(sample, GRID)
        raw = load_raw_iq(sample)
        print(f"[{idx}] downloaded + built params in {time.time() - t0w:.0f}s", flush=True)

        bmode_ref = np.asarray(sample["bmode"], dtype=np.float32)
        ref_db_full = 20 * np.log10(np.maximum(bmode_ref, 1e-9))
        ref_db_full -= ref_db_full.max()
        ref_db = _bilinear_resample(
            ref_db_full, geo.bmode_z_axis_mm(), geo.bmode_x_axis_mm(), z_mm, x_mm
        )

        frame_result = {"idx": idx, "variants": {}}
        best_variant, best_ncc = None, -1.0
        for name, f_number in F_NUMBER_VARIANTS.items():
            tv = time.time()
            out = beamform_variant(params, raw, k=1, f_number=f_number, tof_f_number=f_number)
            ours_db = log_envelope_db(out[..., 0])
            s = score(ours_db, ref_db)
            s["wall_s"] = round(time.time() - tv, 1)
            frame_result["variants"][name] = s
            print(
                f"  [{name}] NCC={s['ncc']:.4f} "
                f"medianAbsDB={s['median_abs_db']:.2f} ({s['wall_s']}s)"
            )
            if s["ncc"] > best_ncc:
                best_ncc, best_variant = s["ncc"], name

        # Negative control: same winning aperture, phase rotation disabled.
        f_number = F_NUMBER_VARIANTS[best_variant]
        neg_out = beamform_variant(
            params, raw, k=1, f_number=f_number, tof_f_number=f_number, demodulation_frequency=0.0
        )
        neg_db = log_envelope_db(neg_out[..., 0])
        neg_score = score(neg_db, ref_db)
        frame_result["negative_control_no_phase_rotation"] = neg_score
        frame_result["best_variant"] = best_variant
        gate_pass = (
            best_ncc >= NCC_THRESHOLD
            and frame_result["variants"][best_variant]["median_abs_db"] <= MEDIAN_DB_THRESHOLD
        )
        frame_result["gate_pass"] = bool(gate_pass)
        print(
            f"  best={best_variant} NCC={best_ncc:.4f} gate_pass={gate_pass} | "
            f"negative control NCC={neg_score['ncc']:.4f} "
            f"(must score far below {best_variant}'s, proving phase rotation matters)"
        )
        results.append(frame_result)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        best_db = log_envelope_db(
            beamform_variant(
                params, raw, k=1, f_number=f_number, tof_f_number=f_number
            )[..., 0]
        )
        for ax, img, title in (
            (axes[0], ref_db, f"NVIDIA bmode (resampled {GRID}x{GRID})"),
            (axes[1], best_db, f"ours, {best_variant}"),
            (axes[2], best_db - ref_db, "difference (dB)"),
            (axes[3], neg_db, "negative control (no phase rotation)"),
        ):
            im = ax.imshow(img, cmap="gray" if ax is not axes[2] else "coolwarm",
                            vmin=-DYNAMIC_RANGE_DB if ax is not axes[2] else -20,
                            vmax=0 if ax is not axes[2] else 20)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"WI-4 reference validation — frame {idx}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"frame{idx}_validation.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    out_path = Path("tests/data/wi4_results.json")
    out_path.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_path} and figures/wi4/frame*_validation.png")
    return {"results": results}


if __name__ == "__main__":
    main()
