"""NVIDIA reference-validation gate: beamform real NV-Raw2Insights-US frames
with `AngularSectorSum` (K=1) and validate the reconstruction against the
dataset's own references (WI-4).

**The original gate was withdrawn — read this before "fixing" the numbers.**

WI-4 was originally specified as: pixel-wise normalised cross-correlation
>= 0.98 and median |dB| <= 1.5 against NVIDIA's shipped `bmode`. That gate is
not reachable by any beamformer, including NVIDIA's own, and this was
established by measurement rather than argument:

1. **The reference disagrees with itself.** NVIDIA ships two reconstructions
   of every frame. Scored against *each other* with the very same pixel-wise
   metric, `bmode` vs `bmode_focused` average NCC 0.197 over 10 frames
   (range -0.071 to 0.510 — on frame 1 the dataset's own two reconstructions
   are *anti-correlated*); see
   `results/gates/nvidia_reference_selfconsistency.json`. Our reconstruction scores
   ~0.26 against `bmode` — that is, we already match the reference *better
   than the reference's own sibling does*. A 0.98 gate asks for agreement no
   pair of real reconstructions in this corpus achieves.
2. **It is not our beamformer.** `zea`'s stock `DelayAndSum` — the beamformer
   OpenH-RF's reference `reconstruct.py` uses for this exact dataset — scores
   0.2587 where our `AngularSectorSum` scores 0.2573, on the same frame with
   the same parameters. Two independent implementations agreeing to within
   0.001 while both sit far below the gate places the discrepancy in the
   reference, not in the code under test.
3. **`bmode` really is a constant-c DAS.** Re-running with the ground-truth
   sound-speed map through zea's heterogeneous mode (as OpenH-RF's
   `reconstruct.py` demonstrates) makes the score *worse*, 0.257 -> 0.134, so
   the constant c=1540 assumption here is correct and the residual is not
   uncorrected aberration.

The cause is physical. B-mode speckle is fully developed: it is a random
interference pattern, not an image of scatterers. Any sub-wavelength
difference between two reconstructions — interpolation kernel, apodisation, a
fraction of an RF cycle in the delay model — reshuffles which speckle grains
are bright without moving a single macroscopic feature. Pixel-wise
correlation therefore measures agreement of *speckle realisation*, which a
correct beamformer is under no obligation to reproduce. (The earlier
hypothesis recorded here — that a residual sub-wavelength timing bug was
suppressing NCC — was chasing this artifact. Two genuine timing bugs were
nonetheless found and fixed along the way; see `t_peak` below.)

**The gate this script now applies** uses metrics that survive a different
speckle draw (`ssl_us.beamform.image_quality`, which documents each one), and
is *reference-relative* rather than absolute:

- **|our gCNR - reference gCNR| <= 0.10** on the shipped `segmentation_map`
  (inclusion vs a depth-matched background) — the primary gate. gCNR is
  bounded in [0, 1] and invariant to any monotonic transform, so it asks "is
  the inclusion resolved, in the right place, with the right contrast?"
  without reference to speckle. It is graded against the reference's own gCNR
  *on the same frame* because frame difficulty varies hugely here — measured
  reference gCNR spans 0.15 (frame 0, where NVIDIA's own image barely shows
  the inclusion) to 0.86 (frame 1) — so an absolute threshold would grade the
  frame, not the beamformer.
- **speckle SNR**, in a lesion-free background band: fully-developed speckle
  has a Rayleigh envelope whose mean/std is the constant ~1.913 regardless of
  gain, medium or transducer. Catches over-smoothing, aliasing and a broken
  envelope path, none of which gCNR would notice. Checked two ways — a wide
  absolute band [1.60, 2.15] for gross failure, and agreement within 0.25 of
  the *reference's own* speckle SNR — because the absolute value legitimately
  drifts with receive aperture (measured 1.91 unlimited vs 1.70 at f#=1.5:
  narrower aperture means larger speckle grains and so fewer independent
  samples per patch). Two measurement details are load-bearing: the ratio must
  be computed **per patch** rather than pooled over the band, since attenuation
  and TGC vary the mean with depth and drag a pooled ratio down (measured 1.41
  -1.69 pooled vs ~1.89 patched on the same images); and the inclusion must be
  excised from the region, or frames with a large lesion collapse further
  still (0.93 on frame 1). The reference's own value is measured on its
  **native** grid, since resampling it onto ours would smooth it and raise the
  bar spuriously.
- **the phase-rotation negative control is retained** (PLAN.md 3.5 trap #1):
  disabling baseband phase rotation (`demodulation_frequency=0`) must
  measurably degrade gCNR. It is only applied on frames whose *reference*
  gCNR >= 0.30 — below that the inclusion is unresolved by anyone, and
  comparing two near-zero contrasts is noise rather than evidence.

Two real bugs found by the earlier calibration work, both still fixed here:
`t_peak` defaults to `1/center_frequency` in zea's `Parameters` when not
supplied and OpenH-RF's FSA scan dict does not set it — forcing `t_peak=0` is
worth NCC 0.076 -> 0.257 (measured, still the largest single effect of any
parameter tried); and the `t0` sign convention, where +t0 beats -t0 by
0.257 vs 0.078. A 16-way sweep over {t_peak, t0 sign, near-field mute,
tx/rx axis order} confirmed the current settings are the best of all
combinations, and that the tx/rx axis order is immaterial here (0.2587 vs
0.2573), as PLAN.md 10.2 predicted from reciprocity.

Each frame is a ~265 MB download. Full resolution is 507x456.

    KERAS_BACKEND=jax N_FRAMES=2 GRID=507 python scripts/beamform/validate_nvidia_reconstruction.py
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

from ssl_us.beamform import image_quality as iqm
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
DYNAMIC_RANGE_DB = 40.0

# Speckle-invariant gate (see module docstring for why the old pixel-wise
# NCC>=0.98 / median|dB|<=1.5 thresholds were withdrawn).
#
# The gate is deliberately *reference-relative*, not absolute. Frames in this
# corpus differ enormously in how visible their inclusion is — measured
# reference gCNR ranges from 0.15 (frame 0: the inclusion is near-invisible in
# NVIDIA's own image) to 0.86 (frame 1) — so any absolute contrast threshold
# would grade the frame's difficulty rather than our beamformer's correctness.
# What a correct beamformer owes us is reproducing *whatever contrast the
# reference achieved on that same frame*.
GCNR_VS_REF_TOLERANCE = 0.10
# Rayleigh envelope mean/std is ~1.913. Two checks, for the same reason gCNR
# is graded against the reference: the *absolute* value legitimately drifts
# below the ideal constant as the receive aperture narrows (larger speckle
# grains -> fewer independent samples per patch; measured 1.91 unlimited vs
# 1.70 at f#=1.5 on the same frame), so a tight absolute band would grade the
# aperture rather than the beamformer. The wide band catches gross failure
# (smoothing, aliasing, a broken envelope path); agreement with the
# reference's own speckle statistics does the real work.
SPECKLE_SNR_RANGE = (1.60, 2.15)
SPECKLE_SNR_VS_REF_TOLERANCE = 0.25
# Depth band used for the background speckle-statistics region: past the
# near-field mute, before the deep low-SNR zone. The inclusion is excised from
# it (see speckle_region) — without that, frames with a large lesion measure
# lesion+background as one population and the ratio collapses (measured: 0.93
# on frame 1, whose inclusion covers 7260 px of the band).
SPECKLE_BAND_MM = (12.0, 30.0)
# The phase-rotation negative control only carries information on frames where
# there is contrast to destroy. Below this reference gCNR the lesion is not
# resolved by anyone and the comparison is noise, so it is recorded as
# not-applicable rather than passed or failed.
NEGATIVE_CONTROL_MIN_REF_GCNR = 0.30


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


def envelope(img: np.ndarray) -> np.ndarray:
    """Linear envelope amplitude. Speckle statistics are only Rayleigh here."""
    return np.sqrt(img[..., 0] ** 2 + img[..., 1] ** 2)


def log_envelope_db(img: np.ndarray) -> np.ndarray:
    db = 20 * np.log10(np.maximum(envelope(img), 1e-9))
    return db - db.max()


def _nearest_resample_mask(mask: np.ndarray, z_src, x_src, z_out, x_out) -> np.ndarray:
    """Resample a label mask onto the output grid — nearest, never interpolated."""
    zi = np.clip(np.round(np.interp(z_out, z_src, np.arange(len(z_src)))), 0, len(z_src) - 1)
    xi = np.clip(np.round(np.interp(x_out, x_src, np.arange(len(x_src)))), 0, len(x_src) - 1)
    return mask[np.ix_(zi.astype(int), xi.astype(int))]


def lesion_background_masks(
    sample: dict, z_out_mm: np.ndarray, x_out_mm: np.ndarray, guard_px: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Inclusion mask and a depth-matched background mask, on the output grid.

    The background is drawn from the *same depth band* as the inclusion, minus
    the inclusion and a guard margin around it. Depth matching matters: TGC and
    attenuation both make contrast depth-dependent, so a background taken from
    the whole image would compare the lesion against a different acoustic
    regime and inflate gCNR.
    """
    seg = np.asarray(sample["segmentation_map"])
    seg = _nearest_resample_mask(
        seg, geo.bmode_z_axis_mm(), geo.bmode_x_axis_mm(), z_out_mm, x_out_mm
    )
    lesion = seg == 1
    if not lesion.any():
        return lesion, np.zeros_like(lesion)

    rows = np.where(lesion.any(axis=1))[0]
    band = np.zeros_like(lesion)
    band[rows.min() : rows.max() + 1, :] = True

    # Grow the lesion by `guard_px` and exclude it, so partially-shadowed
    # pixels just outside the inclusion do not count as background.
    grown = lesion.copy()
    for _ in range(guard_px):
        g = grown.copy()
        g[1:, :] |= grown[:-1, :]
        g[:-1, :] |= grown[1:, :]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    return lesion, band & ~grown


def speckle_region(
    z_out_mm: np.ndarray, shape: tuple[int, int], lesion: np.ndarray, guard_px: int = 10
) -> np.ndarray:
    """Lesion-free background band for Rayleigh speckle statistics, by depth (mm).

    Excising the inclusion (plus a guard margin, since the region just outside
    it is shadowed) is not cosmetic: the Rayleigh mean/std constant only holds
    for a single fully-developed speckle population, and mixing lesion and
    background pixels drives the ratio well below it.
    """
    lo, hi = SPECKLE_BAND_MM
    rows = (z_out_mm >= lo) & (z_out_mm <= hi)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, :] = True

    grown = lesion.copy()
    for _ in range(guard_px):
        g = grown.copy()
        g[1:, :] |= grown[:-1, :]
        g[:-1, :] |= grown[1:, :]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    return mask & ~grown


def score(
    ours_img: np.ndarray,
    ref_db: np.ndarray,
    lesion: np.ndarray,
    background: np.ndarray,
    speckle: np.ndarray,
) -> dict:
    """Speckle-invariant metrics, plus pixel-wise NCC kept as a *reported*
    diagnostic only — it is no longer part of the gate (module docstring)."""
    ours_db = log_envelope_db(ours_img)
    env = envelope(ours_img)

    mask = (ours_db > -DYNAMIC_RANGE_DB) | (ref_db > -DYNAMIC_RANGE_DB)
    a, b = ours_db[mask], ref_db[mask]
    return {
        "gcnr": iqm.gcnr(ours_db[lesion], ours_db[background]),
        "cnr_db": iqm.cnr(ours_db[lesion], ours_db[background]),
        # Per-patch, not pooled: attenuation/TGC vary the mean with depth over
        # the band, which biases a single pooled mean/std downward (see
        # image_quality.speckle_snr_patched).
        "speckle_snr": iqm.speckle_snr_patched(env, speckle, patch=32),
        "speckle_snr_pooled": iqm.speckle_snr(env[speckle]),
        "structural_ncc": iqm.structural_ncc(ours_db, ref_db, window=15),
        # Reported, not gated:
        "ncc_pixelwise": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan"),
        "median_abs_db": float(np.median(np.abs(a - b))),
        "n_pixels": int(mask.sum()),
    }


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
    fig_dir = Path("outputs/gates/nvidia_reconstruction")
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

        lesion, background = lesion_background_masks(sample, z_mm, x_mm)
        speckle = speckle_region(z_mm, (len(z_mm), len(x_mm)), lesion)
        # The reference's own gCNR: the bar our contrast is held against.
        ref_gcnr = iqm.gcnr(ref_db[lesion], ref_db[background])

        # The reference's own speckle statistics, measured on its NATIVE grid:
        # `ref_db` above was bilinearly resampled onto our grid, and that
        # interpolation correlates neighbouring pixels, which inflates the
        # mean/std ratio. Comparing against a smoothed reference would set the
        # bar wrong, so this uses the unresampled bmode and native-grid masks.
        native_lesion = np.asarray(sample["segmentation_map"]) == 1
        native_speckle = speckle_region(
            geo.bmode_z_axis_mm(), native_lesion.shape, native_lesion
        )
        ref_speckle_snr = iqm.speckle_snr_patched(
            np.asarray(sample["bmode"], dtype=np.float32), native_speckle, patch=32
        )
        frame_result_ref_snr = ref_speckle_snr

        frame_result = {
            "idx": idx,
            "variants": {},
            "reference_gcnr": ref_gcnr,
            "reference_speckle_snr": frame_result_ref_snr,
            "n_lesion_px": int(lesion.sum()),
            "n_background_px": int(background.sum()),
        }
        print(
            f"  reference bmode gCNR={ref_gcnr:.4f} speckleSNR={ref_speckle_snr:.3f} "
            f"(lesion {lesion.sum()} px, background {background.sum()} px)"
        )

        # Not every frame contains an inclusion — some are plain speckle
        # phantoms with an empty segmentation map. Those still validate the
        # speckle statistics, they just cannot say anything about contrast.
        # bool(...) around the whole expression: `x and np.isfinite(y)` yields a
        # numpy bool_, which json cannot serialise.
        have_lesion = bool(lesion.any() and np.isfinite(ref_gcnr))
        frame_result["has_inclusion"] = have_lesion
        if not have_lesion:
            print("  (no inclusion in this frame's segmentation map: gCNR checks n/a)")

        best_variant, best_key = None, -np.inf
        for name, f_number in F_NUMBER_VARIANTS.items():
            tv = time.time()
            out = beamform_variant(params, raw, k=1, f_number=f_number, tof_f_number=f_number)
            s = score(out[..., 0], ref_db, lesion, background, speckle)
            s["wall_s"] = round(time.time() - tv, 1)
            frame_result["variants"][name] = s
            print(
                f"  [{name}] gCNR={s['gcnr']:.4f} speckleSNR={s['speckle_snr']:.3f} "
                f"structNCC={s['structural_ncc']:.3f} "
                f"(pixelwise NCC={s['ncc_pixelwise']:.3f}, not gated) ({s['wall_s']}s)"
            )
            # Rank by contrast where there is contrast to rank by; otherwise by
            # how closely the speckle statistics track the reference.
            key = s["gcnr"] if have_lesion else -abs(s["speckle_snr"] - ref_speckle_snr)
            if np.isfinite(key) and key > best_key:
                best_key, best_variant = key, name

        # Negative control: same winning aperture, phase rotation disabled.
        f_number = F_NUMBER_VARIANTS[best_variant]
        neg_out = beamform_variant(
            params, raw, k=1, f_number=f_number, tof_f_number=f_number, demodulation_frequency=0.0
        )
        neg_db = log_envelope_db(neg_out[..., 0])
        neg_score = score(neg_out[..., 0], ref_db, lesion, background, speckle)
        frame_result["negative_control_no_phase_rotation"] = neg_score
        frame_result["best_variant"] = best_variant

        best = frame_result["variants"][best_variant]
        snr_lo, snr_hi = SPECKLE_SNR_RANGE
        checks = {
            "speckle_snr_rayleigh": snr_lo <= best["speckle_snr"] <= snr_hi,
            "speckle_snr_tracks_reference": abs(best["speckle_snr"] - ref_speckle_snr)
            <= SPECKLE_SNR_VS_REF_TOLERANCE,
        }
        if have_lesion:
            checks["gcnr_tracks_reference"] = (
                abs(best["gcnr"] - ref_gcnr) <= GCNR_VS_REF_TOLERANCE
            )
        # Only meaningful where the reference itself resolves the inclusion.
        control_applicable = have_lesion and ref_gcnr >= NEGATIVE_CONTROL_MIN_REF_GCNR
        if control_applicable:
            checks["negative_control_degrades"] = neg_score["gcnr"] < best["gcnr"]
        frame_result["checks"] = {k: bool(v) for k, v in checks.items()}
        frame_result["negative_control_applicable"] = bool(control_applicable)
        frame_result["gate_pass"] = bool(all(checks.values()))
        for cname, ok in checks.items():
            print(f"    {'PASS' if ok else 'FAIL'}  {cname}")
        if not control_applicable:
            why = (
                "no inclusion in this frame"
                if not have_lesion
                else f"reference gCNR {ref_gcnr:.3f} < {NEGATIVE_CONTROL_MIN_REF_GCNR}"
            )
            print(f"    n/a   negative_control_degrades ({why}: no contrast to destroy)")
        if have_lesion:
            print(
                f"  best={best_variant} gCNR={best['gcnr']:.4f} (ref {ref_gcnr:.4f}) "
                f"gate_pass={frame_result['gate_pass']} | negative control gCNR="
                f"{neg_score['gcnr']:.4f} (must be below {best['gcnr']:.4f}, "
                "proving phase rotation matters)"
            )
        else:
            print(
                f"  best={best_variant} speckleSNR={best['speckle_snr']:.3f} "
                f"(ref {ref_speckle_snr:.3f}) gate_pass={frame_result['gate_pass']} "
                "| speckle-only frame, no contrast checks"
            )
        results.append(frame_result)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        best_db = log_envelope_db(
            beamform_variant(
                params, raw, k=1, f_number=f_number, tof_f_number=f_number
            )[..., 0]
        )
        # Lesion/background regions are outlined on both images: the gate is now
        # a regional contrast measure, so the regions it used must be visible.
        def _g(value: float) -> str:
            return f"gCNR={value:.3f}" if np.isfinite(value) else "no inclusion"

        for ax, img, title in (
            (axes[0], ref_db, f"NVIDIA bmode (resampled {GRID}x{GRID})\n{_g(ref_gcnr)}"),
            (axes[1], best_db, f"ours, {best_variant}\n{_g(best['gcnr'])}"),
            (axes[2], best_db - ref_db, "difference (dB)\n(speckle realisation, not gated)"),
            (
                axes[3],
                neg_db,
                f"negative control (no phase rotation)\n{_g(neg_score['gcnr'])}",
            ),
        ):
            im = ax.imshow(img, cmap="gray" if ax is not axes[2] else "coolwarm",
                            vmin=-DYNAMIC_RANGE_DB if ax is not axes[2] else -20,
                            vmax=0 if ax is not axes[2] else 20)
            if ax is not axes[2] and have_lesion:
                ax.contour(lesion, levels=[0.5], colors="lime", linewidths=0.8)
                ax.contour(background, levels=[0.5], colors="deepskyblue", linewidths=0.4)
            ax.set_title(title, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(
            f"WI-4 validation — frame {idx} — gate_pass={frame_result['gate_pass']} "
            "(lime: inclusion, blue: depth-matched background)"
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"frame{idx}_validation.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    out_path = Path("results/gates/nvidia_reconstruction.json")
    # numpy scalars leak into these dicts from the metric functions; json has no
    # encoder for them, so coerce on the way out rather than at every call site.
    def _plain(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")

    out_path.write_text(json.dumps(results, indent=1, default=_plain))
    print(f"\nwrote {out_path} and outputs/gates/nvidia_reconstruction/frame*_validation.png")
    return {"results": results}


if __name__ == "__main__":
    main()
