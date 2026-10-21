"""Stanford reference-validation gate: beamform real Stanford FSA frames with
`AngularSectorSum` (K=1) and validate the reconstruction — the Stanford
analogue of `scripts/beamform/validate_nvidia_reconstruction.py`.

**There is no external reference image to validate against.** NVIDIA ships
`bmode` alongside its raw IQ; Stanford's published FSA `.mat` files carry only
raw channel data (`stanford_mat.py`'s reader, `fetch_stanford_fsa.py`'s
corpus builder — neither touches or finds an `Image`/`bmode` field). So the
gCNR-vs-reference and structural-NCC-vs-reference checks WI-4's gate uses have
nothing to compare against here, and the segmentation-derived lesion/
background masks that gCNR itself needs do not exist (Stanford is unlabelled
in-vivo tissue, not a phantom with a ground-truth inclusion mask).

**What this gate checks instead, all self-contained:**

1. **Absolute Rayleigh speckle SNR**, same band as WI-4's
   (`SPECKLE_SNR_RANGE`), on a background region — a legitimate check on real
   tissue too, since the underlying physics (fully-developed speckle away
   from strong scatterers/lesions) does not depend on which array or which
   phantom produced it. It is the sharpest available check that the
   reconstruction is not smoothed, aliased, or otherwise broken, and it needs
   no reference.
2. **The phase-rotation negative control** (PLAN.md 3.5 trap #1, same as
   WI-4): disabling baseband phase rotation must measurably degrade the
   reconstruction. Graded by **structural agreement**, not by speckle SNR.

   ⚠ The speckle-SNR form of this check was written first and is **wrong on
   this data, in the inverting direction** — it was withdrawn after measuring
   it. Disabling phase rotation turns the aperture sum into a circular complex
   Gaussian, and the envelope of a circular complex Gaussian is *exactly*
   Rayleigh, SNR 1.913. So the broken reconstruction lands closer to the
   Rayleigh ideal than the correct one, which is further from it precisely
   because real tissue has resolved structure and is not fully-developed
   speckle everywhere. Measured over 12 frames: correct 1.702/1.779 (mm /
   lambda), broken 1.846/1.880, and the old check "passed" on 1 frame of 12 —
   it was reporting that the beamformer was broken because it worked.

   What phase rotation does buy is *structure*, so that is what is graded:
   `structural_ncc` between the correct reconstruction and the rotation-
   disabled one must fall measurably below the frame's own cross-variant
   agreement (check 3). Both numbers use the same metric on the same frame, so
   the comparison is between two reconstructions of one scene rather than
   against an absolute constant. The dB contrast drop is recorded alongside as
   the corroborating physical signature (measured 0.73-1.46 dB, always in the
   same direction), but is not itself gated.
3. **Cross-grid-variant structural agreement** — specific to WI-16, and the
   check that actually stands in for WI-4's reference comparison. PLAN.md 3.6
   runs both the mm-matched and lambda-matched grids as an ablation; both
   reconstruct the *same physical scene* from the *same raw channel data*, so
   after resampling one onto the other's grid, `structural_ncc` between them
   should be high. Two independently-gridded reconstructions of one frame
   agreeing on macroscopic structure is direct evidence the beamformer (not
   just the grid arithmetic) is correct — the same logic WI-4's module
   docstring used to justify checking two independent *beamformers* against
   each other instead of against a disagreeing reference.

Usage (a `12g`/`24g` instance or, slowly, this CPU `data` one):

    source /opt/venvs/.venv-zea/bin/activate
    KERAS_BACKEND=jax N_FRAMES=4 python scripts/beamform/validate_stanford_reconstruction.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ssl_us.beamform import image_quality as iqm
from ssl_us.beamform import stanford_frame as sf
from ssl_us.data import stanford_geometry as sgeo
from ssl_us.data.stanford_fsa import open_fsa_store

N_FRAMES = int(os.environ.get("N_FRAMES", "1"))
F_NUMBER_VARIANTS = {"unlimited": 1e-6, "f1.5": 1.5, "f1.0": 1.0}
DYNAMIC_RANGE_DB = 40.0

# Same band WI-4 gates on (`validate_nvidia_reconstruction.py`) — the Rayleigh
# constant is a property of fully-developed speckle, not of this corpus.
SPECKLE_SNR_RANGE = (1.60, 2.15)
# Background region for speckle statistics: mid-depth, away from both the
# near-field mute and the shallow/noisy edges of the [0, 19.9] mm grid.
# Unlike WI-4's SPECKLE_BAND_MM there is no lesion to excise (no segmentation
# ground truth exists for Stanford).
SPECKLE_BAND_MM = (5.0, 15.0)
# Cross-variant structural agreement floor. **Measured, then frozen** (the same
# discipline `validate_nvidia_reconstruction.py` documents for its own gates): over
# 12 frames on `12g`, mm-matched vs lambda-matched agreed at NCC 0.9997-0.9998.
# The floor sits well below that and far above the 0.955 a phase-broken
# reconstruction reaches, so it separates the two without grading frame
# difficulty.
CROSS_VARIANT_NCC_MIN = 0.99
# Minimum NCC *drop*, cross-variant agreement minus correct-vs-phase-broken
# agreement, for the negative control to count as having degraded anything.
# Reference-relative rather than absolute, for WI-4's reason: it grades the
# reconstruction against its own reproducibility scale on that same frame.
# Measured drop 0.045-0.117 over 8 frame/variant pairs; frozen at 0.02.
NEGATIVE_CONTROL_MIN_NCC_DROP = 0.02


def envelope(img: np.ndarray) -> np.ndarray:
    """``(Z, X)`` envelope of one ``(2, Z, X)`` I/Q image.

    The I/Q axis is **first**, matching what `stanford_frame.beamform_frame`
    returns (the store's own `(2, Z, X, K)` layout with K already selected).
    WI-4's own helper takes `(Z, X, 2)` because it works on zea's pre-transpose
    output; copying that indexing here reduced over the wrong axis.
    """
    return np.sqrt(img[0] ** 2 + img[1] ** 2)


def log_envelope_db(img: np.ndarray) -> np.ndarray:
    db = 20 * np.log10(np.maximum(envelope(img), 1e-9))
    return db - db.max()


def speckle_mask(z_mm: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    lo, hi = SPECKLE_BAND_MM
    rows = (z_mm >= lo) & (z_mm <= hi)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, :] = True
    return mask


def _bilinear_resample(img: np.ndarray, z_src, x_src, z_out, x_out) -> np.ndarray:
    """Resample a (nz_src, nx_src) image onto a new (nz_out, nx_out) grid."""
    zi = np.clip(np.interp(z_out, z_src, np.arange(len(z_src))), 0, len(z_src) - 1)
    xi = np.clip(np.interp(x_out, x_src, np.arange(len(x_src))), 0, len(x_src) - 1)
    z0 = np.floor(zi).astype(int)
    z1 = np.minimum(z0 + 1, len(z_src) - 1)
    x0 = np.floor(xi).astype(int)
    x1 = np.minimum(x0 + 1, len(x_src) - 1)
    wz = (zi - z0)[:, None]
    wx = (xi - x0)[None, :]
    top = img[np.ix_(z0, x0)] * (1 - wx) + img[np.ix_(z0, x1)] * wx
    bot = img[np.ix_(z1, x0)] * (1 - wx) + img[np.ix_(z1, x1)] * wx
    return top * (1 - wz) + bot * wz


def score_speckle(env: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "speckle_snr": iqm.speckle_snr_patched(env, mask, patch=32),
        "speckle_snr_pooled": iqm.speckle_snr(env[mask]),
    }


def main() -> dict:
    store = open_fsa_store()
    grids = {name: sgeo.build_grid_variant(name, store.attrs) for name in sgeo.GRID_VARIANTS}
    fig_dir = Path("outputs/gates/stanford_reconstruction")
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for idx in range(min(N_FRAMES, len(store))):
        frame_result: dict = {"idx": idx, "variants": {}}
        print(f"[{idx}] {store.frame_metadata[str(idx)]}", flush=True)

        db_by_variant: dict[str, np.ndarray] = {}
        for name, grid in grids.items():
            mask = speckle_mask(grid.z_mm, (len(grid.z_mm), len(grid.x_mm)))
            best_variant, best_key, best_out = None, -np.inf, None
            for fname, f_number in F_NUMBER_VARIANTS.items():
                tv = time.time()
                out = sf.beamform_frame(store, idx, grid, k=1, f_number=f_number)[:, :, :, 0]
                env = envelope(out)
                s = score_speckle(env, mask)
                s["wall_s"] = round(time.time() - tv, 1)
                frame_result["variants"].setdefault(name, {})[fname] = s
                dist = abs(s["speckle_snr"] - iqm.RAYLEIGH_SNR)
                print(
                    f"  [{name}/{fname}] speckleSNR={s['speckle_snr']:.3f} "
                    f"(Rayleigh {iqm.RAYLEIGH_SNR:.3f}) ({s['wall_s']}s)",
                    flush=True,
                )
                if np.isfinite(dist) and -dist > best_key:
                    best_key, best_variant, best_out = -dist, fname, out
            f_number = F_NUMBER_VARIANTS[best_variant]

            # Negative control: same winning aperture, phase rotation disabled.
            neg_out = sf.beamform_frame(
                store, idx, grid, k=1, f_number=f_number, demodulation_frequency=0.0
            )[:, :, :, 0]
            neg_score = score_speckle(envelope(neg_out), mask)
            best_score = frame_result["variants"][name][best_variant]
            best_db, neg_db = log_envelope_db(best_out), log_envelope_db(neg_out)
            # Graded below, once the frame's own cross-variant agreement is
            # known — the two NCCs are only meaningful against each other.
            neg_ncc = iqm.structural_ncc(best_db, neg_db, window=15)

            frame_result[name] = {
                "best_f_number": best_variant,
                "negative_control": {
                    **neg_score,
                    "structural_ncc_vs_best": float(neg_ncc),
                    # Corroborating physical signature, recorded not gated:
                    # coherent summation buys log-envelope contrast.
                    "db_contrast_drop": float(
                        best_db[mask].std() - neg_db[mask].std()
                    ),
                },
                "checks": {
                    "speckle_snr_rayleigh": bool(
                        SPECKLE_SNR_RANGE[0]
                        <= best_score["speckle_snr"]
                        <= SPECKLE_SNR_RANGE[1]
                    ),
                },
            }
            db_by_variant[name] = best_db

        # Cross-variant structural agreement (module docstring, check 3):
        # resample lambda_matched onto mm_matched's grid and compare.
        mm_grid, lam_grid = grids["mm_matched"], grids["lambda_matched"]
        lam_on_mm = _bilinear_resample(
            db_by_variant["lambda_matched"],
            lam_grid.z_mm,
            lam_grid.x_mm,
            mm_grid.z_mm,
            mm_grid.x_mm,
        )
        cross_ncc = iqm.structural_ncc(db_by_variant["mm_matched"], lam_on_mm, window=15)
        frame_result["cross_variant_structural_ncc"] = cross_ncc
        cross_pass = np.isfinite(cross_ncc) and cross_ncc >= CROSS_VARIANT_NCC_MIN
        frame_result["checks"] = {"cross_variant_structural_ncc": bool(cross_pass)}
        print(f"  cross-variant structural NCC = {cross_ncc:.4f}", flush=True)

        # The negative control, graded now that the frame's own reproducibility
        # scale is known: changing the *grid* must perturb structure far less
        # than removing the *phase rotation* does.
        for name in grids:
            neg = frame_result[name]["negative_control"]
            drop = cross_ncc - neg["structural_ncc_vs_best"]
            neg["structural_ncc_drop_vs_cross_variant"] = float(drop)
            frame_result[name]["checks"]["negative_control_degrades"] = bool(
                np.isfinite(drop) and drop >= NEGATIVE_CONTROL_MIN_NCC_DROP
            )
            print(
                f"  [{name}] negative control: NCC {neg['structural_ncc_vs_best']:.4f} "
                f"(drop {drop:.4f} vs cross-variant), "
                f"dB contrast {neg['db_contrast_drop']:+.2f}",
                flush=True,
            )

        all_checks = list(frame_result["checks"].values())
        for name in grids:
            for cname, ok in frame_result[name]["checks"].items():
                print(f"    {'PASS' if ok else 'FAIL'}  {name}/{cname}", flush=True)
            all_checks += list(frame_result[name]["checks"].values())
        frame_result["gate_pass"] = bool(all(all_checks))
        print(f"  gate_pass={frame_result['gate_pass']}", flush=True)
        results.append(frame_result)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, img, title in (
            (axes[0], db_by_variant["mm_matched"], "mm-matched"),
            (axes[1], db_by_variant["lambda_matched"], "lambda-matched"),
            (axes[2], db_by_variant["mm_matched"] - lam_on_mm, f"diff (dB)\nNCC={cross_ncc:.3f}"),
        ):
            im = ax.imshow(
                img,
                cmap="gray" if ax is not axes[2] else "coolwarm",
                vmin=-DYNAMIC_RANGE_DB if ax is not axes[2] else -20,
                vmax=0 if ax is not axes[2] else 20,
            )
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"WI-16 validation — frame {idx} — gate_pass={frame_result['gate_pass']}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"frame{idx}_validation.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    out_path = Path("results/gates/stanford_reconstruction.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _plain(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")

    out_path.write_text(json.dumps(results, indent=1, default=_plain))
    print(f"\nwrote {out_path} and outputs/gates/stanford_reconstruction/frame*_validation.png")
    return {"results": results}


if __name__ == "__main__":
    main()
