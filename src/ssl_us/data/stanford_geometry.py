"""Shared-grid geometry for the Stanford FSA transfer corpus (WI-16, PLAN.md §3.6).

Two output-pixel grid variants, both built over the *same* physical Stanford
array (element positions are physical, read from the store's own attrs, not
grid-dependent) — only the z/x **pixel spacing** changes between them:

- **mm-matched**: the same pixel spacing as NVIDIA's own B-mode grid
  (:func:`ssl_us.data.geometry.bmode_dz_mm`, ~79.05 um) — anatomy scale
  matches between corpora, speckle grain does not (Stanford's wavelength is
  smaller: 197.1 um vs 236.9 um).
- **lambda-matched**: ``wavelength_m / 3`` at Stanford's own carrier — speckle
  statistics match, physical scale does not.

Both are cheap resample parameters (PLAN.md 3.6), run as an ablation rather
than a single choice.

**Depth range.** The store's own fast-time axis spans -0.97 to 20.18 mm (see
:mod:`ssl_us.data.stanford_fsa`'s module docstring for why sample 0 is
negative), but the *pixel grid* here only covers ``[0, DEPTH_MAX_MM]``:
z < 0 is in front of the array face, where a beamformed pixel is not a
meaningful quantity, and ``DEPTH_MAX_MM`` is the corpus's actual signal extent
(PLAN.md 3.6 rev-6: "real signal to ~19.9 mm" after the publishers' own
truncation). This is also the range the WI-16 headline transfer table uses at
K <= 4 (PLAN.md WI-16 "done when").

**X range.** The full physical array aperture, symmetric about 0 — the same
convention NVIDIA's own bmode grid uses for its 180-element array (PLAN.md
2.3: the bmode grid's x-extent is exactly ``(n_el - 1) * pitch / 2``).

Constants come from the built store's own attrs
(:func:`ssl_us.data.stanford_fsa.open_fsa_store`) rather than being hardcoded
here, unlike NVIDIA's :mod:`ssl_us.data.geometry` — that module hardcodes
because its constants come from a *streamed* sample and hardcoding avoids
requiring the 271 GB corpus just to import it. Stanford's store is a single
local 9 GB file with everything already in its attrs, so reading them at
import time costs nothing and can never drift from what actually built it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as geo

DEPTH_MIN_MM = 0.0
DEPTH_MAX_MM = 19.9

# NVIDIA's own bmode pixel spacing — the "mm-matched" target (PLAN.md 3.6 table).
MM_MATCHED_SPACING_MM = geo.bmode_dz_mm()

GRID_VARIANTS = ("mm_matched", "lambda_matched")


@dataclass(frozen=True)
class GridVariant:
    name: str
    z_mm: np.ndarray  # (nz,)
    x_mm: np.ndarray  # (nx,)
    spacing_mm: float  # target spacing; the realised per-axis spacing may differ
    # slightly (a whole number of pixels must span the fixed [min, max] range).


def _axis(min_mm: float, max_mm: float, spacing_mm: float) -> np.ndarray:
    """Evenly spaced axis covering ``[min_mm, max_mm]``, closest to `spacing_mm`.

    Mirrors how NVIDIA's own (507, 456) grid is defined (`geometry.py`): a
    fixed number of points spanning a fixed physical extent, not a spacing
    applied outward from an origin — so the endpoints are always exact.
    """
    n_steps = max(1, round((max_mm - min_mm) / spacing_mm))
    return np.linspace(min_mm, max_mm, n_steps + 1)


def aperture_half_width_mm(attrs: dict) -> float:
    """Half the physical array aperture, from the store's own element positions."""
    elem_x_m = np.asarray(attrs["elem_x_m"], dtype=np.float64)
    return float((elem_x_m.max() - elem_x_m.min()) / 2.0 * 1e3)


def build_grid_variant(name: str, attrs: dict) -> GridVariant:
    """The z/x pixel axes for one named variant, from a store's ``attrs``.

    Args:
        name: ``"mm_matched"`` or ``"lambda_matched"``.
        attrs: a :class:`~ssl_us.data.stanford_fsa.StanfordFsaStore`'s
            ``.attrs`` (or any dict with the same ``elem_x_m``/``wavelength_m``
            keys — tests pass a minimal one).
    """
    half_x_mm = aperture_half_width_mm(attrs)
    if name == "mm_matched":
        spacing_mm = MM_MATCHED_SPACING_MM
    elif name == "lambda_matched":
        spacing_mm = float(attrs["wavelength_m"]) * 1e3 / 3.0
    else:
        raise ValueError(f"unknown grid variant {name!r}, expected one of {GRID_VARIANTS}")
    z_mm = _axis(DEPTH_MIN_MM, DEPTH_MAX_MM, spacing_mm)
    x_mm = _axis(-half_x_mm, half_x_mm, spacing_mm)
    return GridVariant(name=name, z_mm=z_mm, x_mm=x_mm, spacing_mm=spacing_mm)
