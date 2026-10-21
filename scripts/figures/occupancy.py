"""Sector-occupancy figure: occupancy vs depth, for f# in {1.5, 1.0, 0.75}
and K in {16, 8, 4, 1} (PLAN.md WI-3). Pure geometry — no dataset download.

    poetry run python scripts/figures/occupancy.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ssl_us.beamform import sectors as sec
from ssl_us.data import geometry as geo

F_NUMBERS = (1.5, 1.0, 0.75)
KS = (16, 8, 4, 1)
Z_MM = np.linspace(0.5, geo.BMODE_Z_MAX_MM, 120)


def mean_occupancy_vs_depth(f_number: float, k: int) -> np.ndarray:
    """Mean elements/sector at broadside (x=0), one value per z in Z_MM."""
    g = sec.build_sector_geometry(k=k, f_number=f_number, z_mm=Z_MM, x_mm=np.array([0.0]))
    return g.occupancy[:, 0, :].mean(axis=-1)


def main(out_path: str = "outputs/diagnostics/sector_occupancy_vs_depth.png") -> Path:
    fig, axes = plt.subplots(1, len(F_NUMBERS), figsize=(5 * len(F_NUMBERS), 4), sharey=True)
    for ax, f_number in zip(axes, F_NUMBERS, strict=True):
        for k in KS:
            ax.plot(Z_MM, mean_occupancy_vs_depth(f_number, k), label=f"K={k}")
        ax.axhline(2.0, color="grey", linestyle=":", linewidth=1, label="2 elements/sector")
        half_deg = np.degrees(sec.aperture_half_angle_rad(f_number))
        ax.set_title(f"f# = {f_number}  (±{half_deg:.1f}°)")
        ax.set_xlabel("depth z (mm)")
        ax.set_yscale("log")
    axes[0].set_ylabel("mean elements / sector (broadside)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Angular-sector occupancy vs depth (PLAN.md 3.2 / WI-3)")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
