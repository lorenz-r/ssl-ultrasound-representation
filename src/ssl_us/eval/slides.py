"""Figure styling for the presentation plots: one palette, one set of rcParams.

The figures are produced **bare** — no title, subtitle or caption baked in —
because they are assembled into slides by hand and the deck supplies that text.
`outputs/presentation/README.md` carries what each plot shows and where its
numbers come from, so the provenance lives beside the file rather than burned
into the pixels where it cannot be edited.

Every presentation figure goes through here so a deck does not end up with
eight different greys and four different blues. The palette is the validated
reference instance of the data-viz design system — the hexes and their *order*
are load-bearing (the ordering is what makes adjacent categories separable
under protanopia/deuteranopia), so treat them as data, not as taste.

Measured properties of the categorical order below, on the adjacent pairlist
that bar/line charts use: worst CVD ΔE 9.1 light / 8.4 dark against a target of
8, worst normal-vision ΔE 19.6 light / 19.3 dark against a floor of 15. Under
an *all-pairs* pairlist — scatter, bubble, small multiples — only the **first
three slots** clear the floors, so those forms carry a three-series cap. Past
three, facet instead of adding a fourth hue.

Three rules this module exists to enforce:

1. **Nominal categories never get a value ramp.** One series is one colour
   (slot 1); N series take slots 1..N in order. Colouring bars by their own
   height spends the identity channel re-encoding what length already shows.
2. **Magnitude is sequential — one hue, light→dark.** :func:`sequential_cmap`.
   Never a rainbow. R² ≤ 0 means "no information recovered", not "opposite", so
   it floors at 0 rather than reaching for a diverging ramp.
3. **Markers must stay visible.** A sequential ramp for a *heatmap* may run to
   its palest step (near-zero is allowed to recede into the surface), but a
   scatter marker at that step disappears. `for_markers=True` clips the ramp to
   the step that still clears 2:1 against the surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in validated order. Do not re-order or extend past 8.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}
# Scatter/small-multiple forms clear the all-pairs floors only this far.
ALL_PAIRS_CAP = 3

# Sequential blue ramp, step 100 (palest) -> 700 (darkest).
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
# Index into BLUE_RAMP of the palest step that still clears 2:1 on each
# surface — the floor for anything drawn as a discrete mark.
MARKER_FLOOR = {"light": 3, "dark": 10}   # light: #86b6ef 2.06:1; dark: #184f95 2.15:1

INK = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "primary": "#0b0b0b",
              "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "primary": "#ffffff",
             "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835"},
}

THEMES = ("light", "dark")
FONT_STACK = ["DejaVu Sans", "Bitstream Vera Sans", "sans-serif"]


def tokens(theme: str) -> dict:
    """Ink/surface tokens for `theme`. Text uses these, never a series colour."""
    return INK[theme]


def series(theme: str, n: int, *, all_pairs: bool = False) -> list[str]:
    """`n` categorical colours in validated order.

    `all_pairs=True` for forms where every pair is seen at once (scatter, small
    multiples); it enforces the three-series cap rather than silently handing
    back a fourth hue that fails the separation floors.
    """
    cap = ALL_PAIRS_CAP if all_pairs else len(CATEGORICAL[theme])
    if n > cap:
        raise ValueError(
            f"{n} series exceeds the {'all-pairs' if all_pairs else 'adjacent'} "
            f"cap of {cap}. Facet into small multiples or fold the tail into "
            f"'Other' — do not cycle or generate hues."
        )
    return CATEGORICAL[theme][:n]


def sequential_cmap(theme: str, *, for_markers: bool = False) -> LinearSegmentedColormap:
    """One-hue light→dark ramp for magnitude.

    On the dark surface the ramp is reversed so "more" still reads as "further
    from the background" rather than as "closer to black".
    """
    steps = BLUE_RAMP[MARKER_FLOOR[theme]:] if for_markers and theme == "light" else BLUE_RAMP
    if for_markers and theme == "dark":
        steps = BLUE_RAMP[: MARKER_FLOOR["dark"] + 1]
    if theme == "dark":
        steps = list(reversed(steps))
    return LinearSegmentedColormap.from_list(f"ssl_blue_{theme}", steps)


def _relative_luminance(rgb) -> float:
    """WCAG relative luminance from an RGB triple in 0..1."""
    chan = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in rgb[:3]]
    return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _contrast(a, b) -> float:
    la, lb = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def ink_on(cell_rgb, theme: str) -> str:
    """The theme's text token that contrasts *most* against `cell_rgb`.

    Measured against the cell and maximised, rather than branched on the theme
    or thresholded on luminance. Two reasons, both found by measurement:

    * A theme-conditional rule is wrong in one direction and silently so.
      `surface` is the light token on a light theme and the dark one on a dark
      theme, so "use surface on dark cells" reads 13.7:1 in light mode and
      **1.4:1 — invisible — in dark**.
    * A fixed luminance threshold picks the wrong token near the ramp's middle,
      where neither ink is obviously better; it bottomed out at 2.81:1. Taking
      the max of the two candidates raises the worst case over the whole ramp
      to 4.6:1 and cannot be beaten by any other choice between them.

    Both candidates are text tokens, so this never reaches for a series colour.
    """
    t = INK[theme]
    candidates = (t["primary"], t["surface"])
    return max(candidates, key=lambda ink: _contrast(_hex_rgb(ink), cell_rgb))


def use(theme: str) -> dict:
    """Install rcParams for `theme` and return its ink tokens.

    Chrome is deliberately recessive: hairline solid grid one shade off the
    surface, no top/right spines, no dashes anywhere (a dashed rule reads as
    "threshold" when it is only a grid).
    """
    t = INK[theme]
    mpl.rcParams.update({
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "text.color": t["primary"],
        "axes.labelcolor": t["secondary"],
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.28,
    })
    return t


def strip_chrome(ax, *, x: bool = True, y: bool = True, grid_axis: str = "y") -> None:
    """Drop the top/right spines and keep the grid on one axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not x:
        ax.spines["bottom"].set_visible(False)
    if not y:
        ax.spines["left"].set_visible(False)
    ax.grid(True, axis=grid_axis)
    if grid_axis != "both":
        ax.grid(False, axis="x" if grid_axis == "y" else "y")


def tight(fig, **kw) -> None:
    """Margins for a **bare** figure — no title block, no caption.

    These plots are dropped into slides where the deck supplies the title and
    the commentary, so the figure itself carries only what is needed to read
    the data: axes, legend, direct labels, and any annotation that states a
    number. Anything the author would rewrite anyway is left out.
    """
    fig.subplots_adjust(**{"top": 0.965, "bottom": 0.11, "left": 0.07,
                           "right": 0.985, **kw})


def save(fig, out_dir: Path | str, name: str, theme: str) -> Path:
    """Write `<out_dir>/<name>[.dark].png` and close the figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme == "light" else ".dark"
    path = out_dir / f"{name}{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
