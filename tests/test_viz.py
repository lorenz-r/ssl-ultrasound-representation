"""WI-13's encoder-comparison grid (`viz.plot_encoder_grid`).

The grid exists to be read *along a row* — same label, different encoder — so
the one thing that would quietly invalidate it is a per-panel colour scale: a
panel given a tighter range than its neighbour looks more structured than it is,
and nothing about the resulting figure says so. That is what these assert,
along with the probe score reaching the panel it annotates.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ssl_us.eval import viz  # noqa: E402


@pytest.fixture
def grid_inputs():
    rng = np.random.default_rng(0)
    n = 40
    xys = {name: rng.normal(size=(n, 2)) for name in ("vicreg_A", "vicreg_B")}
    labels = {
        "z_centre": rng.uniform(5.0, 35.0, n),
        "t0_group": rng.integers(0, 3, n),
    }
    return xys, labels


def test_row_shares_one_colour_scale(grid_inputs, tmp_path, monkeypatch):
    """Every panel in a row must map value -> colour identically.

    Asserted on the `vmin`/`vmax` the function actually passes to `scatter`,
    not on a reconstruction of it: the failure mode is letting matplotlib infer
    the range per panel, and only the real call site can show that.
    """
    xys, labels = grid_inputs
    limits = []
    real = viz.plt.Axes.scatter

    def spy(self, *a, **kw):
        limits.append((kw.get("vmin"), kw.get("vmax")))
        return real(self, *a, **kw)

    monkeypatch.setattr(viz.plt.Axes, "scatter", spy)
    out = viz.plot_encoder_grid(
        xys, labels, ["z_centre"], "t", tmp_path / "grid.png"
    )
    assert out.exists() and out.stat().st_size > 0

    z = labels["z_centre"]
    assert len(limits) == len(xys)  # one continuous scatter per encoder
    assert set(limits) == {(float(z.min()), float(z.max()))}


def test_panel_per_encoder_and_key(grid_inputs, tmp_path, monkeypatch):
    """One panel per (encoder, colour key) — a dropped cell is a silent hole."""
    xys, labels = grid_inputs
    titles = []
    real = viz.plt.Axes.set_title
    monkeypatch.setattr(
        viz.plt.Axes, "set_title",
        lambda self, label, *a, **kw: (titles.append(label), real(self, label, *a, **kw))[1],
    )
    viz.plot_encoder_grid(
        xys, labels, ["z_centre", "t0_group"], "t", tmp_path / "grid.png"
    )
    assert len(titles) == len(xys) * 2
    for name in xys:
        for key in ("z_centre", "t0_group"):
            assert any(t.startswith(name) and key in t for t in titles)


def test_cell_notes_reach_the_panel_title(grid_inputs, tmp_path, monkeypatch):
    """The probe R² is the comparable quantity; it has to survive onto the panel."""
    xys, labels = grid_inputs
    seen = []
    real = viz.plt.Axes.set_title

    def spy(self, label, *a, **kw):
        seen.append(label)
        return real(self, label, *a, **kw)

    monkeypatch.setattr(viz.plt.Axes, "set_title", spy)
    viz.plot_encoder_grid(
        xys, labels, ["z_centre"], "t", tmp_path / "grid.png",
        cell_notes={"vicreg_A": {"z_centre": "R²=+0.724"}},
    )
    assert any("R²=+0.724" in s for s in seen)
    assert any(s.startswith("vicreg_B") and "R²" not in s for s in seen)


def test_categorical_labels_do_not_crash(grid_inputs, tmp_path):
    """`t0_group` is integer-categorical — the scatter branch differs."""
    xys, labels = grid_inputs
    out = viz.plot_encoder_grid(
        xys, labels, ["t0_group"], "t", tmp_path / "grid.png"
    )
    assert out.exists() and out.stat().st_size > 0
