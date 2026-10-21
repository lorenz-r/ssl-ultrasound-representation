"""``AngularSectorSum``: the beamformer operation that replaces ``DelayAndSum``
for angular receive sectors (PLAN.md §0.1, WI-4).

zea's own ``DelayAndSum`` (``zea/ops/pipeline.py``) is six lines: sum the
TOF-corrected ``(n_tx, n_pix, n_el, n_ch)`` tensor over elements, then over
transmits. This replaces exactly that step with a segment-sum over elements
into K angular bins *before* the transmit sum — everything upstream (delay
computation, interpolation, baseband phase rotation, the f-number receive
mask, pixel chunking) is untouched, all zea's, per PLAN.md §0.1.

Registered as both a ``zea.ops`` operation and a ``zea`` beamformer, so it
drops straight into zea's ``Beamform`` pipeline in place of ``"delay_and_sum"``::

    from zea.ops.pipeline import Beamform
    beamform = Beamform(beamformer="angular_sector_sum", k=16, f_number=1.0,
                         num_patches=100)

Sector membership is computed from ``flatgrid`` (this pixel chunk's
coordinates) and ``probe_geometry``, both of which zea's pipeline already
threads through to every operation — so, unlike the offline
:mod:`ssl_us.beamform.sectors` module (which precomputes a ``sector_id`` map
once for the whole grid), this operation derives sector membership on the fly
per chunk using the exact same view-angle geometry (PLAN.md §3.2), expressed
in ``keras.ops`` so it runs under whichever backend zea is configured with.
The two are cross-checked against each other in tests: chunk-derived sector
membership must agree with the precomputed map to the pixel.

Output is raw sums, never normalized here (PLAN.md §3.2 / WI-5): baking
occupancy normalization into the beamformer would break the K=16→8→4→2→1
ladder's exact pairwise-sum identity.
"""

from __future__ import annotations

from keras import ops
from zea.internal.core import DataTypes
from zea.internal.registry import beamformer_registry, ops_registry
from zea.ops.base import Operation

from .sectors import aperture_half_angle_rad


@beamformer_registry("angular_sector_sum")
@ops_registry("angular_sector_sum")
class AngularSectorSum(Operation):
    """Segment-sum over receive elements into K angular sectors, then compound
    over transmits.
    """

    def __init__(self, k: int = 16, f_number: float | None = 1.0, **kwargs):
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            **kwargs,
        )
        self.k = int(k)
        self.f_number = f_number
        self.theta_max = aperture_half_angle_rad(f_number)

    def call(self, **kwargs):
        """
        Args:
            data (Tensor): keyed by ``self.key`` (default ``"data"``) — the
                TOF-corrected input of shape ``(n_tx, n_pix, n_el, n_ch)``,
                one :class:`~zea.ops.PatchedGrid` chunk.
            flatgrid (Tensor): pixel coordinates ``[x, y, z]`` for this chunk,
                shape ``(n_pix, 3)`` — threaded through by the enclosing
                pipeline exactly as :class:`~zea.ops.TOFCorrection` receives it.
            probe_geometry (Tensor): element positions ``[x, y, z]``, shape
                ``(n_el, 3)``.

        Returns:
            dict: ``{self.output_key: sector_sums}`` of shape
            ``(n_pix, n_ch, k)`` — raw sums, not normalized by occupancy.
        """
        data = kwargs[self.key]
        flatgrid = kwargs["flatgrid"]
        probe_geometry = kwargs["probe_geometry"]

        x_p = flatgrid[:, 0]
        z_p = flatgrid[:, 2]
        x_e = probe_geometry[:, 0]

        dx = x_e[None, :] - x_p[:, None]  # (n_pix, n_el)
        theta = ops.arctan2(dx, z_p[:, None])  # (n_pix, n_el)
        valid = ops.abs(theta) <= self.theta_max
        frac = (theta + self.theta_max) / (2.0 * self.theta_max)
        sector = ops.clip(ops.floor(frac * self.k), 0, self.k - 1)
        # Elements outside the aperture are routed to bin 0 by `where` and then
        # zeroed by the `valid` mask below, so they contribute nothing to any bin.
        sector_idx = ops.cast(ops.where(valid, sector, ops.zeros_like(sector)), "int32")

        onehot = ops.one_hot(sector_idx, self.k)  # (n_pix, n_el, k)
        onehot = onehot * ops.cast(valid, onehot.dtype)[..., None]

        sector_sums = ops.einsum("tpec,peK->pcK", data, onehot)
        return {self.output_key: sector_sums}
