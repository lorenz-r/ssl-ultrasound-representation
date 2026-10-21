"""Reader for the Stanford FSA transfer corpus (WI-16).

Built by `scripts/data/fetch_stanford_fsa.py` from figshare 28291985. One array
of baseband IQ in NVIDIA's own `(n_tx, n_rx, n_ax)` layout, plus per-frame
provenance — so the beamformer that turns NVIDIA rows into the K-sector store
sees the same shape here, with different constants.

What differs from NVIDIA, and matters when beamforming:

| quantity | NVIDIA | Stanford FSA |
|---|---|---|
| elements | 180, pitch 201.0 um | 192, pitch 200.0 um |
| carrier | 6.5 MHz | 7.813 MHz |
| wavelength | 236.9 um | 197.1 um |
| fast-time rate | 13.33 MHz | 10.42 MHz (31.25 decimated by 3) |
| depth covered | 0-40 mm | 0.39-21.2 mm (publisher truncation) |
| `t0` | 3 discrete levels | one value, and it is **negative** |

That last row is the trap. Stanford's time origin carries the transducer's
lens correction and pulse-peak offset (`(TW.peak + 2*lensCorrection)*lambda/c`,
1.77 us here), so sample 0 sits *before* the transmit event. The NVIDIA path
forces `t_peak = 0` (PLAN.md WI-4); doing the same here would misregister every
pixel by about 1.4 mm. `t0_s` in the store attributes is already the corrected
value: sample *n* is at two-way time `t0_s + n / fs_hz`.

The dataset ships no animal identifiers — figshare flattened the per-rat
directory tree that its own loader scripts walk — so `session` is **inferred**
from acquisition timestamps and is the right grouping unit for a leakage-safe
split, with `source_file` as the finer one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr

DEFAULT_PATH = Path("/mnt/home/benbarkow/stores/wi16_stanford_fsa.zarr")

# Per-frame metadata keys that make sense as probe axes / split units.
LABEL_KEYS = ("session", "source_file", "kind", "date")


@dataclass
class StanfordFsaStore:
    """Baseband IQ for one FSA acquisition per frame, memory-mapped by zarr."""

    path: Path
    root: zarr.Group
    attrs: dict
    frame_metadata: dict[str, dict] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.root["iq"].shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.root["iq"].shape)

    def iq(self, index: int) -> np.ndarray:
        """`(n_tx, n_rx, n_ax)` complex64 — transmit *i* fired element *i*."""
        return np.asarray(self.root["iq"][index])

    def coverage(self, index: int) -> np.ndarray:
        """`(n_tx, n_rx)` — how many HVMux receive halves saw each element.

        Never zero for these acquisitions, but worth carrying: an element read
        by two halves was averaged, and one read by three (the `FSA_Fast`
        sequence uses three receive apertures) has a different noise level from
        its neighbours.
        """
        return np.asarray(self.root["coverage"][index])

    def time_axis_s(self) -> np.ndarray:
        """Two-way time of each fast-time sample, seconds."""
        n_ax = self.shape[-1]
        return self.attrs["t0_s"] + np.arange(n_ax) / self.attrs["fs_hz"]

    def depth_axis_m(self) -> np.ndarray:
        """On-axis depth of each fast-time sample. Negative early samples are
        real — see the module docstring on `t0_s`."""
        return self.time_axis_s() * self.attrs["c_m_s"] / 2.0

    def element_x_m(self) -> np.ndarray:
        return np.asarray(self.attrs["elem_x_m"], dtype=np.float64)

    def labels(self) -> dict[str, np.ndarray]:
        """Per-frame provenance as arrays aligned to frame index."""
        n = len(self)
        out: dict[str, np.ndarray] = {}
        for key in LABEL_KEYS:
            out[key] = np.array([self.frame_metadata[str(i)][key] for i in range(n)])
        return out

    def sessions(self) -> dict[str, list[int]]:
        """Inferred acquisition session -> frame indices. The split unit."""
        groups: dict[str, list[int]] = {}
        for i in range(len(self)):
            groups.setdefault(self.frame_metadata[str(i)]["session"], []).append(i)
        return groups


def open_fsa_store(path: str | Path = DEFAULT_PATH) -> StanfordFsaStore:
    """Open a built corpus, refusing a partial one.

    zarr fills unwritten chunks with zeros, so frames from an interrupted build
    read back as an all-zero acquisition — which beamforms to a black image
    rather than to an error. The manifest's per-file record is the authority on
    how many frames are real.
    """
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path} — run scripts/data/fetch_stanford_fsa.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    root = zarr.open_group(str(path), mode="r")
    written = sum(f["n_frames"] for f in manifest["files"].values())
    declared = int(root["iq"].shape[0])
    if written != declared:
        raise ValueError(
            f"{path} holds {written} of {declared} frames — the rest read back as zeros, "
            f"which beamforms to a black image rather than to an error. Finish the build."
        )
    frame_metadata = json.loads((path / "frame_metadata.json").read_text())
    if len(frame_metadata) != declared:
        raise ValueError(
            f"{path}: {len(frame_metadata)} metadata entries for {declared} frames"
        )
    return StanfordFsaStore(
        path=path, root=root, attrs=dict(root.attrs), frame_metadata=frame_metadata
    )
