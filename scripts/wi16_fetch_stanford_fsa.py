"""WI-16: build the Stanford FSA transfer corpus from figshare.

**PLAN.md §3.6/WI-16 assumed Stanford was focused multi-focal data**, needing
`zea.ops.Refocus` to be decoded into multistatic form before the NVIDIA
pipeline could touch it. That assumption came from the phase-1 repo, which
only ever handled the MultiFocal acquisitions. The published dataset
(10.25452/figshare.plus.28291985, *Pulse-Echo Ultrasound Murine In Vivo
Acquisitions*, Apache-2.0) is three acquisition types over the same animals:

    FSA         117 files   76.3 GB   <- one transmit per element
    Hadamard    117 files   50.9 GB      (encoded FSA, needs decoding)
    MultiFocal   81 files   37.1 GB      (what WI-16 was written against)
    MF_Fast      33 files   75.1 GB

`TX.Apod` on the FSA files has **exactly one non-zero entry per transmit**,
192 transmits for 192 elements — the same full-synthetic-aperture scheme as
NVIDIA's 180-transmit corpus. So `Refocus` is not needed at all: this is
already multistatic, and the identical `AngularSectorSum` path applies. That
is the single biggest simplification available to WI-16, and it is why this
script exists.

**What it writes**, per frame, matching NVIDIA's `(n_tx, n_rx, n_ax)` layout so
the beamformer sees one format:

    <out>/iq          (N, 192, 192, 287) complex64  — baseband IQ, tx x rx x fast-time
    <out>/coverage    (N, 192, 192)      uint8      — HVMux receive halves per element
    <out>/frame_metadata.json            per-frame provenance + session grouping
    <out>/manifest.json                  per-file completion, md5s, verification

**Three reductions take 76 GB of raw acquisition to ~10 GB of corpus**, none of
them lossy in a way that matters:

1. *One frame per file* by default. Each `.mat` holds a 10-frame cine of one
   view; consecutive frames are near-duplicates for probing purposes.
2. *Truncation to 861 fast-time samples.* *The publishers already truncated
   the data* — every file is identically zero past sample ~846 (21.2 mm), which
   is where the directory names' "_truncated" comes from. Storing 2560 samples
   would be storing 1700 zeros. The cutoff is measured per file and asserted,
   never assumed.
3. *Decimation by 3 after IQ demodulation.* The RF arrives at 31.25 MHz, four
   samples per wavelength. Demodulated to baseband it needs only the probe's
   own bandwidth, and `Trans.Bandwidth` is 4-12 MHz — 102 % fractional, i.e.
   +/-4 MHz about the 7.813 MHz carrier. Decimating by 3 leaves +/-5.2 MHz of
   headroom; by 4 (what the retired `benbarkow/us-ssl-mf` set used) leaves
   +/-3.9 MHz, which clips the band edges.

Usage (a `data` instance — download + CPU decode, no GPU):

    source /opt/venvs/.venv/bin/activate
    python scripts/wi16_fetch_stanford_fsa.py \
      --out /mnt/home/benbarkow/stores/wi16_stanford_fsa.zarr

Measured on that instance: 76 MB/s download, ~25 s/frame decode, ~1.5 h total.
Raw `.mat` files are deleted as they are consumed (`--keep-raw` to retain),
so peak scratch use is one file, not 76 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import zarr
from scipy.signal import resample_poly

from ssl_us.beamform import stanford_mat as sm

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article}/versions/{version}"
ARTICLE, VERSION = 28291985, 1

# Fast-time handling. Both numbers are asserted against the data, not trusted:
# `N_KEEP` must cover every non-zero sample, and `DECIM` must leave the probe's
# own bandwidth inside the decimated Nyquist limit.
N_KEEP = 861  # 3 * 287, covers the publishers' ~846-sample truncation
DECIM = 3

# A new acquisition session (one animal, one probe placement) is inferred from
# the filename timestamp: within a date, acquisitions of the same session are
# minutes apart and the next session is an hour or more later. The dataset ships
# no animal ids — figshare flattened the per-rat directory tree the loader
# scripts iterate — so this is a *derived* grouping and is labelled as such.
SESSION_GAP_S = 30 * 60

NAME_RE = re.compile(r"^DATA_(FSA(?:_Fast)?)_(\d{8})_(\d{6})\.mat$")


def figshare_files(article: int = ARTICLE, version: int = VERSION) -> list[dict]:
    with urllib.request.urlopen(FIGSHARE_API.format(article=article, version=version)) as r:
        return json.load(r)["files"]


def fsa_files(files: list[dict], include_fast: bool = True) -> list[dict]:
    """Every FSA acquisition, in acquisition order, with parsed provenance."""
    out = []
    for f in files:
        m = NAME_RE.match(f["name"])
        if m is None:
            continue
        kind, date, tod = m.groups()
        if kind == "FSA_Fast" and not include_fast:
            continue
        out.append({**f, "kind": kind, "date": date, "time": tod})
    return sorted(out, key=lambda f: (f["date"], f["time"]))


def assign_sessions(files: list[dict]) -> None:
    """Group acquisitions into inferred sessions, in place (`session` key)."""
    prev_key, prev_t, session = None, None, -1
    for f in files:
        t = int(f["time"][:2]) * 3600 + int(f["time"][2:4]) * 60 + int(f["time"][4:])
        if f["date"] != prev_key or prev_t is None or t - prev_t > SESSION_GAP_S:
            session += 1
        f["session"] = f"{f['date']}_s{session:02d}"
        prev_key, prev_t = f["date"], t


def download(url: str, dest: Path, expect_md5: str | None = None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 22):
            fh.write(chunk)
    if expect_md5:
        h = hashlib.md5()
        with open(tmp, "rb") as fh:
            while chunk := fh.read(1 << 22):
                h.update(chunk)
        if h.hexdigest() != expect_md5:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"md5 mismatch for {dest.name}: {h.hexdigest()} != {expect_md5}")
    tmp.rename(dest)


def assert_is_fsa(path: Path, meta: sm.StanfordMetadata) -> None:
    """Refuse anything that is not one-transmit-per-element multistatic data.

    A MultiFocal or Hadamard file read through this path would produce a
    plausible `(n_tx, n_rx, n_ax)` array whose transmit axis means something
    entirely different, and nothing downstream would notice.
    """
    if len(meta.groups) != meta.n_elem:
        raise ValueError(
            f"{path.name}: {len(meta.groups)} transmits for {meta.n_elem} elements — "
            f"not full synthetic aperture"
        )
    nnz = {int(np.count_nonzero(g.apod)) for g in meta.groups}
    if nnz != {1}:
        raise ValueError(
            f"{path.name}: transmit apodization has {sorted(nnz)} active elements, "
            f"expected exactly 1 (single-element transmit)"
        )
    focus = {round(g.focus_m, 6) for g in meta.groups}
    if len(focus) != 1:
        raise ValueError(f"{path.name}: {len(focus)} distinct transmit foci, expected 1")


def time_of_sample_zero(path: Path, meta: sm.StanfordMetadata) -> float:
    """Two-way time of fast-time sample 0, seconds.

    `ProcessDataFSA_L12_3v.m` defines it as ``2*startDepth*lambda/c - offset``
    with ``offset = (TW.peak + 2*lensCorrection)*lambda/c``. The lens/pulse-peak
    term is 1.77 us here — 1.4 mm of depth — so dropping it, as the NVIDIA path
    does by forcing ``t_peak = 0``, would misregister every Stanford pixel.
    """
    with h5py.File(path, "r") as f:
        peak = float(np.asarray(f["TW"]["peak"][()]).flat[0])
        lens = float(np.asarray(f["Trans"]["lensCorrection"][()]).flat[0])
    return meta.t0_s - (peak + 2.0 * lens) * meta.wavelength_m / meta.c_m_s


def decode_frame(
    path: Path, meta: sm.StanfordMetadata, frame: int, n_keep: int = N_KEEP, decim: int = DECIM
) -> tuple[np.ndarray, np.ndarray, int]:
    """One frame as `(n_tx, n_rx, n_samp)` complex64, plus HVMux coverage.

    Per transmit rather than per frame: the full-rate intermediate for all 192
    transmits is 755 MB, the decimated result is 85 MB, and there is no reason
    to hold the former.

    The returned cutoff is measured on the **raw** RF, not on the demodulated
    IQ. `rf2iq`'s low-pass rings well past the truncation — 1183 samples where
    the RF stops at 846 — so checking the IQ would reject a perfectly good
    file. Demodulation runs on the full record and truncation happens after,
    so that tail is discarded rather than folded back in.
    """
    n_out = n_keep // decim
    iq = np.empty((meta.n_elem, meta.n_elem, n_out), dtype=np.complex64)
    coverage = np.empty((meta.n_elem, meta.n_elem), dtype=np.uint8)
    last_nonzero = 0
    with h5py.File(path, "r") as f:
        rcv = sm.open_rcv_dataset(f)
        for tx, group in enumerate(meta.groups):
            for event_idx, _ in group.rx_events:
                s = event_idx * meta.n_samp
                tail = rcv[frame, :, s + n_keep : s + meta.n_samp]
                nz = np.flatnonzero(np.abs(tail).max(axis=0))
                if nz.size:
                    last_nonzero = max(last_nonzero, n_keep + int(nz[-1]))
            full, cov = sm.combine_rx_halves(group, rcv, frame, meta)  # (n_samp, n_elem)
            kept = full[:n_keep]
            if kept.shape[0] < n_keep:  # pragma: no cover - short buffer
                kept = np.pad(kept, ((0, n_keep - kept.shape[0]), (0, 0)))
            iq[tx] = resample_poly(kept, 1, decim, axis=0).astype(np.complex64).T
            coverage[tx] = cov.astype(np.uint8)
    return iq, coverage, last_nonzero


def open_or_create(out: Path, n_frames: int, n_elem: int, n_samp: int):
    root = zarr.open_group(str(out), mode="a")
    if "iq" not in root:
        root.create_dataset(
            "iq",
            shape=(n_frames, n_elem, n_elem, n_samp),
            chunks=(1, 24, n_elem, n_samp),
            dtype="complex64",
        )
        root.create_dataset(
            "coverage", shape=(n_frames, n_elem, n_elem), chunks=(1, n_elem, n_elem), dtype="uint8"
        )
    return root


def build(
    out: Path,
    scratch: Path,
    include_fast: bool = True,
    frames_per_file: int = 1,
    limit: int | None = None,
    keep_raw: bool = False,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    files = fsa_files(figshare_files(), include_fast=include_fast)
    assign_sessions(files)
    if limit is not None:
        files = files[:limit]
    n_frames = len(files) * frames_per_file
    gb = sum(f["size"] for f in files) / 1e9
    print(
        f"{len(files)} FSA acquisitions ({gb:.1f} GB raw) -> {n_frames} frames, "
        f"{len({f['session'] for f in files})} inferred sessions"
    )

    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"article": ARTICLE, "version": VERSION, "n_keep": N_KEEP, "decim": DECIM, "files": {}}
    )
    done: dict = manifest["files"]
    meta_path = out / "frame_metadata.json"
    frame_meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    root = None
    t_start = time.time()
    for i, entry in enumerate(files):
        name = entry["name"]
        base = i * frames_per_file
        if name in done:
            continue
        raw = scratch / name
        t0 = time.time()
        if not raw.exists():
            download(entry["download_url"], raw, entry.get("supplied_md5"))
        t_dl = time.time() - t0

        t0 = time.time()
        meta = sm.load_metadata(raw)
        assert_is_fsa(raw, meta)
        if meta.n_frames < frames_per_file:
            raise ValueError(f"{name}: {meta.n_frames} frames, asked for {frames_per_file}")
        attrs = acquisition_attrs(raw, meta)
        if root is None:
            root = open_or_create(out, n_frames, meta.n_elem, N_KEEP // DECIM)
            if "probe" in root.attrs:
                check_constant(dict(root.attrs), attrs, name)
            else:
                root.attrs.update(attrs)
        else:
            check_constant(dict(root.attrs), attrs, name)
        # Frames are spread across the cine rather than taken consecutively:
        # neighbouring frames of a 10-frame sequence differ only by motion.
        picks = np.linspace(0, meta.n_frames - 1, frames_per_file).round().astype(int)
        for k, frame in enumerate(picks):
            iq, coverage, last_nz = decode_frame(raw, meta, int(frame))
            if last_nz >= N_KEEP:
                raise ValueError(
                    f"{name} frame {frame}: signal extends to sample {last_nz}, "
                    f"past the {N_KEEP}-sample window — re-check the truncation"
                )
            root["iq"][base + k] = iq
            root["coverage"][base + k] = coverage
            frame_meta[f"{base + k}"] = {
                "source_file": name,
                "kind": entry["kind"],
                "date": entry["date"],
                "time": entry["time"],
                "session": entry["session"],
                "frame_in_file": int(frame),
                "n_frames_in_file": meta.n_frames,
                "last_nonzero_sample": last_nz,
                # Per-file, not corpus-constant — see `acquisition_attrs`.
                "n_samp_raw": meta.n_samp,
                "end_depth_m": meta.end_depth_m,
                "n_rx_apertures": len(meta.groups[0].rx_events),
                "md5": entry.get("supplied_md5"),
            }
        t_dec = time.time() - t0

        done[name] = {"start": base, "n_frames": frames_per_file, "size": entry["size"]}
        manifest["files"] = done
        manifest_path.write_text(json.dumps(manifest, indent=1))
        meta_path.write_text(json.dumps(frame_meta, indent=1))
        if not keep_raw:
            raw.unlink(missing_ok=True)
        print(
            f"[{base + frames_per_file:4d}/{n_frames}] {name:38s} "
            f"dl {t_dl:5.1f}s ({entry['size'] / 1e6 / max(t_dl, 1e-9):5.1f} MB/s) "
            f"decode {t_dec:5.1f}s | {(time.time() - t_start) / 60:5.1f} min elapsed",
            flush=True,
        )

    manifest["n_frames"] = n_frames
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"done — {n_frames} frames in {(time.time() - t_start) / 60:.1f} min")
    return manifest


def acquisition_attrs(path: Path, meta: sm.StanfordMetadata) -> dict:
    """Everything a beamformer needs, so the store is self-describing.

    Only quantities that are constant across the corpus belong here — the
    `FSA_Fast` acquisitions image to 37.8 mm in 2176 raw samples where the
    plain FSA ones image to 50.5 mm in 2560, and freezing either set into the
    store attributes would misdescribe the other. Both are irrelevant after
    truncation to the publishers' ~21.2 mm of real data, but they are recorded
    per frame rather than silently dropped.
    """
    with h5py.File(path, "r") as f:
        name = "".join(chr(int(c)) for c in np.asarray(f["Trans"]["name"][()]).flatten())
        bandwidth = [float(v) * 1e6 for v in np.asarray(f["Trans"]["Bandwidth"][()]).flatten()]
    return {
        "probe": name,
        "bandwidth_hz": bandwidth,
        "c_m_s": meta.c_m_s,
        "fc_hz": meta.fc_hz,
        "fd_hz": meta.fc_hz,  # rf2iq demodulates at the carrier -> baseband
        "fs_hz": meta.fs_hz / DECIM,  # after decimation
        "fs_raw_hz": meta.fs_hz,
        "wavelength_m": meta.wavelength_m,
        "pitch_m": meta.pitch_m,
        "n_elem": meta.n_elem,
        "elem_x_m": meta.elem_x_m.tolist(),
        "start_depth_m": meta.start_depth_m,
        "t0_s": time_of_sample_zero(path, meta),
        "decim": DECIM,
        "n_keep_raw_samples": N_KEEP,
        "n_ax": N_KEEP // DECIM,
        "layout": "(frame, n_tx, n_rx, n_ax) complex64; tx i fires element i",
        "source": f"figshare {ARTICLE} v{VERSION}, Apache-2.0",
    }


# Fields that must agree across every acquisition; a mismatch means the file
# is a different configuration and its frames cannot share one store.
CONSTANT_ATTRS = (
    "probe",
    "c_m_s",
    "fc_hz",
    "fs_hz",
    "wavelength_m",
    "n_elem",
    "start_depth_m",
    "t0_s",
)


def check_constant(stored: dict, incoming: dict, name: str) -> None:
    for key in CONSTANT_ATTRS:
        a, b = stored.get(key), incoming.get(key)
        same = np.allclose(a, b) if isinstance(a, (int, float)) else a == b
        if not same:
            raise ValueError(f"{name}: {key} is {b!r}, store was built with {a!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/home/benbarkow/stores/wi16_stanford_fsa.zarr")
    ap.add_argument("--scratch", default="/tmp/stanford_raw")
    ap.add_argument("--frames-per-file", type=int, default=1)
    ap.add_argument("--no-fast", action="store_true", help="skip the FSA_Fast acquisitions")
    ap.add_argument("--limit", type=int, default=None, help="first N acquisitions only")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("KERAS_BACKEND", "torch")
    build(
        Path(args.out),
        Path(args.scratch),
        include_fast=not args.no_fast,
        frames_per_file=args.frames_per_file,
        limit=args.limit,
        keep_raw=args.keep_raw,
    )


if __name__ == "__main__":
    main()
