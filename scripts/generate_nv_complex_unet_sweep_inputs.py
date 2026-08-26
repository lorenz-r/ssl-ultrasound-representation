"""Generate nested NV training manifests and deterministic fixed TX masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--mask-counts", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--reference-manifest")
    return parser.parse_args()


def fixed_mask_order(width: int = 180) -> list[int]:
    order = list(range(10, width, 20))
    order += list(range(0, width, 20))
    order += list(range(5, width, 10))
    while len(order) < width:
        remaining = [index for index in range(width) if index not in order]
        candidate = max(
            remaining,
            key=lambda index: (
                min(abs(index - chosen) for chosen in order),
                -index,
            ),
        )
        order.append(candidate)
    return order


def read_indices(path: Path) -> list[int]:
    return [int(value) for value in path.read_text().replace(",", " ").split()]


def main() -> None:
    args = parse_args()
    metadata = json.loads(Path(args.metadata).read_text())
    train_indices = [
        int(frame["global_frame_index"])
        for frame in metadata["frames"]
        if frame.get("split") == "train"
    ]
    if max(args.sample_sizes) > len(train_indices):
        raise ValueError(
            f"requested {max(args.sample_sizes)} samples from "
            f"{len(train_indices)} training frames"
        )

    permutation = torch.randperm(
        len(train_indices), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    nested_indices = [train_indices[index] for index in permutation]
    if args.reference_manifest:
        reference = read_indices(Path(args.reference_manifest))
        if nested_indices[: len(reference)] != reference:
            raise RuntimeError("generated nested order does not match reference manifest")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_size in args.sample_sizes:
        path = output_dir / f"n{sample_size}_train_global_indices.txt"
        path.write_text(",".join(map(str, nested_indices[:sample_size])) + "\n")
        print(f"wrote {path}: {sample_size} frames")

    mask_order = fixed_mask_order()
    for mask_count in args.mask_counts:
        if not 1 <= mask_count < len(mask_order):
            raise ValueError(f"mask count must be in [1, 179], got {mask_count}")
        selected = sorted(mask_order[:mask_count])
        path = output_dir / f"k{mask_count:03d}_masked_tx.txt"
        path.write_text(",".join(map(str, selected)) + "\n")
        print(f"wrote {path}: {mask_count} TX positions")


if __name__ == "__main__":
    main()
