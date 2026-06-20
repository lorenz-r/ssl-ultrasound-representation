"""
Backwards-compatibility shim: the canonical dataset module now lives in the
installable package at `ssl_ultrasound_representation.dataset.dataset`.

Kept so existing imports (`from src.dataset import build_split_datasets`,
`scripts/dataset_scripts/push_to_hf.py`, the HF dataset card) keep working.
"""

from ssl_ultrasound_representation.dataset.dataset import (  # noqa: F401
    CACHE_ENV_VAR,
    DEFAULT_ARRAY_NAME,
    DEFAULT_CACHE_DIR,
    DEFAULT_METADATA_FILENAME,
    DEFAULT_ZARR_DIRNAME,
    DEFAULT_ZARR_FILENAME,
    UltrasoundMAEDataset,
    build_split_datasets,
    load_dataset,
    load_local_dataset,
    resolve_cache_dir,
    split_by_file,
    split_by_subject,
)
