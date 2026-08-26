"""Dataset APIs, imported lazily so Zarr and Parquet remain independent."""

from __future__ import annotations

from importlib import import_module


_ZARR_EXPORTS = {"UltrasoundMAEDataset", "build_split_datasets"}
_NV_EXPORTS = {
    "NvidiaRaw2InsightsDataset",
    "build_nv_raw2insights_datasets",
    "iter_nv_raw2insights_samples",
    "nv_iq_to_mae_frame",
    "nv_sample_to_fsa_params",
}

__all__ = sorted(_ZARR_EXPORTS | _NV_EXPORTS)


def __getattr__(name: str):
    if name in _ZARR_EXPORTS:
        module = import_module(".dataset", __name__)
    elif name in _NV_EXPORTS:
        module = import_module(".nv_raw2insights", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
