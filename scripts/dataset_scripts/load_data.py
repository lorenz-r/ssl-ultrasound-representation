"""
Dataset loader for Verasonics ultrasound .mat files.
Translates scripts/dataset_scripts/mainLoader.m and related processing functions.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List



def load_mat(filepath: str, keys: Optional[List[str]] = None) -> dict:
    """Load a Verasonics .mat file and return a structured dictionary.
    
    Args:
        filepath: path to .mat file
        keys: list of keys to load (None = load all keys)
    """
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):
                continue
            if keys is not None and key not in keys:
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                data[key] = _parse_dataset(obj)
            elif isinstance(obj, h5py.Group):
                data[key] = _parse_group(obj)
    return data


def _parse_dataset(ds: h5py.Dataset) -> np.ndarray:
    arr = ds[()]
    while arr.dtype == object:
        arr = arr[()]
        
    return np.asarray(arr)


def _parse_group(group: h5py.Group) -> dict:
    out = {}
    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Dataset):
            out[key] = _parse_dataset(obj)
        elif isinstance(obj, h5py.Group):
            out[key] = _parse_group(obj)
    return out


def get_scalar(val) -> float:
    """Extract scalar from possibly wrapped array."""
    if isinstance(val, np.ndarray):
        return float(val.flatten()[0])
    return float(val)


def get_channel_data(
    mat_path: str,
    max_samples: Optional[int] = None,
    max_channels: Optional[int] = None,
) -> Tuple[np.ndarray, float, np.ndarray, dict]:
    """
    Load channel data from a Verasonics .mat file.

    Args:
        mat_path: path to .mat file
        max_samples: limit number of samples to load (None = all samples)
        max_channels: limit number of channels to load (None = all channels)

    Returns:
        channel_data: raw channel RF data
        c: speed of sound (m/s)
        t: time vector
        params: additional metadata dict
    """
    required_keys = ['P', 'TW', 'Trans', 'Receive', 'TX']
    data = load_mat(mat_path, keys=required_keys)

    P = data['P']
    TW = data['TW']
    Trans = data['Trans']

    c = get_scalar(P['txFocus']) * 1000
    dt = 1.0 / get_scalar(data['Receive']['ADCRate'])
    num_tx = int(get_scalar(P['numTx']))

    with h5py.File(mat_path, 'r') as f:
        RcvData = f['RcvData']
        full_samples = RcvData.shape[0]
        full_channels = RcvData.shape[1]

        num_samples = min(max_samples, full_samples) if max_samples else full_samples
        num_channels = min(max_channels, full_channels) if max_channels else full_channels

        RcvData_slice = RcvData[:num_samples, :num_channels]

    channel_data = RcvData_slice.flatten()

    t = np.arange(num_samples) * dt * 1e6

    params = {
        'start_depth': get_scalar(P['startDepth']),
        'end_depth': get_scalar(P['endDepth']),
        'num_rays': int(get_scalar(P['numRays'])),
        'num_tx': num_tx,
        'tx_focus': get_scalar(P['txFocus']),
        'tx_delays': _get_tx_delays(data),
        'element_pos': Trans['ElementPos'],
        'num_channels': num_channels,
        'num_samples': num_samples,
    }

    return channel_data, c, t, params


def _get_tx_delays(data: dict) -> np.ndarray:
    TW = data['TW']
    if 'Delay' in data['TX']:
        delays = data['TX']['Delay'].flatten()
    else:
        cum_on = TW['CumOnTime'].flatten()
        delays = cum_on - cum_on[0] if len(cum_on) > 0 else np.array([])
    return np.array([d for d in delays if d > 0])


def _detect_data_type(filename: str) -> str:
    """Detect acquisition type from filename."""
    name_lower = filename.lower()
    if 'fsa' in name_lower:
        return 'FSA'
    if 'hadamard' in name_lower:
        return 'Hadamard'
    if 'multifocal' in name_lower:
        return 'MultiFocal'
    return 'Unknown'


def load_experiment_data(
    data_dir: str = 'data/rats',
    processed_dir: Optional[str] = None,
    save_data: bool = False,
    max_files: Optional[int] = None,
    max_samples: Optional[int] = None,
    max_channels: Optional[int] = None,
) -> List[dict]:
    """
    Process all .mat files in experiment directories.

    Args:
        data_dir: root directory containing experiment data
        processed_dir: directory to save processed channel data
        save_data: whether to save processed data to .mat files
        max_files: limit number of files to process (None = all files)
        max_samples: limit number of samples per file (None = all samples)
        max_channels: limit number of channels per file (None = all channels)

    Returns:
        List of dicts with 'filepath', 'channel_data', 'c', 't', 'params'
    """
    results = []

    data_dir = Path(data_dir)
    if processed_dir is None:
        processed_dir = data_dir.parent / 'processed'
    else:
        processed_dir = Path(processed_dir)

    if save_data:
        processed_dir.mkdir(parents=True, exist_ok=True)

    mat_files = list(data_dir.glob('**/DATA_*.mat'))
    if max_files is not None:
        mat_files = mat_files[:max_files]
    print(f"Found {len(mat_files)} data files to process")

    for mat_path in mat_files:
        filename = mat_path.name
        dtype = _detect_data_type(filename)
        print(f"Processing {mat_path} (type: {dtype})")

        try:
            channel_data, c, t, params = get_channel_data(str(mat_path), max_samples=max_samples, max_channels=max_channels)
            print(f"  channel_data shape: {channel_data.shape}, c={c:.0f} m/s")

            entry = {
                'filepath': str(mat_path),
                'filename': filename,
                'data_type': dtype,
                'channel_data': channel_data,
                'c': c,
                't': t,
                'params': params,
            }
            results.append(entry)

            if save_data:
                print("  Skipping save (scipy not available)")

        except Exception as e:
            print(f"  Error processing {mat_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


if __name__ == '__main__':
    results = load_experiment_data(
        data_dir='data/rats/Rat1',
        processed_dir='data/rats/Rat1/processed',
        save_data=True,
        max_files=1,
        max_samples=1,
        max_channels=3
    )
    print(f"\nProcessed {len(results)} files")