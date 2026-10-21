import torch
import math

def get_sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Generates sinusoidal positional embeddings for a Transformer.
    
    Args:
        seq_len (int): Length of the sequence (e.g., 110592).
        d_model (int): Dimension of the embedding model (e.g., 128).
        
    Returns:
        torch.Tensor: Positional encoding matrix of shape (seq_len, d_model).
    """
    # Initialize a matrix of zeros with shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    
    # Create a column vector of sequence positions: [[0], [1], [2], ..., [seq_len-1]]
    # Shape becomes (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # Calculate the frequency division term: 1 / (10000^(2i/d_model))
    # We use log space for numerical stability
    # Shape becomes (d_model / 2,)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    
    # Apply sine to all even indices in the embedding dimension (0, 2, 4...)
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Apply cosine to all odd indices in the embedding dimension (1, 3, 5...)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def split_dims(d_model: int, n_axes: int) -> list[int]:
    """
    Split `d_model` into `n_axes` even-sized chunks summing to `d_model`.

    Each chunk must be even so its axis gets whole sin/cos pairs. The remainder
    is handed out two dimensions at a time to the leading axes, so with
    d_model=256 over 3 axes the split is (86, 86, 84).
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")
    if n_axes < 1:
        raise ValueError(f"n_axes must be >= 1, got {n_axes}")
    pairs, extra = divmod(d_model // 2, n_axes)
    if pairs == 0:
        raise ValueError(
            f"d_model={d_model} is too small to split across {n_axes} axes "
            "(each axis needs at least one sin/cos pair)"
        )
    return [2 * (pairs + (1 if i < extra else 0)) for i in range(n_axes)]


def build_grid_positional_encoding(grid_sizes: tuple[int, ...], d_model: int) -> torch.Tensor:
    """
    Positional encoding for a flattened N-D token grid: `d_model` is split into
    one contiguous block of dimensions per axis, each block holding that axis'
    sinusoidal encoding. Blocks are concatenated, broadcast over the grid and
    flattened row-major (last axis fastest).

    Dedicated blocks rather than a sum over full-width per-axis encodings: with
    the same encoding function applied to every axis, summing makes the result
    symmetric under permuting coordinates between axes, so e.g. (1,0,0),
    (0,1,0) and (0,0,1) collapse to one identical vector and the encoder cannot
    tell those tokens apart. Disjoint blocks keep every grid position distinct.

    Args:
        grid_sizes: token-grid extent per axis, e.g. (scanlines, focal_zones,
            depth_windows). Flattening order must match the model's patchify.
        d_model: embedding dimension.

    Returns:
        Tensor of shape (prod(grid_sizes), d_model).
    """
    dims = split_dims(d_model, len(grid_sizes))
    blocks = []
    for axis, (n, d_axis) in enumerate(zip(grid_sizes, dims)):
        shape = [1] * len(grid_sizes) + [d_axis]
        shape[axis] = n
        block = get_sinusoidal_positional_encoding(n, d_axis).view(*shape)
        blocks.append(block.expand(*grid_sizes, d_axis))
    return torch.cat(blocks, dim=-1).reshape(-1, d_model)