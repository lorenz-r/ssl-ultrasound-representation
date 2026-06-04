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