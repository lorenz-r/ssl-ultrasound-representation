import torch
import torch.nn as nn

from .linear_projection import LinearEncoder, LinearDecoder
from .pos_encoding import get_sinusoidal_positional_encoding

class SignalTransformerAutoencoder(nn.Module):
    def __init__(self, original_channels=847, d_model=128, nhead=8, num_layers=4, dim_feedforward=512):
        """
        Args:
            original_channels (int): The original signal depth (847 in diagram).
            d_model (int): The embedding dimension (128 in diagram).
            nhead (int): Number of attention heads (must divide d_model evenly).
            num_layers (int): Number of stacked Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network inside the Transformer.
        """
        super().__init__()
        
        # 1. Input Linear Projection (847 => 128)
        # We use a 1D convolution here as a clean way to project channels across a sequence
        self.input_projection = LinearEncoder(in_channels=original_channels, 
                                          out_channels=d_model)
        
        # 2. Transformer Block
        # Setting batch_first=True expects inputs of shape (Batch, Sequence, Features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu", # GELU is standard for modern transformers
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Linear Projection (128 => 847)
        self.output_projection = LinearDecoder(in_channels=d_model, 
                                              out_channels=original_channels)

    def forward(self, x, pos_embedding):
        """
        Args:
            x: Tensor of shape (Batch, 847, 110592)
            pos_embedding: Tensor of shape (110592, 128)
            
        Returns:
            Reconstructed Tensor of shape (Batch, 847, 110592)
        """
        # x is currently (Batch, Channels=847, Sequence=110592)
        
        # Project down to embedding dimension: 847 -> 128
        x = self.input_projection(x) # Shape becomes (Batch, 128, 110592)
        
        # Transformers expect sequence length in the middle dimension if batch_first=True
        # Permute to: (Batch, Sequence=110592, Channels=128)
        x = x.permute(0, 2, 1) 
        
        # Add positional embedding (broadcasted across the batch)
        x = x + pos_embedding.unsqueeze(0).to(x.device)
        
        # Pass through the Transformer block
        x = self.transformer(x) # Shape remains (Batch, 110592, 128)
        
        # Permute back to channel-first for the final projection
        x = x.permute(0, 2, 1) # Shape becomes (Batch, 128, 110592)
        
        # Project back to original dimension: 128 -> 847
        x = self.output_projection(x) # Shape becomes (Batch, 847, 110592)
        
        return x