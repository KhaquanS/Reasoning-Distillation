"""
Sparse Autoencoder for learning compressed representations of teacher activations.

Implementation based on "I Have Covered All the Bases Here: Interpreting Reasoning
Features in Large Language Models via Sparse Autoencoders" (2025).

The autoencoder learns to:
1. Encode high-dimensional teacher activations into a sparse latent space
2. Decode the sparse representation back to the original dimensions
3. Enforce sparsity using modified L1 regularization (weighted by decoder norm)

Architecture (from paper Section 2, Equation 1-2):
    f(x) = σ(W_enc * x + b_enc)        # Encoder with ReLU activation
    x̂(f) = W_dec * f + b_dec            # Decoder
    L = ||x - x̂||²₂ + λ Σᵢ fᵢ ||W_dec,i||₂  # Loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Vanilla Sparse Autoencoder with ReLU activation and modified L1 penalty.

    This implementation exactly matches the paper's architecture:
    - ReLU activation function for non-negativity
    - Squared L2 reconstruction loss
    - Modified L1 penalty: λ Σᵢ fᵢ ||W_dec,i||₂

    Note: latent_dim should be MUCH LARGER than activation_dim (overcomplete representation)
    to allow the SAE to learn disentangled features. The paper uses 16x expansion (65,536 vs 4,096).

    Args:
        activation_dim (int): Dimension of input activations (n in paper, e.g., 4096 for LLaMA-8B)
        latent_dim (int): Dimension of sparse latent representation (m in paper, should be >> n, e.g., 65536)
        sparsity_coefficient (float): λ in paper - L1 penalty weight for sparsity (default: 5.0)
        tie_weights (bool): Whether to tie encoder and decoder weights (decoder = encoder.T)
    """

    def __init__(self, activation_dim=4096, latent_dim=65536, 
                 sparsity_coefficient=5.0, tie_weights=False):
        super(SparseAutoencoder, self).__init__()

        self.activation_dim = activation_dim  # n
        self.latent_dim = latent_dim          # m >> n
        self.sparsity_coefficient = sparsity_coefficient  # λ
        self.tie_weights = tie_weights

        # Encoder: W_enc and b_enc (Equation 1)
        self.encoder = nn.Linear(activation_dim, latent_dim, bias=True)
        
        # Decoder: W_dec and b_dec (Equation 1)
        if tie_weights:
            # Tied weights: W_dec = W_enc^T
            self.decoder = nn.Linear(latent_dim, activation_dim, bias=True)
            # We'll handle weight tying in _initialize_weights
        else:
            self.decoder = nn.Linear(latent_dim, activation_dim, bias=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        Following standard practice for autoencoders.
        """
        # Initialize encoder
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        
        # Tie weights if requested
        if self.tie_weights:
            # W_dec = W_enc^T
            self.decoder.weight.data = self.encoder.weight.data.t()

    def encode(self, x):
        """
        Encode input activations to sparse latent representation.

        Implements: f(x) = σ(W_enc * x + b_enc) where σ is ReLU (Equation 1)

        Args:
            x (Tensor): Input activations of shape (batch_size, seq_len, activation_dim)
                       or (batch_size, activation_dim)

        Returns:
            Tensor: Sparse latent codes f(x) of shape (batch_size, seq_len, latent_dim)
                   or (batch_size, latent_dim)
        """
        # Defensive cast: upstream activations (e.g. from a quantized or
        # hybrid-attention teacher) can silently come back in fp32 even when
        # the SAE itself is bf16. Cast here so callers never have to worry
        # about matching dtypes exactly before calling encode().
        if x.dtype != self.encoder.weight.dtype:
            x = x.to(self.encoder.weight.dtype)
        return F.relu(self.encoder(x))

    def decode(self, latent):
        """
        Decode latent representation back to activation space.

        Implements: x̂(f) = W_dec * f + b_dec (Equation 1)

        Args:
            latent (Tensor): Latent codes f of shape (batch_size, seq_len, latent_dim)
                           or (batch_size, latent_dim)

        Returns:
            Tensor: Reconstructed activations x̂ of shape (batch_size, seq_len, activation_dim)
                   or (batch_size, activation_dim)
        """
        if latent.dtype != self.decoder.weight.dtype:
            latent = latent.to(self.decoder.weight.dtype)
        return self.decoder(latent)

    def forward(self, x):
        """
        Forward pass: encode then decode.

        Args:
            x (Tensor): Input activations

        Returns:
            tuple: (reconstructed, latent)
                - reconstructed: Decoded output x̂
                - latent: Sparse latent representation f
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def compute_sparsity_loss(self, latent):
        """
        Compute modified L1 sparsity penalty: L_sparsity = λ Σᵢ fᵢ ||W_dec,i||₂

        This is the key difference from standard SAEs - each latent activation is weighted
        by the L2 norm of its corresponding decoder weight vector (Equation 2 in paper).

        Args:
            latent (Tensor): Latent codes f of shape (..., latent_dim)

        Returns:
            Tensor: Scalar sparsity loss
        """
        # Compute L2 norm of each decoder column: ||W_dec,i||₂
        # decoder.weight shape: (activation_dim, latent_dim)
        # dec_norms[i] = ||W_dec[:, i]||₂
        dec_norms = torch.norm(self.decoder.weight, p=2, dim=0)  # (latent_dim,)

        # Compute weighted L1: Σᵢ fᵢ ||W_dec,i||₂
        # latent: (..., latent_dim), dec_norms: (latent_dim,)
        weighted_l1 = (latent * dec_norms).sum(dim=-1).mean()

        # Apply sparsity coefficient λ
        sparsity_loss = self.sparsity_coefficient * weighted_l1

        return sparsity_loss

    def reconstruction_loss(self, x, reconstructed):
        """
        Compute squared L2 reconstruction loss: L_recon = ||x - x̂||²₂

        Args:
            x (Tensor): Original activations
            reconstructed (Tensor): Reconstructed activations x̂

        Returns:
            Tensor: Scalar MSE (squared L2) loss
        """
        return F.mse_loss(reconstructed, x, reduction='mean')

    def total_loss(self, x, reconstructed, latent):
        """
        Compute total loss: L = ||x - x̂||²₂ + λ Σᵢ fᵢ ||W_dec,i||₂ (Equation 2)

        Args:
            x (Tensor): Original activations
            reconstructed (Tensor): Reconstructed activations x̂
            latent (Tensor): Sparse latent codes f

        Returns:
            tuple: (total_loss, recon_loss, sparsity_loss)
        """
        recon_loss = self.reconstruction_loss(x, reconstructed)
        sparsity_loss = self.compute_sparsity_loss(latent)
        total = recon_loss + sparsity_loss

        return total, recon_loss, sparsity_loss

    def get_sparsity_stats(self, latent):
        """
        Compute statistics about sparsity in the latent representation.

        Args:
            latent (Tensor): Latent codes

        Returns:
            dict: Dictionary with sparsity statistics
        """
        # Percentage of zero activations
        zero_ratio = (latent == 0).float().mean().item()

        # Average L0 norm (number of non-zero elements)
        l0_norm = (latent != 0).float().sum(dim=-1).mean().item()

        # Average L1 norm
        l1_norm = torch.abs(latent).sum(dim=-1).mean().item()

        # Average activation value (for non-zero elements)
        non_zero_mask = latent != 0
        if non_zero_mask.any():
            avg_nonzero = latent[non_zero_mask].mean().item()
        else:
            avg_nonzero = 0.0

        return {
            'zero_ratio': zero_ratio,
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'avg_nonzero_activation': avg_nonzero
        }