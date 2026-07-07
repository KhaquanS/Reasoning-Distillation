import torch
from pathlib import Path
from .sae import SparseAutoencoder

def load_sae(checkpoint_path, device, dtype):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "sae.pt"

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    encoder_weight = state["encoder.weight"]
    latent_dim, activation_dim = encoder_weight.shape
    sparsity_coefficient = 5.0
    if "sparsity_coefficient" in state:
        sparsity_coefficient = float(state["sparsity_coefficient"])

    sae = SparseAutoencoder(
        activation_dim=activation_dim,
        latent_dim=latent_dim,
        sparsity_coefficient=sparsity_coefficient,
    )
    sae.load_state_dict(state)
    sae = sae.to(dtype).to(device)
    for p in sae.parameters():
        p.requires_grad_(False)
    sae.eval()
    return sae

def load_reasoning_neurons(reason_score_path, top_k, device):
    data = torch.load(reason_score_path, map_location="cpu")
    sorted_indices = data["sorted_indices"]
    top_indices = sorted_indices[:top_k].long()
    return top_indices.to(device)
