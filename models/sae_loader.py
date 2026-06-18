import torch
from pathlib import Path
from .sae import SparseAutoencoder

def load_sae(checkpoint_path, device, dtype):
    sae = SparseAutoencoder()
    state = torch.load(checkpoint_path, map_location="cpu")
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