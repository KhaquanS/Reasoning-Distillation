import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from .sae import SparseAutoencoder


def _resolve_checkpoint_path(path, repo_type="model"):
    """
    Resolve a checkpoint reference to a local file path.

    Accepts two forms:
      - A plain local path or directory, used as-is (existing behavior).
      - An "hf://" reference to a file hosted on the Hugging Face Hub, e.g.

            hf://Khaquan/qwen-khaquanS-distillations/qwen-3.5-4B-L16_16x-SAE/sae.pt

        which is resolved as repo_id="Khaquan/qwen-khaquanS-distillations",
        filename="qwen-3.5-4B-L16_16x-SAE/sae.pt", downloaded via
        huggingface_hub (and transparently cached locally on subsequent
        calls). An optional "@revision" can be appended to the repo id to
        pin a specific branch/tag/commit, e.g.:

            hf://Khaquan/qwen-khaquanS-distillations@main/qwen-3.5-4B-L16_16x-SAE/sae.pt
    """
    path = str(path)
    if not path.startswith("hf://"):
        return Path(path)

    ref = path[len("hf://"):]
    parts = ref.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid hf:// reference '{path}'. Expected "
            "'hf://<namespace>/<repo_name>/<path/to/file>'."
        )

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
    )
    return Path(local_path)


def load_sae(checkpoint_path, device, dtype):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
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
    reason_score_path = _resolve_checkpoint_path(reason_score_path)
    data = torch.load(reason_score_path, map_location="cpu")
    sorted_indices = data["sorted_indices"]
    top_indices = sorted_indices[:top_k].long()
    return top_indices.to(device)