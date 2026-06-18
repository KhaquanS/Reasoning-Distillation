import torch

class ReasoningAlignmentLoss:
    def __init__(self, sae, reasoning_neurons, dec_norms, target_norm):
        self.sae = sae
        self.reasoning_neurons = reasoning_neurons
        self.dec_norms = dec_norms
        self.target_norm = target_norm

    def encode_sae(self, hidden):
        B, T, D = hidden.shape
        h = hidden.reshape(B * T, D)
        # Normalize
        h_norm = h * (self.target_norm / h.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        acts = self.sae.encode(h_norm)
        acts = acts * self.dec_norms
        # Extract reasoning neurons
        return acts[:, self.reasoning_neurons].reshape(B, T, -1)

    def __call__(self, s_hidden, t_hidden, attn_mask):
        with torch.no_grad():
            t_feats = self.encode_sae(t_hidden.float())
        s_proj = self.aligner(s_hidden)   # aligner is set externally
        s_feats = self.encode_sae(s_proj)
        mask = attn_mask.unsqueeze(-1)
        n_real = mask.sum().clamp(min=1.0)
        return ((s_feats - t_feats) ** 2 * mask).sum() / (n_real * self.reasoning_neurons.size(0))