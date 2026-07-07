import torch
import torch.nn.functional as F


class ReasoningAlignmentLoss:
    """
    Distills selected teacher SAE reasoning features into a compact student
    reasoning-feature head.

    Teacher targets are produced by the frozen SAE encoder. Student predictions
    stay in the student's native representation space and are mapped directly to
    the selected reasoning coordinates.
    """
    def __init__(
        self,
        sae,
        reasoning_neurons,
        dec_norms,
        target_norm,
        active_threshold=0.0,
        presence_weight=0.25,
        rank_weight=0.10,
    ):
        self.sae = sae
        self.reasoning_neurons = reasoning_neurons
        self.dec_norms = dec_norms
        self.target_norm = target_norm
        self.active_threshold = active_threshold
        self.presence_weight = presence_weight
        self.rank_weight = rank_weight
        self.aligner = None   # set later

    def encode_teacher_features(self, hidden):
        B, T, D = hidden.shape
        h = hidden.reshape(B * T, D)
        h_norm = h * (self.target_norm / h.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        acts = self.sae.encode(h_norm)
        acts = acts * self.dec_norms
        return acts[:, self.reasoning_neurons].reshape(B, T, -1)

    def __call__(self, s_hidden, t_hidden, attn_mask):
        if self.aligner is None:
            raise RuntimeError("Aligner not set in alignment loss")

        with torch.no_grad():
            t_feats = self.encode_teacher_features(t_hidden.float())
            active = t_feats > self.active_threshold
            target_presence = active.float()

        s_logits = self.aligner(s_hidden)
        s_feats = F.softplus(s_logits)
        mask = attn_mask.unsqueeze(-1).float()
        n_real = mask.sum().clamp(min=1.0)
        n_features = self.reasoning_neurons.size(0)
        denom = n_real * n_features

        value_loss = ((s_feats - t_feats) ** 2 * mask).sum() / denom
        presence_loss = (
            F.binary_cross_entropy_with_logits(s_logits, target_presence, reduction="none") * mask
        ).sum() / denom

        if self.rank_weight > 0:
            centered_s = s_logits - s_logits.mean(dim=-1, keepdim=True)
            centered_t = t_feats - t_feats.mean(dim=-1, keepdim=True)
            rank_loss = ((centered_s - centered_t) ** 2 * mask).sum() / denom
        else:
            rank_loss = s_logits.new_zeros(())

        return value_loss + self.presence_weight * presence_loss + self.rank_weight * rank_loss
