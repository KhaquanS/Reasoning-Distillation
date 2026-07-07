from typing import Optional

import torch.nn as nn


class ReasoningFeatureHead(nn.Module):
    """
    Predicts selected teacher SAE reasoning-feature activations directly from
    student hidden states.

    This avoids forcing student states into the full teacher hidden geometry.
    The head instead learns a compact readout from the student's native layer
    into the K selected SAE coordinates used by ReasonDistill.
    """
    def __init__(
        self,
        student_dim: int,
        num_reasoning_features: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(student_dim, eps=1e-5)
        if hidden_dim is None:
            self.proj = nn.Linear(student_dim, num_reasoning_features, bias=True)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            self.proj = nn.Sequential(
                nn.Linear(student_dim, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, num_reasoning_features, bias=True),
            )
            for module in self.proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.proj(self.norm(x.float()))


ReasoningNeuronAligner = ReasoningFeatureHead
