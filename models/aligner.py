import torch 
import torch.nn as nn

class ReasoningNeuronAligner(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(student_dim, eps=1e-5)
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for LayerNorm numerical stability; output stays float32
        # so it feeds directly into _encode_sae without an extra cast.
        return self.proj(self.norm(x.float()))