import torch.nn as nn


class ReasoningNeuronAligner(nn.Module):
    """
    Projects the student's hidden states (dim=2048) into the teacher's
    hidden space (dim=4096) before the frozen SAE encoder.

    - LayerNorm for stability (using student dim).
    - Linear projection without bias (Xavier initialised) to preserve
      the original student representation scale.
    """
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(student_dim, eps=1e-5)
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        # Cast to float32 for LayerNorm numerical stability
        return self.proj(self.norm(x.float()))