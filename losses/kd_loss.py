import torch
import torch.nn.functional as F


def kl_div_loss(
    student_logits,
    teacher_logits,
    attention_mask,
    temperature=4.0,
):
    """
    Token-wise KL divergence for knowledge distillation.

    Args:
        student_logits: [B, T, V]
        teacher_logits: [B, T, V]
        attention_mask: [B, T] (1 for valid tokens, 0 for padding)
        temperature: Distillation temperature.

    Returns:
        Mean KL divergence over valid tokens.
    """

    # Temperature-scaled distributions
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Token-wise KL divergence
    # Shape: [B, T, V] -> [B, T]
    kl = F.kl_div(
        s_log_probs,
        t_probs,
        reduction="none",
    ).sum(dim=-1)

    # Mask out padding tokens
    mask = attention_mask.float()

    kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)

    # Standard temperature correction
    return kl * (temperature ** 2)