import torch.nn.functional as F

def kl_div_loss(student_logits, teacher_logits, temperature=4.0):
    """
    KL Div between teacher and student logits.
    student_logits, teacher_logits: [B, T, vocab]
    """
    # Soften distributions
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    # KL divergence: sum(p_t * (log(p_t) - log(p_s)))
    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)
    return kl