import torch.nn.functional as F

def cross_entropy_loss(logits, labels, ignore_index=-100):
    """
    Standard next-token prediction loss.
    logits: [B, T, vocab]
    labels: [B, T] (shifted inside)
    """
    # Shift so that we predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    return loss