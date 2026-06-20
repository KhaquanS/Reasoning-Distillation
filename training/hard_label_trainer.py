"""
Hard-label distillation: use teacher's argmax tokens as targets.
Loss = CE(student_logits, teacher_argmax)
"""

import torch
import torch.nn.functional as F
from .base_trainer import BaseTrainer


class HardLabelTrainer(BaseTrainer):
    def __init__(self, student, teacher, tokenizer, config):
        super().__init__(student, teacher, tokenizer, config, aligner=None)
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def _compute_loss(self, batch):
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        # No ground truth labels used; we use teacher argmax

        # Teacher forward
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=False
            )
            t_logits = t_out.logits

        # Student forward
        s_out = self.student(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=False
        )
        s_logits = s_out.logits

        # Teacher argmax as hard targets
        # We need to align vocab sizes if different (but Llama tokenizers are same)
        t_argmax = t_logits.argmax(dim=-1)  # (B, T)

        # Mask padding positions: set to -100 for ignore_index
        # We'll create a target tensor with -100 where attention_mask == 0
        targets = t_argmax.clone()
        targets[attn_mask == 0] = -100

        # Cross-entropy loss (over all positions, but ignoring padding)
        loss = self.ce(
            s_logits.view(-1, s_logits.size(-1)),
            targets.view(-1)
        )
        return loss