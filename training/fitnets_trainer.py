"""
FitNets distillation (Romero et al. 2015).
Uses intermediate feature alignment (MSE) + logit KD + CE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from losses.kd_loss import kl_div_loss
from losses.task_loss import cross_entropy_loss


class FeatureAlignmentModule(nn.Module):
    """Linear projection + LayerNorm for aligning student hidden states to teacher space."""
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.norm = nn.LayerNorm(student_dim, eps=1e-5)
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        return self.proj(self.norm(x.float()))


class FitNetsTrainer(BaseTrainer):
    def __init__(self, student, teacher, tokenizer, config):
        # Create alignment module
        aligner = FeatureAlignmentModule(
            student.config.hidden_size,
            teacher.config.hidden_size
        ).to(config.device)
        super().__init__(student, teacher, tokenizer, config, aligner=aligner)
        self.mse = nn.MSELoss()

    def _compute_loss(self, batch):
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["input_ids"].clone()

        # Teacher forward
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True
            )
            t_hidden = t_out.hidden_states[self.config.teacher_align_layer]
            t_logits = t_out.logits

        # Student forward
        s_out = self.student(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True
        )
        s_hidden = s_out.hidden_states[self.config.student_align_layer]
        s_logits = s_out.logits

        # Alignment loss (MSE on projected hidden states, masked)
        mask = attn_mask.unsqueeze(-1).float()
        align_loss = self.mse(
            self.aligner(s_hidden) * mask,
            t_hidden * mask
        )

        # KD loss
        kd_loss = kl_div_loss(s_logits, t_logits, self.config.temperature)

        # CE loss
        ce_loss = cross_entropy_loss(s_logits, labels, ignore_index=self.tokenizer.pad_token_id)

        total_loss = (
            self.config.alpha_align * align_loss
            + self.config.alpha_kd * kd_loss
            + self.config.beta_ce * ce_loss
        )
        return total_loss