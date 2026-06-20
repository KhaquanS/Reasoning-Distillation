"""
Standard logit-level knowledge distillation (Hinton et al.).
Loss = alpha_kd * KL(softmax(student/T) || softmax(teacher/T)) * T^2
       + beta_ce * CE(student_logits, ground_truth)
"""

import torch
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from losses.kd_loss import kl_div_loss
from losses.task_loss import cross_entropy_loss


class LogitKDTrainer(BaseTrainer):
    def __init__(self, student, teacher, tokenizer, config):
        super().__init__(student, teacher, tokenizer, config, aligner=None)

    def _compute_loss(self, batch):
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["input_ids"].clone()  # for task loss

        # Teacher forward (frozen)
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=False  # only logits needed
            )
            t_logits = t_out.logits

        # Student forward
        s_out = self.student(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=False
        )
        s_logits = s_out.logits

        # KD loss
        kd_loss = kl_div_loss(s_logits, t_logits, self.config.temperature)

        # CE loss on ground truth
        ce_loss = cross_entropy_loss(s_logits, labels, ignore_index=self.tokenizer.pad_token_id)

        total_loss = self.config.alpha_kd * kd_loss + self.config.beta_ce * ce_loss
        return total_loss