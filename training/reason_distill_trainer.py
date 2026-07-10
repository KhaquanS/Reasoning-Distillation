"""
ReasonDistill: reasoning-neuron alignment using SAE + KL + CE.
"""

import torch
from .base_trainer import BaseTrainer
from losses.alignment_loss import ReasoningAlignmentLoss
from losses.kd_loss import kl_div_loss
from losses.task_loss import cross_entropy_loss


class ReasonDistillTrainer(BaseTrainer):
    def __init__(self, student, teacher, tokenizer, sae, reasoning_neurons, aligner, config):
        # aligner is the ReasoningFeatureHead
        super().__init__(student, teacher, tokenizer, config, aligner=aligner)
        self.sae = sae
        self.reasoning_neurons = reasoning_neurons

        # Precompute decoder norms
        with torch.no_grad():
            dec_norms = self.sae.decoder.weight.float().norm(dim=0)

        self.alignment_loss_fn = ReasoningAlignmentLoss(
            sae,
            reasoning_neurons,
            dec_norms,
            config.target_norm,
            active_threshold=config.reason_active_threshold,
            presence_weight=config.reason_presence_weight,
            rank_weight=config.reason_rank_weight,
        )
        self.alignment_loss_fn.aligner = self.aligner

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

        # Alignment loss (selected SAE reasoning features)
        align_loss = self.alignment_loss_fn(s_hidden, t_hidden, attn_mask.float())

        # KD loss
        kd_loss = kl_div_loss(
            student_logits=s_logits,
            teacher_logits=t_logits,
            attention_mask=attn_mask,
            temperature=self.config.temperature,
        )

        # CE loss
        ce_loss = cross_entropy_loss(s_logits, labels, ignore_index=self.tokenizer.pad_token_id)

        total_loss = (
            self.config.alpha_align * align_loss
            + self.config.alpha_kd * kd_loss
            + self.config.beta_ce * ce_loss
        )
        return total_loss
