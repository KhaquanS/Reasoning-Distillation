"""
Base trainer class providing common functionality for all distillation methods.
Subclasses must implement _compute_loss().
"""

import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from utils.csv_logger import AveragedCSVLogger


class BaseTrainer:
    """
    Base class for all distillation trainers.

    Args:
        student: The student model (trainable).
        teacher: The teacher model (frozen).
        tokenizer: Tokenizer used for collation.
        config: Namespace with training hyperparameters.
        aligner: Optional alignment module (used by FitNets and ReasonDistill).
    """
    def __init__(self, student, teacher, tokenizer, config, aligner=None):
        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.aligner = aligner
        self.config = config

        # Set up optimizer
        params = list(student.parameters())
        if aligner is not None:
            params += list(aligner.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )

        self.scheduler = None
        self.start_epoch = 0
        self.global_step = 0  # counts optimizer steps (not batches), across epochs

    def _collate(self, batch):
        """Collate function for DataLoader."""
        texts = [item["text"] for item in batch]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

    def _save_checkpoint(self, epoch, global_step=None):
        """
        Save checkpoint including student, aligner (if any), optimizer, scheduler.

        Epoch-end checkpoints are saved to `epoch_{epoch}`. Mid-epoch, step-based
        checkpoints (triggered by config.save_every_n_steps) are saved to
        `step_{global_step}` instead, so they don't collide with or overwrite
        the epoch-end checkpoints.
        """
        if global_step is not None:
            ckpt_dir = Path(self.config.checkpoint_dir) / f"step_{global_step}"
        else:
            ckpt_dir = Path(self.config.checkpoint_dir) / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        if self.aligner is not None:
            torch.save(self.aligner.state_dict(), ckpt_dir / "aligner.pt")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "global_step": self.global_step,
        }, ckpt_dir / "trainer_state.pt")
        print(f"Checkpoint saved: {ckpt_dir}")

    def load_module_weights(self, checkpoint_dir):
        """
        Load student and optional aligner weights from a checkpoint directory.

        Optimizer, scheduler, and epoch state are intentionally not restored so
        the current YAML config fully controls the new training run.
        """
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        print(f"Loading student weights from {ckpt_dir} ...")
        student = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=self.config.dtype,
            device_map={"": self.config.device}
        )
        self.student.load_state_dict(student.state_dict())
        print("✅ Student weights loaded.")

        if self.aligner is not None:
            aligner_path = ckpt_dir / "aligner.pt"
            if aligner_path.exists():
                self.aligner.load_state_dict(torch.load(aligner_path, map_location=self.config.device))
                print("✅ Aligner weights loaded.")
            else:
                print(f"ℹ️  No aligner.pt found in {ckpt_dir}; keeping freshly initialized aligner.")

        self.start_epoch = 0
        print(f"✅ Module weights successfully loaded from: {ckpt_dir}")
        return self.student, self.aligner

    def _compute_loss(self, batch):
        """
        Compute the total loss for a batch.
        Must be implemented by subclasses.
        Returns a scalar tensor.
        """
        raise NotImplementedError

    def train(self, train_dataset):
        """Main training loop."""
        loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=0,
            pin_memory=(self.config.device == "cuda")
        )
        steps_per_epoch = len(loader)
        total_opt_steps = (steps_per_epoch * self.config.epochs) // self.config.accum_steps
        # If there's a remainder from integer division, we need an extra step
        if (steps_per_epoch * self.config.epochs) % self.config.accum_steps != 0:
            total_opt_steps += 1

        # Total optimizer steps for this run. Tracked separately from the
        # scheduler object itself since not every scheduler type exposes a
        # `.total_steps` attribute (OneCycleLR does, CosineAnnealingWarmRestarts
        # does not).
        self.total_opt_steps = total_opt_steps

        scheduler_type = getattr(self.config, "scheduler_type", "onecycle")

        if scheduler_type == "cosine_restarts":
            # Warm up linearly for `warmup_ratio` of the run, then hand off to
            # cosine annealing with warm restarts (SGDR). Each restart pops the
            # LR back up to `config.lr` and anneals it down to `config.min_lr`,
            # which lets the optimizer escape shallow minima instead of grinding
            # to a halt once the LR decays close to zero.
            warmup_steps = max(1, round(total_opt_steps * self.config.warmup_ratio))
            restart_interval = getattr(self.config, "restart_interval_steps", None)
            if not restart_interval:
                # Default to a single restart halfway through the (post-warmup)
                # run if the user didn't specify an explicit interval.
                restart_interval = max(1, (total_opt_steps - warmup_steps) // 2)
            restart_mult = getattr(self.config, "restart_mult", 1) or 1

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.min_lr / self.config.lr,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            restart_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=restart_interval,
                T_mult=restart_mult,
                eta_min=self.config.min_lr
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, restart_scheduler],
                milestones=[warmup_steps]
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                total_steps=total_opt_steps,
                pct_start=self.config.warmup_ratio,
                anneal_strategy="cos",
                final_div_factor=self.config.lr / self.config.min_lr
            )

        self.print_freq = max(1, steps_per_epoch // 10)
        target_log_rows = getattr(self.config, "loss_log_entries_per_epoch", 1000)
        log_dir = Path(getattr(self.config, "log_dir", self.config.checkpoint_dir))
        self.loss_logger = AveragedCSVLogger(
            log_dir / "training_loss.csv",
            steps_per_epoch=steps_per_epoch,
            target_rows_per_epoch=target_log_rows,
            append=(self.start_epoch > 0),
        )
        print(
            f"Logging averaged losses every {self.loss_logger.log_interval} steps "
            f"to {self.loss_logger.log_path}"
        )

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                self._run_epoch(epoch, loader)
        finally:
            self.loss_logger.close()

        print(f"Training complete. Final checkpoint: {self.config.checkpoint_dir}/epoch_{self.config.epochs}")

    def _run_epoch(self, epoch, loader):
        """Run a single epoch."""
        self.student.train()
        if self.aligner is not None:
            self.aligner.train()

        epoch_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            # Compute the scheme-specific loss
            loss = self._compute_loss(batch)
            raw_loss = loss.detach().item()

            # Gradient accumulation
            loss = loss / self.config.accum_steps
            loss.backward()

            if (step + 1) % self.config.accum_steps == 0 or (step + 1) == len(loader):
                params = list(self.student.parameters())
                if self.aligner is not None:
                    params += list(self.aligner.parameters())
                torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                self.optimizer.step()
                self.global_step += 1

                # Only step scheduler if we haven't exceeded total steps
                if self.global_step < self.total_opt_steps:
                    self.scheduler.step()

                self.optimizer.zero_grad()

                save_every_n_steps = getattr(self.config, "save_every_n_steps", None)
                if save_every_n_steps and self.global_step % save_every_n_steps == 0:
                    self._save_checkpoint(epoch, global_step=self.global_step)

            epoch_loss += raw_loss
            num_batches += 1
            lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.lr
            self.loss_logger.add(
                epoch=epoch,
                step=step,
                loss=raw_loss,
                lr=lr,
                force=((step + 1) == len(loader)),
            )

            if step % self.print_freq == 0 and step > 0:
                avg = epoch_loss / num_batches
                print(f"  step {step:5d} | loss {avg:.6f} | lr {lr:.2e}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} finished | avg loss: {avg_loss:.6f}")
        self._save_checkpoint(epoch + 1)