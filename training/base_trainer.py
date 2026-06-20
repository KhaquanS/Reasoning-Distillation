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
        self.scheduler_state = None

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

    def _save_checkpoint(self, epoch):
        """Save checkpoint including student, aligner (if any), optimizer, scheduler."""
        ckpt_dir = Path(self.config.checkpoint_dir) / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        if self.aligner is not None:
            torch.save(self.aligner.state_dict(), ckpt_dir / "aligner.pt")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch
        }, ckpt_dir / "trainer_state.pt")
        print(f"Checkpoint saved: {ckpt_dir}")

    def _load_checkpoint(self, checkpoint_dir):
        """
        Load checkpoint and restore state.
        Sets self.start_epoch to the saved epoch number.
        Returns updated student and aligner.
        """
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        state_path = ckpt_dir / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"trainer_state.pt missing in {ckpt_dir}")

        state = torch.load(state_path, map_location=self.config.device)
        self.start_epoch = state["epoch"]  # already completed epoch

        # Reload student
        student = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=self.config.dtype,
            device_map={"": self.config.device}
        )
        self.student.load_state_dict(student.state_dict())

        # Reload aligner if present
        if self.aligner is not None:
            aligner_path = ckpt_dir / "aligner.pt"
            if aligner_path.exists():
                self.aligner.load_state_dict(torch.load(aligner_path, map_location=self.config.device))

        # Restore optimizer
        self.optimizer.load_state_dict(state["optimizer"])
        # Store scheduler state for later restoration (after scheduler creation)
        self.scheduler_state = state.get("scheduler", None)

        print(f"Resumed from checkpoint: {ckpt_dir}, starting epoch {self.start_epoch}")
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

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.lr,
            total_steps=total_opt_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy="cos",
            final_div_factor=self.config.lr / self.config.min_lr
        )

        # Restore scheduler state if available
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
            self.scheduler_state = None
        elif self.start_epoch > 0:
            # Fast-forward scheduler
            skip = (steps_per_epoch * self.start_epoch) // self.config.accum_steps
            print(f"Fast-forwarding scheduler {skip} steps...")
            for _ in range(skip):
                self.scheduler.step()

        self.print_freq = max(1, steps_per_epoch // 10)

        for epoch in range(self.start_epoch, self.config.epochs):
            self._run_epoch(epoch, loader)

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

            # Gradient accumulation
            loss = loss / self.config.accum_steps
            loss.backward()

            if (step + 1) % self.config.accum_steps == 0 or (step + 1) == len(loader):
                params = list(self.student.parameters())
                if self.aligner is not None:
                    params += list(self.aligner.parameters())
                torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            if step % self.print_freq == 0 and step > 0:
                avg = epoch_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"  step {step:5d} | loss {avg:.6f} | lr {lr:.2e}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} finished | avg loss: {avg_loss:.6f}")
        self._save_checkpoint(epoch + 1)