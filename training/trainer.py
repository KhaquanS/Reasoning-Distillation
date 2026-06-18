import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM

from losses.alignment_loss import ReasoningAlignmentLoss
from losses.kd_loss import kl_div_loss
from losses.task_loss import cross_entropy_loss


class ReasonDistillTrainer:
    def __init__(
        self,
        student,
        teacher,
        tokenizer,
        sae,
        reasoning_neurons,
        aligner,
        config
    ):
        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.sae = sae
        self.reasoning_neurons = reasoning_neurons
        self.aligner = aligner
        self.config = config

        # Precompute decoder norms
        with torch.no_grad():
            self.dec_norms = self.sae.decoder.weight.float().norm(dim=0)

        self.alignment_loss_fn = ReasoningAlignmentLoss(
            sae, reasoning_neurons, self.dec_norms, config.target_norm
        )

        # Attach aligner to the loss function
        self.alignment_loss_fn.aligner = self.aligner

        self.optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(aligner.parameters()),
            lr=config.lr,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )

        # Scheduler will be set in train()
        self.scheduler = None
        self.scheduler_state = None   # store saved scheduler state for later
        self.start_epoch = 0

    def train(self, train_dataset, val_dataset=None):
        loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=0,
            pin_memory=True
        )
        steps_per_epoch = len(loader)
        self.print_freq = max(1, steps_per_epoch // 10)  # print 10 logs per epoch
        total_opt_steps = (steps_per_epoch * self.config.epochs) // self.config.accum_steps

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.lr,
            total_steps=total_opt_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy="cos",
            final_div_factor=self.config.lr / self.config.min_lr
        )

        # If we have a saved scheduler state from load_checkpoint, restore it now
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
            self.scheduler_state = None  # prevent reuse

        # Fast-forward scheduler if resuming after some epochs
        if self.start_epoch > 0:
            skip = (steps_per_epoch * self.start_epoch) // self.config.accum_steps
            for _ in range(skip):
                self.scheduler.step()

        for epoch in range(self.start_epoch, self.config.epochs):
            self._run_epoch(epoch, loader)

    def _run_epoch(self, epoch, loader):
        self.student.train()
        self.aligner.train()
        epoch_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            labels = batch["input_ids"].clone()  # for task loss

            # ----- Teacher forward -----
            with torch.no_grad():
                t_out = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True
                )
                t_hidden = t_out.hidden_states[self.config.teacher_align_layer]
                t_logits = t_out.logits

            # ----- Student forward -----
            s_out = self.student(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True
            )
            s_hidden = s_out.hidden_states[self.config.student_align_layer]
            s_logits = s_out.logits

            # Losses
            loss_align = self.alignment_loss_fn(
                s_hidden, t_hidden, attn_mask.float()
            )
            loss_kd = kl_div_loss(s_logits, t_logits, self.config.temperature)
            loss_task = cross_entropy_loss(s_logits, labels, ignore_index=self.tokenizer.pad_token_id)

            total_loss = (
                self.config.alpha_align * loss_align +
                self.config.alpha_kd * loss_kd +
                self.config.beta_ce * loss_task
            )

            # Gradient accumulation
            loss = total_loss / self.config.accum_steps
            loss.backward()

            if (step + 1) % self.config.accum_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    list(self.student.parameters()) + list(self.aligner.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            epoch_loss += total_loss.item()
            num_batches += 1

            if step % self.print_freq == 0 and step > 0:
                print(f"  step {step} | loss {epoch_loss/num_batches:.6f} | lr {self.scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} finished | avg loss: {avg_loss:.6f}")
        # Save checkpoint
        self._save_checkpoint(epoch+1)

    def _collate(self, batch):
        texts = [item["text"] for item in batch]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

    def _save_checkpoint(self, epoch):
        ckpt_dir = Path(self.config.checkpoint_dir) / f"rsnd_epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        torch.save(self.aligner.state_dict(), ckpt_dir / "aligner.pt")
        # Save optimizer and scheduler state for full resumption
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch
        }, ckpt_dir / "trainer_state.pt")
        print(f"Checkpoint saved: {ckpt_dir}")

    def load_checkpoint(self, checkpoint_dir):
        """
        Resume training from a previously saved checkpoint.

        Args:
            checkpoint_dir (str or Path): Directory containing:
                - student model (saved with save_pretrained)
                - aligner.pt
                - trainer_state.pt (optimizer, scheduler, epoch)

        This method:
            - Restores the student model weights
            - Restores the aligner weights
            - Restores the optimizer state
            - Stores the scheduler state (to be loaded after scheduler creation)
            - Sets self.start_epoch to the saved epoch number (next epoch to train)
        """
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # 1. Load trainer state (optimizer, scheduler, epoch)
        state_path = ckpt_dir / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"trainer_state.pt not found in {ckpt_dir}")
        state = torch.load(state_path, map_location=self.config.device)

        # 2. Set start epoch (this is the epoch we have already completed,
        #    so training will resume from this index)
        self.start_epoch = state["epoch"]   # e.g., if saved epoch=3, start at 3

        # 3. Load aligner state
        aligner_path = ckpt_dir / "aligner.pt"
        if not aligner_path.exists():
            raise FileNotFoundError(f"aligner.pt not found in {ckpt_dir}")
        self.aligner.load_state_dict(
            torch.load(aligner_path, map_location=self.config.device)
        )

        # 4. Load student model from pretrained (using the saved config and weights)
        student_loaded = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=self.config.dtype,
            device_map={"": self.config.device},
        )
        # Copy weights into our existing student object
        self.student.load_state_dict(student_loaded.state_dict())

        # 5. Restore optimizer state
        self.optimizer.load_state_dict(state["optimizer"])

        # 6. Store scheduler state for later loading (after scheduler is created)
        self.scheduler_state = state.get("scheduler", None)

        print(f"Resumed from checkpoint: {ckpt_dir}")
        print(f"\tStarting at epoch {self.start_epoch} (0-indexed)")
