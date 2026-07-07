"""
Training utilities for ReasonDistill-compatible Sparse Autoencoders.

The objective follows Galichin et al. (2025):
    ||x - x_hat||_2^2 + lambda * sum_i f_i ||W_dec,i||_2

The trainer attaches to a teacher transformer layer by reading the selected
hidden state, flattens valid token activations, and optimizes a vanilla ReLU SAE.
"""

import json
import math
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class SAETrainingConfig:
    teacher_model_id: str
    layer: int
    checkpoint_dir: str
    max_length: int = 1024
    sequence_batch_size: int = 4
    token_batch_size: int = 4096
    max_train_tokens: int = 1_000_000_000
    latent_dim: int | None = None
    expansion_factor: int = 16
    lr: float = 5e-5
    min_lr: float = 0.0
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    sparsity_coefficient: float = 5.0
    sparsity_warmup_fraction: float = 0.05
    lr_decay_fraction: float = 0.20
    save_every_steps: int = 1000
    log_every_steps: int = 50
    log_every_samples: int = 1000
    log_path: str | None = None
    num_workers: int = 0
    cache_dir: str | None = None
    max_samples: int | None = None
    seed: int = 42


class LastFractionLinearDecay:
    """Keeps LR flat, then linearly decays it over the final training fraction."""

    def __init__(self, optimizer, total_steps, base_lr, min_lr=0.0, decay_fraction=0.20):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.decay_steps = max(1, int(self.total_steps * decay_fraction))
        self.decay_start = max(0, self.total_steps - self.decay_steps)
        self.step_num = 0
        self._set_lr(self.base_lr)

    def _set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def step(self):
        self.step_num += 1
        if self.step_num <= self.decay_start:
            lr = self.base_lr
        else:
            progress = min(1.0, (self.step_num - self.decay_start) / self.decay_steps)
            lr = self.base_lr + progress * (self.min_lr - self.base_lr)
        self._set_lr(lr)
        return lr

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, state):
        self.step_num = int(state.get("step_num", 0))
        if self.step_num <= self.decay_start:
            lr = self.base_lr
        else:
            progress = min(1.0, (self.step_num - self.decay_start) / self.decay_steps)
            lr = self.base_lr + progress * (self.min_lr - self.base_lr)
        self._set_lr(lr)


class SampleCSVLogger:
    """Averages SAE metrics and writes one CSV row per configured sample window."""

    fieldnames = [
        "sample_start",
        "sample_end",
        "num_samples",
        "step_start",
        "step_end",
        "num_steps",
        "tokens_start",
        "tokens_end",
        "num_tokens",
        "avg_total_loss",
        "avg_reconstruction_loss",
        "avg_sparsity_loss",
        "avg_sparsity",
        "avg_zero_ratio",
        "avg_l0_norm",
        "avg_l1_norm",
        "avg_avg_nonzero_activation",
        "sparsity_coefficient",
        "lr",
    ]

    def __init__(self, log_path, log_every_samples, append=False):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_every_samples = max(1, int(log_every_samples))
        self.mode = "a" if append and self.log_path.exists() else "w"
        self.file = self.log_path.open(self.mode, newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if self.mode == "w":
            self.writer.writeheader()
            self.file.flush()
        self.reset()

    def reset(self):
        self.sample_start = None
        self.sample_end = None
        self.step_start = None
        self.step_end = None
        self.tokens_start = None
        self.tokens_end = None
        self.num_samples = 0
        self.num_steps = 0
        self.num_tokens = 0
        self.sums = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "sparsity_loss": 0.0,
            "zero_ratio": 0.0,
            "l0_norm": 0.0,
            "l1_norm": 0.0,
            "avg_nonzero_activation": 0.0,
        }

    def add(
        self,
        *,
        sample_start,
        sample_end,
        step,
        tokens_start,
        tokens_end,
        total_loss,
        reconstruction_loss,
        sparsity_loss,
        stats,
        sparsity_coefficient,
        lr,
        force=False,
    ):
        if self.sample_start is None:
            self.sample_start = sample_start
            self.step_start = step
            self.tokens_start = tokens_start

        num_samples = sample_end - sample_start + 1
        num_tokens = tokens_end - tokens_start
        self.sample_end = sample_end
        self.step_end = step
        self.tokens_end = tokens_end
        self.num_samples += num_samples
        self.num_steps += 1
        self.num_tokens += num_tokens
        self.sums["total_loss"] += float(total_loss)
        self.sums["reconstruction_loss"] += float(reconstruction_loss)
        self.sums["sparsity_loss"] += float(sparsity_loss)
        self.sums["zero_ratio"] += float(stats["zero_ratio"])
        self.sums["l0_norm"] += float(stats["l0_norm"])
        self.sums["l1_norm"] += float(stats["l1_norm"])
        self.sums["avg_nonzero_activation"] += float(stats["avg_nonzero_activation"])

        if self.num_samples < self.log_every_samples and not force:
            return

        denom = max(1, self.num_steps)
        avg_zero_ratio = self.sums["zero_ratio"] / denom
        self.writer.writerow(
            {
                "sample_start": self.sample_start,
                "sample_end": self.sample_end,
                "num_samples": self.num_samples,
                "step_start": self.step_start,
                "step_end": self.step_end,
                "num_steps": self.num_steps,
                "tokens_start": self.tokens_start,
                "tokens_end": self.tokens_end,
                "num_tokens": self.num_tokens,
                "avg_total_loss": self.sums["total_loss"] / denom,
                "avg_reconstruction_loss": self.sums["reconstruction_loss"] / denom,
                "avg_sparsity_loss": self.sums["sparsity_loss"] / denom,
                "avg_sparsity": avg_zero_ratio,
                "avg_zero_ratio": avg_zero_ratio,
                "avg_l0_norm": self.sums["l0_norm"] / denom,
                "avg_l1_norm": self.sums["l1_norm"] / denom,
                "avg_avg_nonzero_activation": self.sums["avg_nonzero_activation"] / denom,
                "sparsity_coefficient": sparsity_coefficient,
                "lr": lr,
            }
        )
        self.file.flush()
        self.reset()

    def close(self):
        self.file.close()


class SAETrainer:
    def __init__(self, teacher, tokenizer, sae, dataset, config: SAETrainingConfig, device):
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.sae = sae
        self.dataset = dataset
        self.config = config
        self.device = device
        self.global_step = 0
        self.tokens_seen = 0
        self.samples_seen = 0

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=config.lr,
            betas=config.adam_betas,
            weight_decay=config.weight_decay,
        )
        estimated_steps = math.ceil(config.max_train_tokens / max(1, config.token_batch_size))
        self.scheduler = LastFractionLinearDecay(
            self.optimizer,
            total_steps=estimated_steps,
            base_lr=config.lr,
            min_lr=config.min_lr,
            decay_fraction=config.lr_decay_fraction,
        )

    def _collate(self, batch):
        texts = [item["text"] for item in batch]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def _layer_activations(self, batch):
        with torch.no_grad():
            outputs = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states
        hidden_index = self.config.layer + 1
        if hidden_index >= len(hidden_states):
            last_layer = len(hidden_states) - 2
            raise ValueError(
                f"Layer {self.config.layer} is out of range. "
                f"Teacher exposes transformer layers 0..{last_layer}."
            )

        activations = hidden_states[hidden_index]
        mask = batch["attention_mask"].bool()
        activations = activations[mask].float()
        if activations.size(0) > self.config.token_batch_size:
            indices = torch.randperm(activations.size(0), device=activations.device)
            activations = activations[indices[: self.config.token_batch_size]]
        return activations

    def _sparsity_lambda(self):
        warmup_steps = max(
            1,
            int(math.ceil(self.scheduler.total_steps * self.config.sparsity_warmup_fraction)),
        )
        progress = min(1.0, self.global_step / warmup_steps)
        return self.config.sparsity_coefficient * progress

    def _save_checkpoint(self, name):
        ckpt_dir = Path(self.config.checkpoint_dir) / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.sae.state_dict(), ckpt_dir / "sae.pt")
        torch.save(self.sae.state_dict(), Path(self.config.checkpoint_dir) / "sae.pt")
        with (ckpt_dir / "config.json").open("w") as f:
            json.dump(asdict(self.config), f, indent=2)
        with (Path(self.config.checkpoint_dir) / "config.json").open("w") as f:
            json.dump(asdict(self.config), f, indent=2)
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "tokens_seen": self.tokens_seen,
                "samples_seen": self.samples_seen,
            },
            ckpt_dir / "trainer_state.pt",
        )
        print(f"SAE checkpoint saved: {ckpt_dir}")

    def load_checkpoint(self, checkpoint_dir):
        ckpt_dir = Path(checkpoint_dir)
        self.sae.load_state_dict(torch.load(ckpt_dir / "sae.pt", map_location=self.device))
        trainer_state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu")
        self.optimizer.load_state_dict(trainer_state["optimizer"])
        self.scheduler.load_state_dict(trainer_state["scheduler"])
        self.global_step = int(trainer_state.get("global_step", 0))
        self.tokens_seen = int(trainer_state.get("tokens_seen", 0))
        self.samples_seen = int(trainer_state.get("samples_seen", 0))

    def train(self):
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.sequence_batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=self.config.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        self.sae.train()
        running = {"loss": 0.0, "recon": 0.0, "sparsity": 0.0, "l0": 0.0}
        running_count = 0
        log_path = self.config.log_path or str(Path(self.config.checkpoint_dir) / "sae_training_metrics.csv")
        csv_logger = SampleCSVLogger(
            log_path,
            self.config.log_every_samples,
            append=(self.global_step > 0),
        )
        print(
            f"Logging SAE metrics every {self.config.log_every_samples} samples "
            f"to {csv_logger.log_path}"
        )
        pbar = tqdm(total=self.config.max_train_tokens, initial=self.tokens_seen, desc="SAE tokens")

        try:
            while self.tokens_seen < self.config.max_train_tokens:
                for batch in loader:
                    if self.tokens_seen >= self.config.max_train_tokens:
                        break

                    samples_this_step = int(batch["input_ids"].size(0))
                    sample_start = self.samples_seen + 1
                    sample_end = self.samples_seen + samples_this_step
                    tokens_start = self.tokens_seen
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    activations = self._layer_activations(batch)
                    if activations.numel() == 0:
                        self.samples_seen = sample_end
                        continue
                    remaining_tokens = self.config.max_train_tokens - self.tokens_seen
                    if activations.size(0) > remaining_tokens:
                        activations = activations[:remaining_tokens]

                    self.sae.sparsity_coefficient = self._sparsity_lambda()
                    reconstructed, latent = self.sae(activations)
                    loss, recon_loss, sparsity_loss = self.sae.total_loss(
                        activations,
                        reconstructed,
                        latent,
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    self.samples_seen = sample_end
                    tokens_this_step = int(activations.size(0))
                    self.tokens_seen += tokens_this_step
                    pbar.update(tokens_this_step)

                    stats = self.sae.get_sparsity_stats(latent.detach())
                    total_loss_value = float(loss.detach())
                    recon_loss_value = float(recon_loss.detach())
                    sparsity_loss_value = float(sparsity_loss.detach())
                    lr = self.scheduler.get_lr()

                    csv_logger.add(
                        sample_start=sample_start,
                        sample_end=sample_end,
                        step=self.global_step,
                        tokens_start=tokens_start,
                        tokens_end=self.tokens_seen,
                        total_loss=total_loss_value,
                        reconstruction_loss=recon_loss_value,
                        sparsity_loss=sparsity_loss_value,
                        stats=stats,
                        sparsity_coefficient=self.sae.sparsity_coefficient,
                        lr=lr,
                        force=(self.tokens_seen >= self.config.max_train_tokens),
                    )

                    running["loss"] += total_loss_value
                    running["recon"] += recon_loss_value
                    running["sparsity"] += sparsity_loss_value
                    running["l0"] += stats["l0_norm"]
                    running_count += 1

                    if self.global_step % self.config.log_every_steps == 0:
                        denom = max(1, running_count)
                        print(
                            f"step {self.global_step} | samples {self.samples_seen:,} | "
                            f"tokens {self.tokens_seen:,} | "
                            f"loss {running['loss'] / denom:.6f} | "
                            f"recon {running['recon'] / denom:.6f} | "
                            f"sparsity {running['sparsity'] / denom:.6f} | "
                            f"l0 {running['l0'] / denom:.2f} | "
                            f"lambda {self.sae.sparsity_coefficient:.4f} | "
                            f"lr {lr:.2e}"
                        )
                        running = {"loss": 0.0, "recon": 0.0, "sparsity": 0.0, "l0": 0.0}
                        running_count = 0

                    if (
                        self.config.save_every_steps > 0
                        and self.global_step % self.config.save_every_steps == 0
                    ):
                        self._save_checkpoint(f"step_{self.global_step}")
        finally:
            csv_logger.close()
            pbar.close()

        self._save_checkpoint("final")
