import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.amdeepseek import AMDeepSeekDataset
from models.sae import SparseAutoencoder
from models.student_loader import load_tokenizer
from training.sae_trainer import SAETrainer, SAETrainingConfig
from utils.seed import set_seed


OPTIONAL_CONFIG = {
    "cache_dir": None,
    "max_samples": None,
    "max_length": 1024,
    "sequence_batch_size": 4,
    "token_batch_size": 4096,
    "max_train_tokens": 1_000_000_000,
    "latent_dim": None,
    "expansion_factor": 16,
    "lr": 5e-5,
    "min_lr": 0.0,
    "weight_decay": 0.0,
    "adam_betas": (0.9, 0.999),
    "max_grad_norm": 1.0,
    "sparsity_coefficient": 5.0,
    "sparsity_warmup_fraction": 0.05,
    "lr_decay_fraction": 0.20,
    "save_every_steps": 1000,
    "log_every_steps": 50,
    "log_every_samples": 1000,
    "log_path": None,
    "num_workers": 0,
    "seed": 42,
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "attn_implementation": None,
    "resume_from": None,
}


def _load_yaml_config(config_path):
    with Path(config_path).open("r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a mapping at the top level")

    config = dict(OPTIONAL_CONFIG)
    for section in ("model", "sae", "data", "training", "logging", "experiment"):
        values = raw.get(section, {})
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
        config.update(values)

    required = ["teacher_model_id", "layer", "checkpoint_dir"]
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")

    config["layer"] = int(config["layer"])
    config["adam_betas"] = tuple(config["adam_betas"])
    if len(config["adam_betas"]) != 2:
        raise ValueError("training.adam_betas must contain exactly two values")

    return argparse.Namespace(**config)


def _str_or_none(value):
    if value is None:
        return None
    lowered = str(value).lower()
    if lowered in {"", "none", "null"}:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a ReasonDistill-compatible SAE on AMDeepSeek teacher activations. "
            "The paper trained SAEs on 1B tokens; override --max-train-tokens for smaller runs."
        )
    )
    parser.add_argument("--config", type=str, help="Path to a YAML SAE training config.")
    parser.add_argument("--teacher-model-id", type=str, help="Hugging Face teacher model id.")
    parser.add_argument("--layer", type=int, help="Zero-indexed teacher transformer layer for SAE activations.")
    parser.add_argument("--checkpoint-dir", type=str, help="Output directory for SAE checkpoints.")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--sequence-batch-size", type=int, default=None)
    parser.add_argument("--token-batch-size", type=int, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--expansion-factor", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--adam-betas", type=float, nargs=2, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--sparsity-coefficient", type=float, default=None)
    parser.add_argument("--sparsity-warmup-fraction", type=float, default=None)
    parser.add_argument("--lr-decay-fraction", type=float, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--log-every-samples", type=int, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    cli = parser.parse_args()
    if cli.config:
        args = _load_yaml_config(cli.config)
    else:
        args = argparse.Namespace(**OPTIONAL_CONFIG)

    for key, value in vars(cli).items():
        if key == "config" or value is None:
            continue
        setattr(args, key, value)

    required = ["teacher_model_id", "layer", "checkpoint_dir"]
    missing = [key for key in required if getattr(args, key, None) is None]
    if missing:
        raise ValueError(
            f"Missing required argument(s): {', '.join('--' + key.replace('_', '-') for key in missing)}"
        )
    args.adam_betas = tuple(args.adam_betas)
    if len(args.adam_betas) != 2:
        raise ValueError("--adam-betas must contain exactly two values")
    return args


def _torch_dtype(name, device):
    if device != "cuda":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _load_teacher(args, device, dtype):
    kwargs = {
        "torch_dtype": dtype,
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    attn_implementation = _str_or_none(args.attn_implementation)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if device == "cuda":
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.teacher_model_id, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _torch_dtype(args.dtype, device)

    tokenizer = load_tokenizer(args.teacher_model_id, cache_dir=args.cache_dir)
    dataset = AMDeepSeekDataset(
        split="train",
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )
    teacher = _load_teacher(args, device, dtype)

    activation_dim = int(teacher.config.hidden_size)
    latent_dim = args.latent_dim or activation_dim * int(args.expansion_factor)
    sae = SparseAutoencoder(
        activation_dim=activation_dim,
        latent_dim=int(latent_dim),
        sparsity_coefficient=float(args.sparsity_coefficient),
    ).to(device)

    config = SAETrainingConfig(
        teacher_model_id=args.teacher_model_id,
        layer=args.layer,
        checkpoint_dir=args.checkpoint_dir,
        max_length=args.max_length,
        sequence_batch_size=args.sequence_batch_size,
        token_batch_size=args.token_batch_size,
        max_train_tokens=args.max_train_tokens,
        latent_dim=int(latent_dim),
        expansion_factor=args.expansion_factor,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_betas=args.adam_betas,
        max_grad_norm=args.max_grad_norm,
        sparsity_coefficient=args.sparsity_coefficient,
        sparsity_warmup_fraction=args.sparsity_warmup_fraction,
        lr_decay_fraction=args.lr_decay_fraction,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps,
        log_every_samples=args.log_every_samples,
        log_path=args.log_path,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    trainer = SAETrainer(teacher, tokenizer, sae, dataset, config, device)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
