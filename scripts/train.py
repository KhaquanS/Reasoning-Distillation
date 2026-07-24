import argparse
import sys
import torch
import yaml
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.math500 import Math500Dataset
from data_loaders.metamathqa import MetaMathQADataset
from data_loaders.mixture import MixtureDataset
from data_loaders.amdeepseek import AMDeepSeekDataset
from models.student_loader import load_teacher, load_student, load_tokenizer, load_student_checkpoint
from models.sae_loader import load_sae, load_reasoning_neurons
from models.aligner import ReasoningFeatureHead
from training import (
    LogitKDTrainer,
    FitNetsTrainer,
    HardLabelTrainer,
    ReasonDistillTrainer,
)
from utils.seed import set_seed


OPTIONAL_CONFIG = {
    "mix_ratio": None,
    "max_samples": None,
    "skip_samples": 0,
    "sae_checkpoint": None,
    "reason_score_path": None,
    "student_checkpoint": None,
    "log_dir": None,
    "loss_log_entries_per_epoch": 1000,
    "teacher_align_layer": 19,
    "student_align_layer": 10,
    "target_norm": 64.0,
    "reason_active_threshold": 0.0,
    "reason_presence_weight": 0.25,
    "reason_rank_weight": 0.10,
    "reasoning_neuron_count": 196,
    "teacher_quantize_8bit": True,
    "save_every_n_steps": None,
    "scheduler_type": "onecycle",
    "restart_interval_steps": None,
    "restart_mult": 1,
}

CONFIG_SECTIONS = {
    "experiment": ["seed"],
    "data": ["dataset", "mix_ratio", "max_samples", "skip_samples", "cache_dir"],
    "method": ["sae_checkpoint", "reason_score_path", "reasoning_neuron_count"],
    "model": [
        "teacher",
        "student",
        "student_checkpoint",
        "teacher_quantize_8bit",
    ],
    "alignment": [
        "teacher_align_layer",
        "student_align_layer",
        "target_norm",
        "reason_active_threshold",
        "reason_presence_weight",
        "reason_rank_weight",
    ],
    "training": [
        "max_length",
        "batch_size",
        "accum_steps",
        "epochs",
        "lr",
        "min_lr",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
        "adam_betas",
        "scheduler_type",
        "restart_interval_steps",
        "restart_mult",
    ],
    "loss": ["temperature", "alpha_kd", "alpha_align", "beta_ce"],
    "logging": ["checkpoint_dir", "log_dir", "loss_log_entries_per_epoch", "save_every_n_steps"],
}

REQUIRED_CONFIG = [
    "method",
    "dataset",
    "cache_dir",
    "teacher",
    "student",
    "max_length",
    "batch_size",
    "accum_steps",
    "epochs",
    "lr",
    "min_lr",
    "warmup_ratio",
    "weight_decay",
    "max_grad_norm",
    "adam_betas",
    "temperature",
    "alpha_kd",
    "alpha_align",
    "beta_ce",
    "checkpoint_dir",
    "seed",
]

VALID_METHODS = {"logit_kd", "fitnets", "hard_label", "reasondistill"}
VALID_DATASETS = {"math500", "metamathqa", "mixture", "amdeepseek"}


def _load_yaml_config(config_path):
    with Path(config_path).open("r") as f:
        raw_config = yaml.safe_load(f) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("YAML config must be a mapping at the top level")

    config = dict(OPTIONAL_CONFIG)
    for section_name, keys in CONFIG_SECTIONS.items():
        section = raw_config.get(section_name, {})
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"Config section '{section_name}' must be a mapping")
        for key in keys:
            if key in section:
                config[key] = section[key]

    method_section = raw_config.get("method", {})
    if isinstance(method_section, dict) and "name" in method_section:
        config["method"] = method_section["name"]

    missing = [key for key in REQUIRED_CONFIG if key not in config or config[key] is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")

    if config["method"] not in VALID_METHODS:
        raise ValueError(f"Unknown method '{config['method']}'. Expected one of {sorted(VALID_METHODS)}")
    if config["dataset"] not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset '{config['dataset']}'. Expected one of {sorted(VALID_DATASETS)}")
    if config["dataset"] == "mixture" and config["mix_ratio"] is None:
        raise ValueError("data.mix_ratio is required when data.dataset is 'mixture'")
    if config["method"] == "reasondistill":
        if config["sae_checkpoint"] is None or config["reason_score_path"] is None:
            raise ValueError(
                "method.sae_checkpoint and method.reason_score_path are required for reasondistill"
            )
    if len(config["adam_betas"]) != 2:
        raise ValueError("training.adam_betas must contain exactly two values")
    if config["scheduler_type"] not in ("onecycle", "cosine_restarts"):
        raise ValueError(
            f"Unknown training.scheduler_type '{config['scheduler_type']}'. "
            "Expected 'onecycle' or 'cosine_restarts'"
        )
    if config["scheduler_type"] == "cosine_restarts" and config["restart_interval_steps"] is not None:
        if int(config["restart_interval_steps"]) <= 0:
            raise ValueError("training.restart_interval_steps must be a positive integer if set")
    if config["save_every_n_steps"] is not None and int(config["save_every_n_steps"]) <= 0:
        raise ValueError("logging.save_every_n_steps must be a positive integer if set")

    config["config_path"] = str(config_path)
    return SimpleNamespace(**config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train from a YAML experiment config. Training flags are intentionally disabled."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment config.",
    )
    args = parser.parse_args()
    return _load_yaml_config(args.config)


def main():
    args = parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Config: method={args.method}, dataset={args.dataset}, samples={args.max_samples}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.student, cache_dir=args.cache_dir)

    # Load datasets
    if args.dataset == "math500":
        train_ds = Math500Dataset(split="train", cache_dir=args.cache_dir)
    elif args.dataset == "metamathqa":
        train_ds = MetaMathQADataset(split="train", cache_dir=args.cache_dir)
    elif args.dataset == "amdeepseek":
        train_ds = AMDeepSeekDataset(
            split="train",
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            skip_samples=args.skip_samples
        )
    else:  # mixture
        ds_a = Math500Dataset(split="train", cache_dir=args.cache_dir)
        ds_b = MetaMathQADataset(split="train", cache_dir=args.cache_dir)
        train_ds = MixtureDataset.create(ds_a, ds_b, args.mix_ratio, seed=args.seed)

    print(f"Dataset size: {len(train_ds)} samples")

    # Load models
    print("Loading teacher model...")
    teacher = load_teacher(
        args.teacher,
        device,
        dtype,
        quantize_8bit=args.teacher_quantize_8bit,
        cache_dir=args.cache_dir,
    )
    print("Loading student model...")
    student = load_student(
        args.student,
        device,
        dtype,
        cache_dir=args.cache_dir,
    )

    # Create config object
    class Config:
        pass

    config = Config()
    config.device = device
    config.dtype = dtype
    config.teacher_align_layer = args.teacher_align_layer
    config.student_align_layer = args.student_align_layer
    config.target_norm = args.target_norm
    config.reason_active_threshold = args.reason_active_threshold
    config.reason_presence_weight = args.reason_presence_weight
    config.reason_rank_weight = args.reason_rank_weight
    config.batch_size = args.batch_size
    config.accum_steps = args.accum_steps
    config.epochs = args.epochs
    config.lr = args.lr
    config.min_lr = args.min_lr
    config.warmup_ratio = args.warmup_ratio
    config.weight_decay = args.weight_decay
    config.max_grad_norm = args.max_grad_norm
    config.adam_betas = tuple(args.adam_betas)
    config.scheduler_type = args.scheduler_type
    config.restart_interval_steps = args.restart_interval_steps
    config.restart_mult = args.restart_mult
    config.temperature = args.temperature
    config.alpha_kd = args.alpha_kd
    config.alpha_align = args.alpha_align
    config.beta_ce = args.beta_ce
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir or args.checkpoint_dir
    config.loss_log_entries_per_epoch = args.loss_log_entries_per_epoch
    config.max_length = args.max_length
    config.save_every_n_steps = args.save_every_n_steps

    print(f"Training config: LR={config.lr}, min_lr={config.min_lr}, max_grad_norm={config.max_grad_norm}")
    print(f"Loss weights: alpha_kd={config.alpha_kd}, alpha_align={config.alpha_align}, beta_ce={config.beta_ce}")

    # Instantiate the appropriate trainer
    print(f"Instantiating trainer for method: {args.method}")
    if args.method == "logit_kd":
        trainer = LogitKDTrainer(student, teacher, tokenizer, config)
    elif args.method == "fitnets":
        trainer = FitNetsTrainer(student, teacher, tokenizer, config)
    elif args.method == "hard_label":
        trainer = HardLabelTrainer(student, teacher, tokenizer, config)
    elif args.method == "reasondistill":
        print("Loading SAE...")
        sae = load_sae(args.sae_checkpoint, device, dtype)
        print("Loading reasoning neurons...")
        reasoning_neurons = load_reasoning_neurons(
            args.reason_score_path,
            args.reasoning_neuron_count,
            device,
        )
        print(f"Using {reasoning_neurons.numel()} reasoning features")
        aligner = ReasoningFeatureHead(
            student_dim=student.config.hidden_size,
            num_reasoning_features=reasoning_neurons.numel(),
        ).to(device)
        trainer = ReasonDistillTrainer(
            student, teacher, tokenizer, sae, reasoning_neurons, aligner, config
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Optionally initialize module weights from a previous checkpoint.
    # Now supports HF references (hf://...) for directories.
    if args.student_checkpoint:
        print(f"📦 Loading student checkpoint from: {args.student_checkpoint}")
        try:
            checkpoint_model, aligner_state = load_student_checkpoint(
                args.student_checkpoint, device, dtype, cache_dir=args.cache_dir
            )
            
            # Overwrite the trainer's student with the checkpoint model
            trainer.student = checkpoint_model
            print("✅ Student model weights loaded from checkpoint.")
            
            # If aligner state exists, load it into the trainer's aligner
            if aligner_state is not None and trainer.aligner is not None:
                trainer.aligner.load_state_dict(aligner_state)
                print("✅ Aligner weights loaded from checkpoint and applied.")
            elif trainer.aligner is not None:
                print("ℹ️  No aligner.pt found in checkpoint; keeping freshly initialized aligner.")
            else:
                print("ℹ️  Trainer has no aligner; skipping aligner loading.")
            
            # Rebuild optimizer with the loaded weights.
            # Note: This resets Adam momentum. That's expected when resuming
            # with a different dataset slice (new samples = new gradient landscape).
            params = list(trainer.student.parameters())
            if trainer.aligner is not None:
                params += list(trainer.aligner.parameters())
            trainer.optimizer = torch.optim.AdamW(
                params,
                lr=config.lr,
                betas=config.adam_betas,
                weight_decay=config.weight_decay
            )
            print("✅ Optimizer rebuilt with loaded weights (momentum reset).")
            print("⚠️  Note: Optimizer momentum was reset. Initial loss may spike.")
            print("   This is expected and should recover within ~50 steps.")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("⚠️  Starting from scratch with fresh model weights.")
            import traceback
            traceback.print_exc()
    else:
        print("ℹ️  No student_checkpoint specified; starting from scratch.")

    print("Starting training...")
    trainer.train(train_ds)
    print("Training complete!")


if __name__ == "__main__":
    main()