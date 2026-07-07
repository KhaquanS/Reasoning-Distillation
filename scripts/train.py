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
from models.student_loader import load_teacher, load_student, load_tokenizer
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
}

CONFIG_SECTIONS = {
    "experiment": ["seed"],
    "data": ["dataset", "mix_ratio", "max_samples", "cache_dir"],
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
    ],
    "loss": ["temperature", "alpha_kd", "alpha_align", "beta_ce"],
    "logging": ["checkpoint_dir", "log_dir", "loss_log_entries_per_epoch"],
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
            max_samples=args.max_samples
        )
    else:  # mixture
        ds_a = Math500Dataset(split="train", cache_dir=args.cache_dir)
        ds_b = MetaMathQADataset(split="train", cache_dir=args.cache_dir)
        train_ds = MixtureDataset.create(ds_a, ds_b, args.mix_ratio, seed=args.seed)

    # Load models
    teacher = load_teacher(
        args.teacher,
        device,
        dtype,
        quantize_8bit=args.teacher_quantize_8bit,
        cache_dir=args.cache_dir,
    )
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
    config.temperature = args.temperature
    config.alpha_kd = args.alpha_kd
    config.alpha_align = args.alpha_align
    config.beta_ce = args.beta_ce
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir or args.checkpoint_dir
    config.loss_log_entries_per_epoch = args.loss_log_entries_per_epoch
    config.max_length = args.max_length

    # Instantiate the appropriate trainer
    if args.method == "logit_kd":
        trainer = LogitKDTrainer(student, teacher, tokenizer, config)
    elif args.method == "fitnets":
        trainer = FitNetsTrainer(student, teacher, tokenizer, config)
    elif args.method == "hard_label":
        trainer = HardLabelTrainer(student, teacher, tokenizer, config)
    elif args.method == "reasondistill":
        sae = load_sae(args.sae_checkpoint, device, dtype)
        reasoning_neurons = load_reasoning_neurons(
            args.reason_score_path,
            args.reasoning_neuron_count,
            device,
        )
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
    # Optimizer, scheduler, and epoch state are rebuilt from the current YAML.
    if args.student_checkpoint and Path(args.student_checkpoint).exists():
        trainer.load_module_weights(args.student_checkpoint)

    trainer.train(train_ds)


if __name__ == "__main__":
    main()
