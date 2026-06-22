import argparse
import torch
from pathlib import Path
from data_loaders.math500 import Math500Dataset
from data_loaders.metamathqa import MetaMathQADataset
from data_loaders.mixture import MixtureDataset
from models.student_loader import load_teacher, load_student, load_tokenizer
from models.sae_loader import load_sae, load_reasoning_neurons
from models.aligner import ReasoningNeuronAligner
from training import (
    LogitKDTrainer,
    FitNetsTrainer,
    HardLabelTrainer,
    ReasonDistillTrainer,
)
from utils.seed import set_seed
import os

def main():
    parser = argparse.ArgumentParser()
    # Method selection
    parser.add_argument("--method", type=str,
                        choices=["logit_kd", "fitnets", "hard_label", "reasondistill"],
                        default="reasondistill",
                        help="Distillation method to use")
    # Data
    parser.add_argument("--dataset", choices=["math500", "metamathqa", "mixture"], default="mixture")
    parser.add_argument("--mix_ratio", type=float, default=0.5, help="ratio of math500 in mixture")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.95])
    # KD hyperparameters (common)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha_kd", type=float, default=0.5)
    parser.add_argument("--alpha_align", type=float, default=0.5)  # for FitNets / ReasonDistill
    parser.add_argument("--beta_ce", type=float, default=0.5)
    # Method-specific paths (only needed for ReasonDistill)
    parser.add_argument("--sae_checkpoint", type=str, default=None, help="Required for reasondistill")
    parser.add_argument("--reason_score_path", type=str, default=None, help="Required for reasondistill")
    # Common paths
    parser.add_argument("--student_checkpoint", type=str, default=None, help="Resume student from checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load tokenizer
    tokenizer = load_tokenizer("meta-llama/Llama-3.2-1B", cache_dir=args.cache_dir)

    # Load datasets
    if args.dataset == "math500":
        train_ds = Math500Dataset(split="train", cache_dir=args.cache_dir)
    elif args.dataset == "metamathqa":
        train_ds = MetaMathQADataset(split="train", cache_dir=args.cache_dir)
    else:  # mixture
        ds_a = Math500Dataset(split="train", cache_dir=args.cache_dir)
        ds_b = MetaMathQADataset(split="train", cache_dir=args.cache_dir)
        train_ds = MixtureDataset.create(ds_a, ds_b, args.mix_ratio, seed=args.seed)

    # Load models
    teacher = load_teacher(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device, dtype, quantize_8bit=True, cache_dir=args.cache_dir
    )
    student = load_student(
        "meta-llama/Llama-3.2-1B",
        device, dtype, cache_dir=args.cache_dir
    )
    if args.student_checkpoint:
        student = load_student(args.student_checkpoint, device, dtype, cache_dir=None)

    # Create config object
    class Config:
        pass
    config = Config()
    config.device = device
    config.dtype = dtype
    config.teacher_align_layer = 19
    config.student_align_layer = 10
    config.target_norm = 64.0   # for ReasonDistill
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
    config.max_length = args.max_length

    # Instantiate the appropriate trainer
    if args.method == "logit_kd":
        trainer = LogitKDTrainer(student, teacher, tokenizer, config)
    elif args.method == "fitnets":
        trainer = FitNetsTrainer(student, teacher, tokenizer, config)
    elif args.method == "hard_label":
        trainer = HardLabelTrainer(student, teacher, tokenizer, config)
    elif args.method == "reasondistill":
        if args.sae_checkpoint is None or args.reason_score_path is None:
            raise ValueError("--sae_checkpoint and --reason_score_path required for reasondistill")
        sae = load_sae(args.sae_checkpoint, device, dtype)
        reasoning_neurons = load_reasoning_neurons(args.reason_score_path, 196, device)
        aligner = ReasoningNeuronAligner(2048, 4096).to(device)
        trainer = ReasonDistillTrainer(
            student, teacher, tokenizer, sae, reasoning_neurons, aligner, config
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Optionally load checkpoint if resume is requested (via student_checkpoint)
    if args.student_checkpoint and Path(args.student_checkpoint).exists():
        # Assume the checkpoint directory contains trainer_state.pt
        trainer._load_checkpoint(args.student_checkpoint)

    trainer.train(train_ds)

if __name__ == "__main__":
    main()