"""
compute_reason_score.py

Compute ReasonScore (Galichin et al., 2025) for SAE features using the AM-DeepSeek dataset.
Saves reasonscore.pt with sorted feature indices for ReasonDistill.

Usage:
    python scripts/compute_reason_score.py \
        --sae_checkpoint ./checkpoints/qwen_sae_layer14_16x/sae.pt \
        --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --layer 14 \
        --output_dir ./checkpoints/qwen_sae_layer14_16x \
        --cache_dir ./cache \
        --max_samples 10000 \
        --batch_size 4 \
        --max_length 2048
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.amdeepseek import AMDeepSeekDataset
from models.student_loader import load_teacher, load_tokenizer
from models.sae_loader import load_sae
from utils.seed import set_seed


# Reasoning vocabulary from the paper (Appendix A.1)
REASONING_VOCABULARY = [
    "alternatively", " alternatively", "Alternatively", " Alternatively",
    "hmm", " hmm", "Hmm", " Hmm",
    "maybe", " maybe", "Maybe", " Maybe",
    "wait", " wait", "Wait", " Wait",
    "perhaps", " perhaps", "Perhaps", " Perhaps",
    "let me", " let me", "Let me", " Let me",
    "therefore", " therefore", "Therefore", " Therefore",
    "however", " however", "However", " However",
    "But", "Another",
]


def tokenize_reasoning_vocabulary(tokenizer):
    """
    Tokenize the reasoning vocabulary and return a set of token IDs to match.
    Some reasoning phrases may be multi-token; we handle both single-token
    and multi-token matches.
    """
    token_ids = []
    for word in REASONING_VOCABULARY:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            token_ids.append(torch.tensor(ids, dtype=torch.long))
    return token_ids


def build_reasoning_masks(input_ids, reasoning_token_ids):
    """
    Vectorized reasoning-token matching.

    The previous implementation did a Python triple loop
    (patterns x batch x sequence position) calling torch.equal() per
    position, which forces a GPU sync on every single call -- with 34
    patterns, a batch of 10, and sequences up to 2048 tokens that's up to
    ~700K syncing calls per batch. This version uses input_ids.unfold() to
    build all sliding windows at once and compares them in a single batched
    op per pattern, then uses advanced indexing (no host sync, no per-
    position Python loop) to paint the resulting masks.

    Returns:
        windowed_mask: (B, T) bool tensor, True for token positions inside
            a reasoning phrase's window (2 preceding, 3 subsequent tokens,
            per the paper).
        per_word_masks: list of (B, T) bool tensors, one per pattern in
            reasoning_token_ids, True only at the pattern's exact token
            positions (no window expansion) -- used for the per-word
            entropy statistics.
    """
    B, T = input_ids.shape
    device = input_ids.device
    windowed_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    per_word_masks = []

    for ids in reasoning_token_ids:
        ids_len = ids.numel()
        word_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        if ids_len <= T:
            # (B, T - ids_len + 1, ids_len): every sliding window of length
            # ids_len, compared against the pattern in one batched op.
            windows = input_ids.unfold(dimension=1, size=ids_len, step=1)
            matches = (windows == ids).all(dim=-1)  # (B, T - ids_len + 1)

            if matches.any():
                b_idx, start_idx = matches.nonzero(as_tuple=True)

                # Exact pattern positions: [start, start + ids_len)
                offsets = torch.arange(ids_len, device=device)
                exact_b = b_idx.unsqueeze(1).expand(-1, ids_len).reshape(-1)
                exact_pos = (start_idx.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
                word_mask[exact_b, exact_pos] = True

                # Windowed positions: [start - 2, start + ids_len + 3),
                # clamped to sequence bounds.
                win_offsets = torch.arange(-2, ids_len + 3, device=device)
                win_pos = start_idx.unsqueeze(1) + win_offsets.unsqueeze(0)  # (matches, window_width)
                valid = (win_pos >= 0) & (win_pos < T)
                win_pos_clamped = win_pos.clamp(0, T - 1)
                win_b = b_idx.unsqueeze(1).expand(-1, win_offsets.numel())
                windowed_mask[win_b[valid], win_pos_clamped[valid]] = True

        per_word_masks.append(word_mask)

    return windowed_mask, per_word_masks


def compute_reason_score(
    sae,
    teacher,
    tokenizer,
    dataset,
    layer,
    device,
    batch_size=4,
    max_length=2048,
    alpha=1.0,
    epsilon=1e-12,
    max_samples=None,
):
    """
    Compute ReasonScore for all SAE features.

    Implements Equation 5 from the paper:
        ReasonScore_i = (μ(i, D_R^W) / Σ_j μ(j, D_R^W)) * H_i^α
                      - (μ(i, D_{-R}^W) / Σ_j μ(j, D_{-R}^W))

    where H_i is the entropy penalty (Equation 4).
    """
    sae.eval()
    teacher.eval()

    # Tokenize reasoning vocabulary. Move to `device` once here rather than
    # calling .to(device) inside the matching loop every batch/pattern.
    reasoning_token_ids = [ids.to(device) for ids in tokenize_reasoning_vocabulary(tokenizer)]
    print(f"Reasoning vocabulary tokenized into {len(reasoning_token_ids)} patterns")

    # Accumulators
    # For reasoning tokens (windowed)
    sum_pos = torch.zeros(sae.latent_dim, device=device, dtype=torch.float64)
    count_pos = 0
    # For non-reasoning tokens
    sum_neg = torch.zeros(sae.latent_dim, device=device, dtype=torch.float64)
    count_neg = 0
    # Per-word means for entropy (μ(i, D_{r_j}^W))
    per_word_sums = defaultdict(lambda: torch.zeros(sae.latent_dim, device=device, dtype=torch.float64))
    per_word_counts = defaultdict(int)

    # Dataloader
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    total_samples = 0
    pbar = tqdm(loader, desc="Computing ReasonScore")

    # Fixed once up front rather than re-read every batch. sae.encoder.weight
    # never changes dtype mid-run, and pinning it here makes it obvious what
    # dtype every downstream tensor in this loop is supposed to be.
    target_dtype = sae.encoder.weight.dtype

    with torch.no_grad():
        for batch in pbar:
            if max_samples and total_samples >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            B, T = input_ids.shape

            # Get teacher hidden states at the specified layer
            outputs = teacher(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            # hidden_states[0] = embeddings, hidden_states[1] = layer 0, etc.
            # So layer L corresponds to hidden_states[L + 1]
            #
            # NOTE: with hybrid/linear-attention teachers (e.g. Qwen3.5's
            # Gated DeltaNet layers) and/or a checkpoint that carries a
            # baked-in quantization_config, `transformers` frequently
            # upcasts intermediate hidden states to fp32 internally for
            # numerical stability (this is especially likely when the fast
            # fla/causal-conv1d kernels aren't installed and it falls back
            # to the pure-torch path, as seen in this run's log). So we
            # cannot assume hidden_states already matches the model's
            # nominal dtype -- cast explicitly right here.
            activations = hidden_states[layer + 1].to(target_dtype)  # (B, T, D)

            # Flatten to (B*T, D)
            flat_acts = activations.reshape(-1, activations.shape[-1])

            # Normalize to target norm (as done in training). Do the divide
            # in fp32 for numerical stability (bf16 has ~3 decimal digits of
            # precision, so norms and their reciprocals can be noisy), then
            # cast the *result* back to target_dtype explicitly -- don't
            # rely on Python-float/tensor type promotion to keep this bf16.
            target_norm = 64.0
            flat_norm = flat_acts.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = (target_norm / flat_norm).to(target_dtype)
            flat_acts_normed = flat_acts * scale

            # Encode with SAE (sae.encode() also defensively casts, but we
            # keep this explicit so dtype bugs show up here, not three
            # frames deep inside sae.py)
            assert flat_acts_normed.dtype == target_dtype, (
                f"expected {target_dtype}, got {flat_acts_normed.dtype}"
            )
            latent = sae.encode(flat_acts_normed)  # (B*T, latent_dim)

            # Reshape back to (B, T, latent_dim)
            latent = latent.reshape(B, T, -1)

            # Build reasoning masks (windowed union + exact per-word) in a
            # single vectorized pass -- see build_reasoning_masks() docstring.
            reasoning_mask, per_word_masks = build_reasoning_masks(input_ids, reasoning_token_ids)

            # Separate reasoning vs non-reasoning tokens
            pos_mask = reasoning_mask & attn_mask.bool()
            neg_mask = ~reasoning_mask & attn_mask.bool()

            pos_flat = pos_mask.reshape(-1)
            neg_flat = neg_mask.reshape(-1)

            # Accumulate means
            if pos_flat.any():
                pos_feats = latent.reshape(-1, sae.latent_dim)[pos_flat]
                sum_pos += pos_feats.sum(dim=0).double()
                count_pos += pos_feats.size(0)

            if neg_flat.any():
                neg_feats = latent.reshape(-1, sae.latent_dim)[neg_flat]
                sum_neg += neg_feats.sum(dim=0).double()
                count_neg += neg_feats.size(0)

            # Per-word means for entropy, reusing the exact-position masks
            # build_reasoning_masks() already computed above (previously
            # this re-ran the same O(B*T) matching loop from scratch here).
            for word_idx, word_mask in enumerate(per_word_masks):
                word_mask = word_mask & attn_mask.bool()
                if word_mask.any():
                    word_flat = word_mask.reshape(-1)
                    word_feats = latent.reshape(-1, sae.latent_dim)[word_flat]
                    per_word_sums[word_idx] += word_feats.sum(dim=0).double()
                    per_word_counts[word_idx] += word_feats.size(0)

            total_samples += B
            pbar.set_postfix({"samples": total_samples, "pos": count_pos, "neg": count_neg})

    # Compute means
    mu_pos = sum_pos / max(1, count_pos) if count_pos > 0 else torch.zeros(sae.latent_dim, device=device)
    mu_neg = sum_neg / max(1, count_neg) if count_neg > 0 else torch.zeros(sae.latent_dim, device=device)

    # Per-word means. Iterate over the full range of pattern indices, not
    # just `for idx in per_word_sums` -- per_word_sums/per_word_counts are
    # defaultdicts that only gain an entry for a pattern once it's actually
    # matched at least once. A pattern that never occurs anywhere in the
    # dataset (entirely plausible over only 10k samples for something like
    # a rare capitalization/spacing variant) would otherwise be silently
    # missing from mu_per_word and crash the word_means stack below with a
    # KeyError. Zero occurrences correctly means a zero mean vector here.
    mu_per_word = {}
    for idx in range(len(reasoning_token_ids)):
        if per_word_counts.get(idx, 0) > 0:
            mu_per_word[idx] = per_word_sums[idx] / per_word_counts[idx]
        else:
            mu_per_word[idx] = torch.zeros(sae.latent_dim, device=device)

    # Compute entropy penalty H_i (Equation 4)
    # H_i = -1/log(|R|) * Σ_j p_i(r_j) * log(p_i(r_j))
    num_words = len(reasoning_token_ids)
    if num_words > 1:
        # Build matrix of per-word means: (num_words, latent_dim)
        word_means = torch.stack([mu_per_word[i] for i in range(num_words)], dim=0)  # (num_words, latent_dim)
        # Normalize to probabilities: p_i(r_j) = μ(i, D_{r_j}^W) / Σ_k μ(i, D_{r_k}^W)
        probs = word_means / (word_means.sum(dim=0, keepdim=True).clamp(min=epsilon))  # (num_words, latent_dim)
        # Entropy: H_i = -1/log(num_words) * Σ_j p_i(r_j) * log(p_i(r_j))
        log_probs = torch.log(probs.clamp(min=epsilon))
        entropy = - (probs * log_probs).sum(dim=0) / math.log(num_words)  # (latent_dim,)
    else:
        entropy = torch.ones(sae.latent_dim, device=device)

    # Compute ReasonScore (Equation 5)
    # First term: (μ(i, D_R^W) / Σ_j μ(j, D_R^W)) * H_i^α
    sum_pos_all = mu_pos.sum().clamp(min=epsilon)
    term1 = (mu_pos / sum_pos_all) * (entropy ** alpha)

    # Second term: μ(i, D_{-R}^W) / Σ_j μ(j, D_{-R}^W)
    sum_neg_all = mu_neg.sum().clamp(min=epsilon)
    term2 = mu_neg / sum_neg_all

    reason_scores = term1 - term2  # (latent_dim,)

    return reason_scores.cpu()


def main():
    parser = argparse.ArgumentParser(description="Compute ReasonScore for SAE features")
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Path to trained SAE checkpoint (.pt or directory containing sae.pt)")
    parser.add_argument("--teacher", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Teacher model ID")
    parser.add_argument("--layer", type=int, required=True,
                        help="Teacher layer index (0-indexed) used for SAE training")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save reasonscore.pt")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory for models and datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (None = all)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Entropy penalty exponent (default: 1.0)")
    parser.add_argument("--teacher_quantize_8bit", action="store_true", default=False,
                        help="Load teacher in 8-bit quantization (default: False)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Using device: {device}, dtype: {dtype}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.teacher, cache_dir=args.cache_dir)

    # Load dataset (AM-DeepSeek)
    print("Loading AM-DeepSeek dataset...")
    dataset = AMDeepSeekDataset(
        split="train",
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Load SAE
    print(f"Loading SAE from {args.sae_checkpoint}...")
    sae = load_sae(args.sae_checkpoint, device, dtype)
    
    # Force SAE to target dtype
    sae = sae.to(dtype=dtype)
    print(f"SAE dtype: {sae.encoder.weight.dtype}")

    # Load teacher
    print(f"Loading teacher {args.teacher}...")
    teacher = load_teacher(
        args.teacher,
        device,
        dtype,
        quantize_8bit=args.teacher_quantize_8bit,
        cache_dir=args.cache_dir,
    )
    # If not quantized, ensure teacher is in the same dtype
    if not args.teacher_quantize_8bit:
        teacher = teacher.to(dtype)
    print(f"Teacher dtype: {next(teacher.parameters()).dtype}")

    # Compute ReasonScore
    print("Computing ReasonScore...")
    scores = compute_reason_score(
        sae=sae,
        teacher=teacher,
        tokenizer=tokenizer,
        dataset=dataset,
        layer=args.layer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        alpha=args.alpha,
        max_samples=args.max_samples,
    )

    # Sort features by ReasonScore (descending)
    sorted_indices = torch.argsort(scores, descending=True)

    # Save in the format expected by load_reasoning_neurons()
    # The loader expects a dict with "sorted_indices" key
    output_path = Path(args.output_dir) / "reasonscore.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sorted_indices": sorted_indices}, output_path)
    print(f"Saved ReasonScore to {output_path}")
    print(f"Top 10 reasoning features: {sorted_indices[:10].tolist()}")
    print(f"Top 10 scores: {scores[sorted_indices[:10]].tolist()}")


if __name__ == "__main__":
    main()