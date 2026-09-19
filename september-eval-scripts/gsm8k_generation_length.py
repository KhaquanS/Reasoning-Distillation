#!/usr/bin/env python3
"""Compare fresh generation lengths on the same first 500 GSM8K test items.

Run from the repository root:
    python september-eval-scripts/gsm8k_generation_length.py --mode think

Only the two September checkpoints are evaluated. No training or grading occurs.
"""

import argparse
import csv
import gc
import hashlib
import json
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def final_answer_boundary(text, thinking):
    """First nonempty answer block or balanced box outside a thinking block.

    This is a format heuristic, not a correctness judgment. An opening <think>
    may already be in the prompt; thinking=True accounts for that case.
    """
    in_think = thinking
    answer_start = None
    for match in re.finditer(r'<(/?)(think|answer)\s*>|\\boxed\s*\{', text, re.IGNORECASE):
        if match.group(2):
            closing, tag = match.group(1), match.group(2).lower()
            if tag == 'think':
                in_think = not bool(closing)
                answer_start = None
            elif not in_think:
                if not closing:
                    answer_start = match.end()
                elif answer_start is not None:
                    content = text[answer_start:match.start()].strip()
                    answer_start = None
                    if content and content.lower() != 'answer':
                        return match.end(), 'answer_tag'
            continue
        if in_think or answer_start is not None:
            continue
        depth = 1
        escaped = False
        for i in range(match.end(), len(text)):
            char = text[i]
            if escaped:
                escaped = False
                continue
            if char == '\\':
                escaped = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    content = text[match.end():i].strip()
                    if content and content.lower() != 'answer':
                        return i + 1, 'boxed'
                    break
    return None


class FinalAnswerStopper:
    """Per-row stopping callback; never halt other unfinished batch members.

    Full text is decoded only when the newest token can close a tag or box.
    Parsing completion-only text prevents matches against prompt examples.
    Store actual stop lengths because HF pads stopped rows until the batch ends.
    """

    def __init__(self, tokenizer, input_width, batch_size, thinking, eos_token_ids):
        self.tokenizer = tokenizer
        self.input_width = input_width
        self.thinking = thinking
        self.eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)
        self.stop_lengths = [None] * batch_size
        self.stop_kinds = [None] * batch_size
        self.eos_seen = [False] * batch_size
        self.token_text = {}

    def should_check(self, row, token):
        if self.stop_lengths[row] is not None or self.eos_seen[row]:
            return False
        if token in self.eos:
            self.eos_seen[row] = True
            return False
        if token not in self.token_text:
            self.token_text[token] = self.tokenizer.decode(
                [token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # These ASCII characters must be present in the token completing a marker.
        return '>' in self.token_text[token] or '}' in self.token_text[token]

    def observe(self, row, generated):
        text = self.tokenizer.decode(generated, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        boundary = final_answer_boundary(text, self.thinking)
        if boundary is not None and self.stop_lengths[row] is None:
            self.stop_lengths[row] = len(generated)
            self.stop_kinds[row] = boundary[1]

    def __call__(self, input_ids, scores=None, **kwargs):
        import torch
        latest = input_ids[:, -1].tolist()
        for row, token in enumerate(latest):
            if self.should_check(row, token):
                self.observe(row, input_ids[row, self.input_width:].tolist())
        return torch.tensor([n is not None for n in self.stop_lengths],
                            device=input_ids.device, dtype=torch.bool)


def completion_length(sequence, input_width, eos_token_ids, max_new_tokens, final_answer_stop=None):
    """Slice the padded prompt; count through first EOS, excluding EOS/padding.

    Internal special tokens (e.g. generated thinking delimiters) count as tokens.
    Do not filter pad IDs globally: pad and EOS can be the same token.
    """
    generated = list(sequence[input_width:])
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)
    first_eos = next((i for i, token in enumerate(generated) if token in eos), None)
    if final_answer_stop is not None:
        if not 0 < final_answer_stop <= len(generated):
            raise ValueError('Invalid final-answer stop length.')
        if first_eos is None or final_answer_stop <= first_eos:
            return {
                'token_ids': generated[:final_answer_stop],
                'generated_tokens': final_answer_stop,
                'generation_steps_including_eos': final_answer_stop,
                'finish_reason': 'final_answer',
                'hit_token_limit': False,
            }
    for position, token in enumerate(generated):
        if token in eos:
            return {
                'token_ids': generated[:position],
                'generated_tokens': position,
                'generation_steps_including_eos': position + 1,
                'finish_reason': 'eos',
                'hit_token_limit': False,
            }
    if len(generated) != max_new_tokens:
        raise ValueError('Generation stopped without EOS before the configured token limit.')
    return {
        'token_ids': generated,
        'generated_tokens': len(generated),
        'generation_steps_including_eos': len(generated),
        'finish_reason': 'length',
        'hit_token_limit': True,
    }


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(records):
    lengths = [r['generated_tokens'] for r in records]
    ended = [r['generated_tokens'] for r in records if r['finish_reason'] == 'eos']
    if not lengths:
        raise ValueError('Cannot summarize an empty run.')
    capped = sum(r['hit_token_limit'] for r in records)
    return {
        'num_samples': len(lengths),
        'mean_generated_tokens': statistics.mean(lengths),
        'median_generated_tokens': statistics.median(lengths),
        'std_generated_tokens': statistics.stdev(lengths) if len(lengths) > 1 else 0.,
        'p90_generated_tokens': percentile(lengths, .90),
        'p95_generated_tokens': percentile(lengths, .95),
        'min_generated_tokens': min(lengths),
        'max_generated_tokens': max(lengths),
        'total_generated_tokens': sum(lengths),
        'num_eos_terminated': len(ended),
        'num_final_answer_terminated': sum(r['finish_reason'] == 'final_answer' for r in records),
        'num_hit_token_limit': capped,
        'token_limit_rate': capped / len(lengths),
        'mean_tokens_eos_terminated_only': statistics.mean(ended) if ended else None,
    }


def paired_rows(base, distilled):
    if len(base) != len(distilled):
        raise ValueError('Models have different numbers of samples.')
    rows = []
    for a, b in zip(base, distilled):
        if (a['id'], a['question'], a['gold_answer']) != (b['id'], b['question'], b['gold_answer']):
            raise ValueError('The model outputs are not paired on identical questions.')
        if a['prompt'] != b['prompt']:
            raise ValueError('Formatted prompts differ between the two model tokenizers.')
        rows.append({
            'id': a['id'], 'base_tokens': a['generated_tokens'],
            'distilled_tokens': b['generated_tokens'],
            'distilled_minus_base_tokens': b['generated_tokens'] - a['generated_tokens'],
            'base_hit_token_limit': a['hit_token_limit'],
            'distilled_hit_token_limit': b['hit_token_limit'],
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['think', 'no_think'], required=True,
                        help='Use the same thinking setting for both models.')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='First N test questions, without shuffling (default: 500).')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stop-at-final-answer', action='store_true',
                        help='Stop each row at a completed answer block or boxed answer outside thinking; changes the length metric.')
    parser.add_argument('--cache-dir', default='./cache')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--output-dir', type=Path,
                        help='New output directory; existing directories are never overwritten.')
    args = parser.parse_args()
    if min(args.num_samples, args.batch_size, args.max_new_tokens) <= 0:
        parser.error('Sample count, batch size and token budget must be positive.')
    if args.temperature < 0:
        parser.error('Temperature must be nonnegative (0 enables greedy decoding).')
    return args


def run_model(spec, questions, args, output_dir):
    import torch
    from transformers import StoppingCriteriaList, set_seed
    from tqdm import tqdm
    from custom_eval.generation import TokenProgressCallback, _format_prompt
    from custom_eval.modeling import load_model_and_tokenizer

    print(f'\nLoading {spec.name}: {spec.checkpoint} / {spec.subfolder or "root"}', flush=True)
    model, tokenizer = load_model_and_tokenizer(spec, args.cache_dir)
    # Reset after loading so initialization does not consume the generation RNG.
    set_seed(args.seed)
    eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        raise ValueError('Tokenizer has no EOS token; cannot measure stopping reliably.')
    tokenizer.padding_side = 'left'
    vocab_fingerprint = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    ).hexdigest()
    info = {
        'checkpoint': spec.checkpoint, 'subfolder': spec.subfolder,
        'resolved_model_commit': getattr(model.config, '_commit_hash', None),
        'tokenizer_vocab_sha256': vocab_fingerprint,
        'eos_token_id': eos_ids, 'pad_token_id': tokenizer.pad_token_id,
    }
    records = []
    started = time.perf_counter()
    with (output_dir / f'{spec.name}.jsonl').open('w') as stream:
        for start in tqdm(range(0, len(questions), args.batch_size), desc=spec.name):
            batch = questions[start:start + args.batch_size]
            prompts = [_format_prompt(q['question'], 'gsm8k', tokenizer, 'qwen', spec.enable_thinking)
                       for q in batch]
            # Match September prompt tokenization, but fail rather than truncate input.
            encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors='pt')
            if encoded['input_ids'].shape[1] > 4096:
                raise ValueError('A prompt exceeds the September 4096-token input limit.')
            input_ids = encoded['input_ids'].to(model.device)
            attention_mask = encoded['attention_mask'].to(model.device)
            input_width = input_ids.shape[1]
            generation_args = dict(
                max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids, num_return_sequences=1, num_beams=1,
                return_dict_in_generate=False,
            )
            if args.temperature > 0:
                generation_args.update(temperature=args.temperature, top_p=.95, top_k=20)
            
            stopping_criteria = []
            stopper = None

            if args.stop_at_final_answer:
                stopper = FinalAnswerStopper(
                    tokenizer,
                    input_width,
                    len(batch),
                    spec.enable_thinking,
                    eos_ids,
                )
                stopping_criteria.append(stopper)

            progress_bar = tqdm(
                total=args.max_new_tokens,
                desc=f"Generating {len(batch)} questions",
                unit="tok",
                leave=False,
            )
            progress_callback = TokenProgressCallback(
                progress_bar,
                args.max_new_tokens,
            )
            stopping_criteria.append(progress_callback)

            generation_args["stopping_criteria"] = StoppingCriteriaList(
                stopping_criteria
            )

            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_args,
                    )
            finally:
                progress_callback.close()
            sequences = outputs.cpu().tolist()
            for row, (q, prompt, sequence) in enumerate(zip(batch, prompts, sequences)):
                stop_length = stopper.stop_lengths[row] if stopper is not None else None
                measured = completion_length(sequence, input_width, eos_ids, args.max_new_tokens, stop_length)
                raw = tokenizer.decode(measured.pop('token_ids'), skip_special_tokens=True)
                record = {
                    **q, 'prompt': prompt, **measured, 'raw_output': raw,
                    'has_closing_think_tag': '</think>' in raw,
                    'answer_tag_count': raw.count('<answer>'),
                    'final_answer_stop_kind': stopper.stop_kinds[row] if stopper is not None else None,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                records.append(record)
            stream.flush()
            del outputs, input_ids, attention_mask, encoded
    info['elapsed_generation_seconds'] = time.perf_counter() - started
    info.update(summarize(records))
    (output_dir / f'{spec.name}-summary.json').write_text(json.dumps(info, indent=2))
    print(f"{spec.name}: mean={info['mean_generated_tokens']:.2f} tokens; "
          f"median={info['median_generated_tokens']:.1f}; "
          f"hit limit={info['num_hit_token_limit']}/{len(records)}", flush=True)
    # Only lightweight results escape this function; GPU model/tensors are released.
    return records, info


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import torch
    from datasets import load_dataset
    from custom_eval.config import ModelSpec

    if not torch.cuda.is_available():
        raise RuntimeError('Use the CUDA-enabled Vast.ai evaluation environment.')
    if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16; pass --dtype float16.')
    # Load once, without the evaluator\'s embedded fallback or shuffling.
    dataset = load_dataset('openai/gsm8k', 'main', split='test', cache_dir=args.cache_dir)
    if len(dataset) < args.num_samples:
        raise ValueError(f'Requested {args.num_samples} questions, dataset has {len(dataset)}.')
    questions = [{'id': str(i), 'question': dataset[i]['question'],
                  'gold_answer': dataset[i]['answer'].split('####')[-1].strip()}
                 for i in range(args.num_samples)]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output_dir = args.output_dir or repo / 'eval_outputs' / f'gsm8k-length-{args.mode}-{stamp}'
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    packages = {}
    for package in ['torch', 'transformers', 'datasets', 'causal-conv1d', 'flash-linear-attention', 'triton']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    metadata = {
        'settings': settings, 'gpu': torch.cuda.get_device_name(0), 'torch_cuda': torch.version.cuda,
        'python': platform.python_version(), 'packages': packages,
        'dataset': {'path': 'openai/gsm8k', 'name': 'main', 'split': 'test',
                    'fingerprint': getattr(dataset, '_fingerprint', None),
                    'selection': f'First {args.num_samples} rows, no shuffle'},
        'questions_sha256': hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest(),
        'length_definition': 'Generated token IDs before the first EOS, excluding EOS and trailing batch padding; includes reasoning, answer and other generated special tokens.',
        'stopping_policy': ('First completed answer block or balanced boxed answer outside thinking; count through the token completing the marker.'
                            if args.stop_at_final_answer else 'Natural EOS or token budget.'),
        'censoring_note': 'Lengths are measured under max_new_tokens. Budget-limited responses may have continued; EOS-only means describe a selected subset.',
    }
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (output_dir / 'questions.json').write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    common = dict(enable_thinking=args.mode == 'think', dtype=args.dtype, device_map='auto', model_type='qwen')
    specs = [
        ModelSpec(name='base', checkpoint='Qwen/Qwen3.5-2B', **common),
        ModelSpec(name='distilled', checkpoint='Khaquan/qwen-khaquanS-distillations',
                  subfolder='qwen_reasondistill_final_continue/epoch_1',
                  strip_language_model_prefix=True, **common),
    ]
    all_records, summaries = {}, {}
    for spec in specs:
        all_records[spec.name], summaries[spec.name] = run_model(spec, questions, args, output_dir)
        gc.collect()
        torch.cuda.empty_cache()
    if summaries['base']['tokenizer_vocab_sha256'] != summaries['distilled']['tokenizer_vocab_sha256']:
        raise ValueError('Tokenizer vocabularies differ; native token counts are not directly comparable.')
    paired = paired_rows(all_records['base'], all_records['distilled'])
    with (output_dir / 'paired_lengths.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    base_mean = summaries['base']['mean_generated_tokens']
    distilled_mean = summaries['distilled']['mean_generated_tokens']
    comparison = {
        'mean_distilled_minus_base_tokens': statistics.mean(r['distilled_minus_base_tokens'] for r in paired),
        'distilled_to_base_mean_ratio': distilled_mean / base_mean if base_mean else None,
        'distilled_shorter_count': sum(r['distilled_minus_base_tokens'] < 0 for r in paired),
        'equal_length_count': sum(r['distilled_minus_base_tokens'] == 0 for r in paired),
        'distilled_longer_count': sum(r['distilled_minus_base_tokens'] > 0 for r in paired),
    }
    (output_dir / 'summary.json').write_text(json.dumps({'metadata': metadata, 'models': summaries, 'comparison': comparison}, indent=2))
    print(f'\nMean length: Base {base_mean:.2f} | ReasonDistill {distilled_mean:.2f} tokens')
    print(f"Paired mean difference (RD - Base): {comparison['mean_distilled_minus_base_tokens']:+.2f} tokens")
    if any(s['num_hit_token_limit'] for s in summaries.values()):
        print('Some responses hit the token limit. These are capped lengths, not unconstrained generation lengths.')
    print(f'Results: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
