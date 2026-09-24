"""Math protocol and pass@k integration tests; no model or dataset downloads."""

import importlib.util
import json
import tempfile
import unittest
from math import comb
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HAS_EVAL_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "datasets", "yaml", "math_verify")
)

if HAS_EVAL_DEPS:
    import yaml

    from custom_eval.benchmarks import aime25, math500
    from custom_eval.benchmarks.base import Benchmark, EvalExample
    from custom_eval.benchmarks.loaders import load_required_dataset
    from custom_eval.config import EvalConfig, ModelSpec, load_config
    from custom_eval.generation import GeneratedCandidate, extract_final_response
    from custom_eval.math_scoring import aime_exact_match, extract_boxed_answer, math_equivalent
    from custom_eval.metrics import estimate_pass_at_k
    from custom_eval.prompts.templates import build_prompt
    from custom_eval.runner import run_evaluation

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_EVAL_DEPS, "Evaluation dependencies are required")
class MathProtocolTests(unittest.TestCase):
    def test_nested_and_escaped_braces(self):
        cases = {
            r'answer \boxed{\frac{1}{\sqrt{2}}}': r'\frac{1}{\sqrt{2}}',
            r'\boxed{\{1,2\}}': r'\{1,2\}',
            r'\boxed{1} then \fbox{2}': '2',
            r'\boxed {\left(3,\frac{\pi}{2}\right)}': r'\left(3,\frac{\pi}{2}\right)',
            r'\boxed{1} then \boxed{\frac{2}{': None,
            r'\boxed{}': None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_boxed_answer(text), expected)

    def test_only_final_math_answer_is_scored(self):
        for benchmark in ('math500', 'aime25'):
            with self.subTest(benchmark=benchmark):
                extract = lambda text: extract_final_response(text, benchmark, True, 'qwen')
                self.assertEqual(extract(r'reasoning \boxed{17}'), '')
                self.assertEqual(extract(r'\boxed{17}</think>Final \boxed{42}'), '42')
                self.assertEqual(extract(r'\boxed{17}</think>No final answer'), '')
                self.assertEqual(extract(r'</think>\boxed{1} then \boxed{'), '')
                self.assertEqual(extract(r'<think>\boxed{1}</think><think>\boxed{2}'), '')
                self.assertEqual(extract_final_response(r'Final \boxed{42}', benchmark, False), '42')

    def test_symbolic_equivalence_and_false_numeric_matches(self):
        pairs = [
            (r'\frac{2}{4}', r'\frac{1}{2}', True),
            (r'2\sqrt{2}', r'\sqrt{8}', True),
            ('x+x', '2x', True),
            (r'\{2,1\}', r'\{1,2\}', True),
            (r'\frac{1}{3}', r'\frac{1}{2}', False),
            ('(1,2)', '(1,3)', False),
            ('(0,1)', '[0,1]', False),
            ('2y', '2x', False),
            ('', '0', False),
            ('garbage', '42', False),
            (r'\text{Evelyn}', r'\text{Evelyn}', True),
        ]
        for pred, gold, expected in pairs:
            with self.subTest(pred=pred, gold=gold):
                self.assertEqual(math_equivalent(pred, gold), expected)

    def test_aime_exact_integer(self):
        for pred in ('7', '007', ' 007 '):
            self.assertTrue(aime_exact_match(pred, '7'))
        self.assertTrue(aime_exact_match('000', 0))
        for pred in ('7.0', '7/1', '7 or 8', '-7', '1000', 'answer is 7', ''):
            self.assertFalse(aime_exact_match(pred, '7'))

    def test_prompts_have_one_box_instruction_and_no_placeholder(self):
        for benchmark in ('aime25', 'math500'):
            prompt = build_prompt('QUESTION', benchmark)
            self.assertEqual(prompt.count('QUESTION'), 1)
            self.assertEqual(prompt.count(r'\boxed{}'), 1)
            self.assertNotIn(r'\boxed{answer}', prompt)
        self.assertIn('0 and 999', build_prompt('QUESTION', 'aime25'))


@unittest.skipUnless(HAS_EVAL_DEPS, "Evaluation dependencies are required")
class MathDatasetTests(unittest.TestCase):
    def test_canonical_sources_and_zero_answer(self):
        with patch.object(aime25, 'load_required_dataset', return_value=(
            [{'id': 'aime-0', 'problem': 'Question?', 'answer': 0}], {'path': 'math-ai/aime25'}
        )) as loader:
            b = aime25.load(max_samples=1)
            loader.assert_called_once_with('math-ai/aime25', 30, None, 'test')
            self.assertEqual(b.examples[0].answer, '0')
            self.assertTrue(b.scorer('000', b.examples[0].answer))
        with patch.object(math500, 'load_required_dataset', return_value=(
            [{'unique_id': 'test/algebra/1', 'problem': 'Question?', 'answer': r'\frac{1}{2}',
              'solution': 'Do not use the solution as the gold answer', 'subject': 'Algebra', 'level': 1}],
            {'path': 'HuggingFaceH4/MATH-500'}
        )) as loader:
            b = math500.load()
            loader.assert_called_once_with('HuggingFaceH4/MATH-500', 500, None, 'test')
            self.assertEqual(b.examples[0].answer, r'\frac{1}{2}')
            self.assertEqual(b.examples[0].id, 'test/algebra/1')

    def test_dataset_failures_do_not_produce_toy_scores(self):
        with patch('custom_eval.benchmarks.loaders.load_dataset', side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                load_required_dataset('math-ai/aime25', 30)
        with patch('custom_eval.benchmarks.loaders.load_dataset', return_value=[{}] * 29):
            with self.assertRaisesRegex(ValueError, 'Expected 30'):
                load_required_dataset('math-ai/aime25', 30)
        with self.assertRaisesRegex(ValueError, 'official test split'):
            load_required_dataset('math-ai/aime25', 30, split='train')

    def test_bad_schema_is_rejected(self):
        for rows in (
            [{'id': '0', 'problem': 'q', 'answer': None}],
            [{'id': '0', 'problem': 'q', 'answer': '1000'}],
            [{'id': '0', 'problem': 'q', 'answer': '0'}] * 2,
        ):
            with patch.object(aime25, 'load_required_dataset', return_value=(rows, {})):
                with self.assertRaises(ValueError):
                    aime25.load()


@unittest.skipUnless(HAS_EVAL_DEPS, "Evaluation dependencies are required")
class PassKTests(unittest.TestCase):
    def test_estimator_matches_all_subsets(self):
        for n in range(1, 10):
            for c in range(n + 1):
                values = [estimate_pass_at_k(n, c, k) for k in range(1, n + 1)]
                self.assertEqual(values, sorted(values))
                for k, score in enumerate(values, 1):
                    expected = 1 - (comb(n - c, k) if n - c >= k else 0) / comb(n, k)
                    self.assertAlmostEqual(score, expected)
        self.assertEqual(estimate_pass_at_k(8, 1, 2), 0.25)
        for args in ((2, 1, 4), (8, 9, 2), (8, 1, 0)):
            with self.assertRaises(ValueError):
                estimate_pass_at_k(*args)

    def test_requested_configs(self):
        for filename in ('test_math.yaml', 'sweep_pass_k.yaml'):
            cfg = load_config(ROOT / 'configs' / filename)
            self.assertEqual(len(cfg.models), 3)
            self.assertTrue(all(m.enable_thinking for m in cfg.models))
            self.assertEqual(cfg.models[0].checkpoint, 'Qwen/Qwen3.5-2B')
            self.assertEqual(cfg.models[1].subfolder, 'qwen-logitKD-final/qwen_logit_kd_continue/epoch_1')
            self.assertEqual(cfg.models[2].subfolder, 'qwen_reasondistill_final_continue/epoch_1')
            self.assertEqual(cfg.max_new_tokens, 4096)
            self.assertEqual(cfg.benchmark_options['aime25']['max_new_tokens'], 16384)
        self.assertEqual(cfg.pass_at_k, [2, 4, 8])
        self.assertEqual(cfg.num_samples, 8)
        self.assertEqual(set(cfg.benchmarks), {'aime25', 'math500', 'arc-c', 'gsm8k', 'mmlu', 'hellaswag'})

    def test_invalid_sweeps(self):
        source = yaml.safe_load((ROOT / 'configs/sweep_pass_k.yaml').read_text())
        invalid = [
            {'pass_at_k': []}, {'pass_at_k': [2, 2]}, {'pass_at_k': [0, 4]},
            {'pass_at_k': [2.5, 4]}, {'pass_at_k': True}, {'num_samples': 4},
            {'sample_batch_size': 0}, {'sample_batch_size': 9}, {'temperature': 0},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.yaml'
            for change in invalid:
                cfg = {**source, 'generation': {**source['generation'], **change}}
                path.write_text(yaml.safe_dump(cfg))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_config(path)

    def test_scalar_pass_one_and_real_data_guard(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = EvalConfig(models=[ModelSpec('single', 'test')], benchmarks=['aime25'],
                             output_dir=d, pass_at_k=1, require_real_data=True)
            example = EvalExample('0', 'q', '7', {'source': 'embedded_fallback'})
            model = SimpleNamespace(device='cpu', get_memory_footprint=lambda: 0)
            benchmark = Benchmark('aime25', [example], aime_exact_match)
            with (
                patch('custom_eval.runner.load_model_and_tokenizer', return_value=(model, object())),
                patch('custom_eval.runner.load_benchmark', return_value=benchmark),
                patch('custom_eval.runner.generate_candidates_batch') as generate,
                patch('custom_eval.runner.torch.cuda.is_available', return_value=False),
            ):
                with self.assertRaisesRegex(RuntimeError, 'refusing embedded fallback'):
                    run_evaluation(cfg)
                generate.assert_not_called()
                example.metadata = {'source': {'path': 'math-ai/aime25'}}
                generate.return_value = [[GeneratedCandidate('prompt', 'raw', '7',
                                                              generated_tokens=4, hit_token_limit=False)]]
                run_evaluation(cfg)
            result = json.loads((Path(d) / 'single__aime25.json').read_text())
            self.assertEqual(result['summary']['score'], 1.0)
            self.assertEqual(result['summary']['correct'], 1)
            self.assertEqual(result['summary']['pass_at_k'], 1)
            self.assertEqual(result['summary']['pass_at_k_scores'], {'1': 1.0})

    def test_chunked_runner_scores_shared_samples_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = EvalConfig(
                models=[ModelSpec('model', 'test', enable_thinking=True)],
                benchmarks=['aime25'], output_dir=d, pass_at_k=[2, 4, 8],
                num_samples=8, sample_batch_size=3, batch_size=2,
                benchmark_options={'aime25': {'max_new_tokens': 16384}},
            )
            examples = [EvalExample('0', 'q0', '7'), EvalExample('1', 'q1', '7')]
            benchmark = Benchmark('aime25', examples, aime_exact_match)
            generated = []
            def generate(**kwargs):
                count = kwargs['pass_at_k']
                offset = sum(generated)
                generated.append(count)
                self.assertEqual(kwargs['max_new_tokens'], 16384)
                return [[GeneratedCandidate('prompt', 'raw', '7' if offset+j == 0 and q == 'q0' else '8',
                                            generated_tokens=20, hit_token_limit=False)
                         for j in range(count)] for q in kwargs['questions']]
            model = SimpleNamespace(device='cpu', get_memory_footprint=lambda: 0)
            with (
                patch('custom_eval.runner.load_model_and_tokenizer', return_value=(model, object())),
                patch('custom_eval.runner.load_benchmark', return_value=benchmark),
                patch('custom_eval.runner.generate_candidates_batch', side_effect=generate),
                patch('custom_eval.runner.torch.cuda.is_available', return_value=False),
            ):
                run_evaluation(cfg)
            self.assertEqual(generated, [3, 3, 2])
            result = json.loads((Path(d) / 'model__aime25.json').read_text())
            self.assertEqual(result['summary']['pass_at_k_scores'], {'2': 0.125, '4': 0.25, '8': 0.5})
            self.assertEqual([len(r['candidates']) for r in result['records']], [8, 8])
            self.assertEqual(result['records'][0]['num_correct'], 1)
            self.assertEqual(result['summary']['num_samples_per_question'], 8)
            self.assertTrue((Path(d) / 'run_config.json').exists())
            self.assertTrue((Path(d) / 'index.json').exists())


if __name__ == '__main__':
    unittest.main()
