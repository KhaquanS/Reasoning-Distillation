"""Token-length regression tests; no model downloads or GPU required."""

import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'september-eval-scripts/gsm8k_generation_length.py'
spec = importlib.util.spec_from_file_location('gsm8k_generation_length', SCRIPT)
lengths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lengths)


class GenerationLengthTests(unittest.TestCase):
    def test_left_padded_prompt_and_trailing_padding_are_not_counted(self):
        # Prompt width is 5 even though its attention mask sums to 2.
        result = lengths.completion_length([0, 0, 0, 11, 12, 21, 22, 99, 0, 0], 5, 99, 5)
        self.assertEqual(result['generated_tokens'], 2)
        self.assertEqual(result['token_ids'], [21, 22])
        self.assertEqual(result['generation_steps_including_eos'], 3)
        self.assertEqual(result['finish_reason'], 'eos')

    def test_pad_equals_eos(self):
        result = lengths.completion_length([99, 11, 21, 99, 99, 99], 2, 99, 4)
        self.assertEqual(result['generated_tokens'], 1)

    def test_multiple_eos_ids(self):
        result = lengths.completion_length([11, 21, 98, 99], 1, [98, 99], 3)
        self.assertEqual(result['token_ids'], [21])

    def test_generated_special_tokens_are_preserved(self):
        result = lengths.completion_length([11, 777, 21, 778, 99], 1, 99, 4)
        self.assertEqual(result['generated_tokens'], 3)

    def test_budget_limit_and_eos_at_final_step_are_distinguished(self):
        capped = lengths.completion_length([11, 21, 22, 23], 1, 99, 3)
        ended = lengths.completion_length([11, 21, 22, 99], 1, 99, 3)
        self.assertTrue(capped['hit_token_limit'])
        self.assertEqual(capped['generated_tokens'], 3)
        self.assertFalse(ended['hit_token_limit'])
        self.assertEqual(ended['generated_tokens'], 2)

    def test_unexplained_early_stop_is_not_called_token_limit(self):
        with self.assertRaises(ValueError):
            lengths.completion_length([11, 21], 1, 99, 10)

    def test_summary_includes_capped_responses(self):
        records = [dict(generated_tokens=2, finish_reason='eos', hit_token_limit=False),
                   dict(generated_tokens=10, finish_reason='length', hit_token_limit=True)]
        result = lengths.summarize(records)
        self.assertEqual(result['mean_generated_tokens'], 6)
        self.assertEqual(result['median_generated_tokens'], 6)
        self.assertEqual(result['token_limit_rate'], .5)
        self.assertEqual(result['mean_tokens_eos_terminated_only'], 2)
        self.assertAlmostEqual(result['p90_generated_tokens'], 9.2)

    def test_all_capped_has_no_completed_only_mean(self):
        result = lengths.summarize([dict(generated_tokens=10, finish_reason='length', hit_token_limit=True)])
        self.assertIsNone(result['mean_tokens_eos_terminated_only'])

    def test_pairing_rejects_mismatched_questions_and_prompts(self):
        base = dict(id='0', question='X', gold_answer='3', prompt='X formatted',
                    generated_tokens=10, hit_token_limit=False)
        distilled = dict(base, generated_tokens=6)
        self.assertEqual(lengths.paired_rows([base], [distilled])[0]['distilled_minus_base_tokens'], -4)
        for changed in [dict(distilled, question='Y'), dict(distilled, prompt='Y formatted')]:
            with self.assertRaises(ValueError):
                lengths.paired_rows([base], [changed])


if __name__ == '__main__':
    unittest.main()
