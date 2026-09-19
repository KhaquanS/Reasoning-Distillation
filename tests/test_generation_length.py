"""Token-length regression tests; no model downloads or GPU required."""

import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'september-eval-scripts/gsm8k_generation_length.py'
spec = importlib.util.spec_from_file_location('gsm8k_generation_length', SCRIPT)
lengths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lengths)


class GenerationLengthTests(unittest.TestCase):
    def test_final_answer_stopping_excludes_batch_padding(self):
        result = lengths.completion_length([11, 21, 22, 0, 0], 1, 99, 4, final_answer_stop=2)
        self.assertEqual(result['token_ids'], [21, 22])
        self.assertEqual(result['finish_reason'], 'final_answer')
        self.assertFalse(result['hit_token_limit'])

    def test_final_answer_stopping_with_pad_equal_to_eos(self):
        result = lengths.completion_length([11, 21, 22, 99, 99], 1, 99, 4, final_answer_stop=2)
        self.assertEqual(result['generated_tokens'], 2)
        self.assertEqual(result['finish_reason'], 'final_answer')

    def test_final_answer_on_last_budget_step_is_not_censored(self):
        result = lengths.completion_length([11, 21, 22], 1, 99, 2, final_answer_stop=2)
        self.assertEqual(result['finish_reason'], 'final_answer')
        self.assertFalse(result['hit_token_limit'])

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


class FinalAnswerBoundaryTests(unittest.TestCase):
    def test_thinking_answer_does_not_stop_generation(self):
        self.assertIsNone(lengths.final_answer_boundary('<answer>3</answer>', thinking=True))
        text = '<answer>2</answer></think><answer>3</answer>'
        self.assertEqual(lengths.final_answer_boundary(text, thinking=True), (len(text), 'answer_tag'))

    def test_first_answer_wins_without_reasoning(self):
        first = '<answer>3</answer>'
        self.assertEqual(lengths.final_answer_boundary(first + first, False), (len(first), 'answer_tag'))

    def test_think_reopening_is_respected(self):
        text = '</think><think><answer>2</answer>'
        self.assertIsNone(lengths.final_answer_boundary(text, True))
        text += '</think><answer>3</answer>'
        self.assertEqual(lengths.final_answer_boundary(text, True), (len(text), 'answer_tag'))

    def test_plain_boxed_answer_and_nested_latex(self):
        for text in [r'\boxed{3}', r'\boxed{\frac{1}{2}}', r'\boxed{\{3\}}']:
            self.assertEqual(lengths.final_answer_boundary(text, False), (len(text), 'boxed'))

    def test_box_in_answer_block_waits_for_closing_tag(self):
        text = r'<answer>\boxed{3}'
        self.assertIsNone(lengths.final_answer_boundary(text, False))
        text += '</answer>'
        self.assertEqual(lengths.final_answer_boundary(text, False), (len(text), 'answer_tag'))

    def test_incomplete_empty_and_placeholder_answers_do_not_stop(self):
        for text in [r'\boxed{3', r'\boxed{\frac{1}{2}', r'\boxed{}', r'\boxed{answer}',
                     '<answer></answer>', '<answer>3</ans', '</answer>']:
            self.assertIsNone(lengths.final_answer_boundary(text, False), text)

    def test_no_think_mode_still_ignores_generated_reasoning(self):
        text = '<think>' + r'\boxed{2}'
        self.assertIsNone(lengths.final_answer_boundary(text, False))
        text += '</think>' + r'\boxed{3}'
        self.assertEqual(lengths.final_answer_boundary(text, False), (len(text), 'boxed'))

    def test_callback_stops_rows_independently_and_ignores_prompt(self):
        class Vector:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values

        class Matrix:
            device = 'cpu'
            def __init__(self, rows): self.rows = rows
            def __getitem__(self, key):
                row, col = key
                if isinstance(row, slice): return Vector([r[col] for r in self.rows[row]])
                return Vector(self.rows[row][col])

        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: ''.join(chr(t) for t in tokens))
        prompt = list(map(ord, '<answer>prompt example</answer>'))
        stopper = lengths.FinalAnswerStopper(tokenizer, len(prompt), 2, False, 0)
        first = list(map(ord, '<answer>3</answer>'))
        second = list(map(ord, 'still reasoning >'))
        fake_torch = SimpleNamespace(tensor=lambda values, **kwargs: values, bool=bool)
        with patch.dict('sys.modules', {'torch': fake_torch}):
            done = stopper(Matrix([prompt + first, prompt + second]))
            self.assertEqual(done, [True, False])
            second += list(map(ord, '<answer>4</answer>'))
            done = stopper(Matrix([prompt + first + [0], prompt + second]))
            self.assertEqual(done, [True, True])
        self.assertEqual(stopper.stop_lengths, [len(first), len(second)])

    def test_marker_can_span_tokens_and_share_token_with_suffix(self):
        pieces = {1: '<answer>', 2: '3</ans', 3: 'wer>\nNext'}
        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: ''.join(pieces[t] for t in tokens))
        stopper = lengths.FinalAnswerStopper(tokenizer, 0, 1, False, 99)
        for end in range(1, 4):
            if stopper.should_check(0, end): stopper.observe(0, list(range(1, end + 1)))
            if end < 3: self.assertIsNone(stopper.stop_lengths[0])
        self.assertEqual(stopper.stop_lengths[0], 3)

    def test_eos_does_not_count_as_answer_stop(self):
        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: '')
        stopper = lengths.FinalAnswerStopper(tokenizer, 0, 1, False, 99)
        self.assertFalse(stopper.should_check(0, 99))
        self.assertTrue(stopper.eos_seen[0])
        self.assertIsNone(stopper.stop_lengths[0])


if __name__ == '__main__':
    unittest.main()
