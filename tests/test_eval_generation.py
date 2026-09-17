"""Offline regression checks: python3 -m unittest discover -s tests -v."""
import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import the production functions without loading models or optional dependencies.
with patch.dict(sys.modules):
    for name in ('custom_eval', 'custom_eval.prompts'):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    torch = types.ModuleType('torch')
    torch.LongTensor = torch.FloatTensor = object
    torch.no_grad = nullcontext
    sys.modules['torch'] = torch
    transformers = types.ModuleType('transformers')
    for name in ('PreTrainedModel', 'PreTrainedTokenizer', 'StoppingCriteria'):
        setattr(transformers, name, object)
    transformers.StoppingCriteriaList = list
    sys.modules['transformers'] = transformers
    tqdm = types.ModuleType('tqdm')
    tqdm.tqdm = object
    sys.modules['tqdm'] = tqdm
    formatter = load_module('custom_eval.prompts.qwen_formatter',
                            'custom_eval/prompts/qwen_formatter.py')
    templates = load_module('custom_eval.prompts.templates',
                            'custom_eval/prompts/templates.py')
    generation = load_module('custom_eval.generation', 'custom_eval/generation.py')


class Tensor(list):
    @property
    def shape(self):
        return (len(self), len(self[0]))

    def to(self, device):
        return self

    def repeat_interleave(self, count, dim):
        return Tensor([row[:] for row in self for _ in range(count)])


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        suffix = '<think>\n' if kwargs.get('enable_thinking') else '<think>\n\n</think>\n'
        return messages[-1]['content'] + suffix

    def __call__(self, prompts, **kwargs):
        # Unequal prompt lengths in a left-padded batch.
        rows = [[ord(c) for c in prompt] for prompt in prompts]
        width = max(map(len, rows))
        return {
            'input_ids': Tensor([[0] * (width - len(row)) + row for row in rows]),
            'attention_mask': Tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]),
        }

    def decode(self, tokens, **kwargs):
        return ''.join(chr(token) for token in tokens if token)


class Model:
    device = 'cpu'

    def __init__(self, responses):
        self.responses = responses

    def generate(self, input_ids, attention_mask, **kwargs):
        assert len(input_ids) == len(self.responses)
        return [row + [ord(c) for c in response]
                for row, response in zip(input_ids, self.responses)]


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.imports = patch.dict(sys.modules, {'custom_eval.prompts.templates': templates})
        self.imports.start()
        self.addCleanup(self.imports.stop)

    def test_padded_batch_and_pass_at_k_contain_only_completions(self):
        for k in (1, 2):
            for thinking in (False, True):
                with self.subTest(k=k, thinking=thinking):
                    responses = [r'\boxed{3}', r'\boxed{4}'] if k == 1 else [
                        r'\boxed{3}', r'\boxed{30}', r'\boxed{4}', r'\boxed{40}']
                    groups = generation.generate_candidates_batch(
                        Model(responses), Tokenizer(), ['short', 'long question ' * 30],
                        'gsm8k', enable_thinking=thinking, pass_at_k=k)
                    self.assertEqual([len(group) for group in groups], [k, k])
                    self.assertEqual([c.raw_output for group in groups for c in group], responses)
                    self.assertEqual([c.final_response for group in groups for c in group],
                                     ['3', '4'] if k == 1 else ['3', '30', '4', '40'])

    def test_thinking_flag_reaches_template_only_for_qwen(self):
        for model_type in ('qwen', 'llama'):
            for enabled in (False, True):
                tokenizer = Tokenizer()
                generation._format_prompt('question', 'gsm8k', tokenizer, model_type, enabled)
                kwargs = tokenizer.calls[0]
                if model_type == 'qwen':
                    self.assertEqual(kwargs['enable_thinking'], enabled)
                else:
                    self.assertNotIn('enable_thinking', kwargs)

    def test_last_box_overrides_placeholder_and_intermediate_answer(self):
        text = r'Format: \boxed{answer}. Intermediate: \boxed{2}. Final: \boxed{3}'
        for enabled in (False, True):
            self.assertEqual(generation.extract_final_response(text, 'gsm8k', enabled), '3')

    def test_thinking_opening_tag_can_be_in_prompt(self):
        for opening in ('', '<think>\n'):
            raw = opening + 'reasoning with \\boxed{2}</think>\n\\boxed{3}'
            candidate = generation.generate_candidates_batch(
                Model([raw]), Tokenizer(), ['question'], 'gsm8k', enable_thinking=True)[0][0]
            self.assertEqual(candidate.thinking_content, r'reasoning with \boxed{2}')
            self.assertEqual(candidate.final_response, '3')

    def test_multiple_choice_prompt_example_does_not_leak(self):
        responses = ['{"answer": "A"}', '{"answer": "B"}']
        candidates = generation.generate_candidates_batch(
            Model(responses), Tokenizer(), ['short', 'long question ' * 30],
            'mmlu', enable_thinking=False)
        self.assertEqual([group[0].final_response for group in candidates], ['A', 'B'])


if __name__ == '__main__':
    unittest.main()
