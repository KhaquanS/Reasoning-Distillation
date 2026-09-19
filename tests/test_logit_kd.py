"""Logit KD regression and offline training checks (no model downloads)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HAS_TRAINING_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "datasets", "yaml")
)

if HAS_TRAINING_DEPS:
    import torch
    import torch.nn.functional as F
    import yaml
    from transformers import GPT2Config, GPT2LMHeadModel
    from scripts import train
    from training.logit_kd_trainer import LogitKDTrainer


@unittest.skipUnless(HAS_TRAINING_DEPS, "Training dependencies are required")
class LogitKDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _model(self):
        return GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_positions=16, n_embd=8, n_layer=1, n_head=1,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            pad_token_id=0, eos_token_id=1, bos_token_id=1,
        ))

    def _trainer(self):
        teacher = self._model().eval()
        teacher.requires_grad_(False)
        config = SimpleNamespace(
            lr=2.0e-5, adam_betas=(0.9, 0.98), weight_decay=0.01,
            temperature=2.5, alpha_kd=0.3, beta_ce=0.9,
        )
        return LogitKDTrainer(
            self._model(), teacher, SimpleNamespace(pad_token_id=0), config
        )

    def test_loss_matches_forward_kl_and_next_token_ce(self):
        trainer = self._trainer()
        batch = {
            "input_ids": torch.tensor([[2, 3, 4, 5], [6, 7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        }
        actual = trainer._compute_loss(batch)
        with torch.no_grad():
            s_logits = trainer.student(**batch).logits
            t_logits = trainer.teacher(**batch).logits
            temperature = trainer.config.temperature
            s_log_probs = (s_logits / temperature).log_softmax(-1)
            t_log_probs = (t_logits / temperature).log_softmax(-1)
            token_kl = (t_log_probs.exp() * (t_log_probs - s_log_probs)).sum(-1)
            mask = batch["attention_mask"]
            kd = (token_kl * mask).sum() / mask.sum() * temperature ** 2
            ce = F.cross_entropy(
                s_logits[:, :-1].reshape(-1, 16),
                batch["input_ids"][:, 1:].reshape(-1), ignore_index=0,
            )
        torch.testing.assert_close(actual, 0.3 * kd + 0.9 * ce)
        actual.backward()
        gradients = [p.grad for p in trainer.student.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
        self.assertTrue(any(grad.abs().sum() > 0 for grad in gradients))
        self.assertTrue(all(p.grad is None for p in trainer.teacher.parameters()))

    def test_padding_does_not_change_loss(self):
        trainer = self._trainer()
        unpadded = {
            "input_ids": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        padded = {
            "input_ids": torch.tensor([[2, 3, 4, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }
        torch.testing.assert_close(
            trainer._compute_loss(unpadded), trainer._compute_loss(padded)
        )

    def test_configs_match_both_reasondistill_stages(self):
        for suffix, reference in (
            ("", "qwen_reasondistill_final"),
            ("_continue", "qwen_reasondistill_final_continue"),
        ):
            with self.subTest(stage=suffix or "initial"):
                kd = train._load_yaml_config(ROOT / f"configs/qwen_logit_kd{suffix}.yaml")
                reason = train._load_yaml_config(ROOT / f"configs/{reference}.yaml")
                keys = (
                    train.CONFIG_SECTIONS["training"]
                    + train.CONFIG_SECTIONS["data"]
                    + ["teacher", "student", "teacher_quantize_8bit", "seed",
                       "temperature", "alpha_kd", "beta_ce",
                       "loss_log_entries_per_epoch", "save_every_n_steps"]
                )
                for key in keys:
                    self.assertEqual(getattr(kd, key), getattr(reason, key), key)
                self.assertEqual(kd.method, "logit_kd")
                self.assertEqual(kd.alpha_align, 0.0)
                self.assertIsNone(kd.sae_checkpoint)
                self.assertIsNone(kd.reason_score_path)
                self.assertNotEqual(kd.checkpoint_dir, reason.checkpoint_dir)
                self.assertNotEqual(kd.log_dir, reason.log_dir)
                if suffix:
                    self.assertEqual(kd.student_checkpoint, "./checkpoints/qwen_logit_kd/epoch_1")
                else:
                    self.assertIsNone(kd.student_checkpoint)

    def test_two_stage_training_saves_and_loads_kd_checkpoint(self):
        class Tokenizer:
            pad_token_id = 0

            def __call__(self, texts, **kwargs):
                ids = torch.tensor([[2, 3, 4, 5] for _ in texts])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

            def save_pretrained(self, path):
                Path(path, "tokenizer_config.json").write_text("{}")

        with tempfile.TemporaryDirectory() as directory:
            first_checkpoint = Path(directory) / "initial" / "epoch_1"
            for suffix, stage in (("", "initial"), ("_continue", "continue")):
                raw = yaml.safe_load((ROOT / f"configs/qwen_logit_kd{suffix}.yaml").read_text())
                raw["training"].update(accum_steps=2, max_length=8)
                raw["data"].update(max_samples=4, skip_samples=4 if suffix else 0)
                raw["model"]["student_checkpoint"] = str(first_checkpoint) if suffix else None
                raw["logging"].update(
                    checkpoint_dir=str(Path(directory) / stage),
                    log_dir=str(Path(directory) / stage / "logs"),
                    save_every_n_steps=1,
                )
                config_path = Path(directory) / f"{stage}.yaml"
                config_path.write_text(yaml.safe_dump(raw))
                args = train._load_yaml_config(config_path)
                teacher = self._model().eval().requires_grad_(False)
                student = self._model()

                def load_checkpoint(path, *args, **kwargs):
                    self.assertEqual(Path(path), first_checkpoint)
                    return GPT2LMHeadModel.from_pretrained(path), None

                with (
                    patch.object(train, "parse_args", return_value=args),
                    patch.object(train.torch.cuda, "is_available", return_value=False),
                    patch.object(train, "load_tokenizer", return_value=Tokenizer()),
                    patch.object(train, "AMDeepSeekDataset", return_value=[{"text": "test"}] * 4) as dataset,
                    patch.object(train, "load_teacher", return_value=teacher),
                    patch.object(train, "load_student", return_value=student),
                    patch.object(train, "load_student_checkpoint", side_effect=load_checkpoint) as checkpoint,
                ):
                    train.main()
                self.assertEqual(dataset.call_args.kwargs["skip_samples"], 4 if suffix else 0)
                self.assertEqual(checkpoint.call_count, 1 if suffix else 0)
                output = Path(directory) / stage
                state = torch.load(output / "epoch_1/trainer_state.pt", weights_only=True)
                self.assertEqual(state["global_step"], 2)
                self.assertTrue((output / "step_1/model.safetensors").exists())
                self.assertTrue((output / "epoch_1/model.safetensors").exists())
                self.assertTrue((output / "logs/training_loss.csv").exists())
                self.assertFalse((output / "epoch_1/aligner.pt").exists())
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_failed_continuation_does_not_train_from_scratch(self):
        args = train._load_yaml_config(ROOT / "configs/qwen_logit_kd_continue.yaml")
        with (
            patch.object(train, "parse_args", return_value=args),
            patch.object(train.torch.cuda, "is_available", return_value=False),
            patch.object(train, "load_tokenizer"),
            patch.object(train, "AMDeepSeekDataset", return_value=[{"text": "test"}]),
            patch.object(train, "load_teacher", return_value=self._model()),
            patch.object(train, "load_student", return_value=self._model()),
            patch.object(train, "load_student_checkpoint", side_effect=FileNotFoundError("missing")),
            patch.object(LogitKDTrainer, "train") as training,
        ):
            with self.assertRaisesRegex(RuntimeError, "Refusing to continue"):
                train.main()
            training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
