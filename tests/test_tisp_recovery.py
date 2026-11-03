"""CPU regression checks for evaluation of a completed TISP checkpoint."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
import torch

from src.scripts import finetune_dlb_tisp as app


class TISPRecovery(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "selected.pt"
        self.model = torch.nn.Linear(2, 1)
        self.expected = {name: torch.ones_like(value) for name, value in self.model.state_dict().items()}
        torch.save(self.expected, self.path)
        self.cfg = OmegaConf.create({
            "learning_rate": 2e-4, "weight_decay": 0.1, "train_steps": 5000,
            "evaluation_checkpoint": str(self.path), "completed_training_steps": 5000,
            "dirs": {"output": str(Path(self.directory.name) / "evaluation")},
            "sequence_length": 100000, "tisp_loss": "pseudo_poisson_kl",
        })
        self.accelerator = SimpleNamespace(
            prepare=lambda *items: items, unwrap_model=lambda model: model,
            wait_for_everyone=lambda: None, is_main_process=True, device=torch.device("cpu"),
        )

    def run_recovery(self):
        return app.train_tisp(
            cfg=self.cfg, model=self.model, train_loader=Mock(), valid_loader=Mock(),
            test_loader=Mock(), tokenizer=None, accelerator=self.accelerator,
        )

    def test_reload_without_training_or_checkpoint_replacement(self):
        original = self.path.read_bytes()

        def evaluate(**kwargs):
            for name, value in kwargs["model"].state_dict().items():
                torch.testing.assert_close(value, self.expected[name])
            split = kwargs["split"]
            return {f"{split}/avg_pcc": 0.3 if split == "valid" else 0.9}

        with patch.object(app, "evaluate_tisp", side_effect=evaluate) as evaluation, \
                patch.object(app, "save_tisp_checkpoint") as save, \
                patch.object(torch.optim.AdamW, "step") as step:
            metrics = self.run_recovery()
        self.assertEqual([call.kwargs["split"] for call in evaluation.call_args_list], ["valid", "test"])
        self.assertEqual(metrics["best_pcc"], 0.3)
        self.assertEqual(metrics["train_steps_completed"], 5000)
        step.assert_not_called()
        save.assert_not_called()
        self.assertEqual(self.path.read_bytes(), original)

    def test_reject_incomplete_training_evidence(self):
        for completed in (None, 4999):
            with self.subTest(completed=completed):
                self.cfg.completed_training_steps = completed
                with self.assertRaisesRegex(ValueError, "completed_training_steps"):
                    self.run_recovery()

    def test_missing_checkpoint_cannot_fall_back_to_random_weights(self):
        self.path.unlink()
        with patch.object(app, "save_tisp_checkpoint") as save:
            with self.assertRaises(FileNotFoundError):
                self.run_recovery()
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
