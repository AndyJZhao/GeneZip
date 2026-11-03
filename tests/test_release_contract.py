"""Run on a CPU compute node; no model or dataset downloads."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def config(*overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="main", overrides=list(overrides))


class ReleaseContract(unittest.TestCase):
    def test_regional_pretraining_targets(self):
        for prior, name, anchor in (("genezip", "GeneZip-70M", 32),
                                    ("regulome", "GeneZip-Regulome", 64),
                                    ("promoter_distal", "GeneZip-PromoterDistal", 64),
                                    ("intergenic", "GeneZip-Intergenic", 32)):
            cfg = config(f"prior={prior}")
            self.assertEqual(cfg.pretraining_name, name)
            self.assertEqual(cfg.alias, name)
            self.assertEqual(cfg.bp_per_token, anchor)
            self.assertTrue(cfg.hf_repo.endswith("/" + name))
            if prior != "genezip":
                self.assertTrue(cfg.use_routing_ceiling)
                self.assertEqual(list(cfg.k_max_list), [5000, 2000])

    def test_shell_commands_compose(self):
        scripts = sorted((ROOT / "shell/dnalongbench").glob("*.sh"))
        scripts += sorted((ROOT / "shell/pretrain").glob("*.sh"))
        self.assertEqual(len(scripts), 8)
        for script in scripts:
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)
                output = subprocess.check_output(["bash", str(script)], text=True, cwd=ROOT,
                                                 env={**os.environ, "PRINT_ONLY": "true", "PROJECT_ROOT": str(ROOT)})
                argv = shlex.split(output.split("Running command:\n", 1)[1])
                start = next(i for i, arg in enumerate(argv) if arg.endswith(".py"))
                cfg = config(*argv[start + 1:])
                self.assertFalse(cfg.upload_to_hf)
                if cfg.task in ("eqtl", "etgp"):
                    self.assertEqual(cfg.data_max_length, 450000)
                    self.assertFalse(cfg.use_lora)
                    self.assertTrue(cfg.freeze_encoder)
                if cfg.task == "tisp":
                    self.assertEqual(cfg.sequence_length, 100000)
                    self.assertTrue(cfg.freeze_encoder)
                    self.assertEqual(cfg.weight_decay, 0.1)
                    self.assertFalse(cfg.use_routing_ceiling)
                    self.assertIsNone(cfg.k_max_list)

    def test_checkpoint_config_read_does_not_mutate(self):
        from src.utils.config import load_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hydra_cfg.yaml"
            original = b"# @package _global_\nmodel_cfg:\n  d_model: [512, 512, 768]\n"
            path.write_bytes(original)
            self.assertEqual(list(load_config(str(path)).model_cfg.d_model), [512, 512, 768])
            self.assertEqual(path.read_bytes(), original)

    def test_public_parquet_loader_preserves_valid_split(self):
        from datasets import Dataset
        from src.utils.hf_utils import load_hf_dataset_from_cfg, split_train_valid_test
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "data").mkdir()
            for split, values in (("train", [1, 2, 3]), ("valid", [9])):
                Dataset.from_dict({"value": values}).to_parquet(str(path / "data" / f"{split}-00000.parquet"))
            cfg = {"path": directory, "hf_repo": "fixture/data", "revision": "pinned",
                   "format": "parquet", "splits": ["train", "valid"]}
            with patch("src.utils.hf_utils.snapshot_download") as download:
                dataset = load_hf_dataset_from_cfg(cfg, cache_dir=str(path / "cache"))
            self.assertEqual(download.call_args.kwargs["revision"], "pinned")
            train, valid, test = split_train_valid_test(dataset, seed=0)
            self.assertEqual(list(train["value"]), [1, 2, 3])
            self.assertEqual(list(valid["value"]), [9])
            self.assertIsNone(test)

    def test_partial_dnalongbench_directory_is_completed(self):
        from src.utils.hf_utils import resolve_hf_dataset_root
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "eqtl").mkdir()
            with patch("src.utils.hf_utils.snapshot_download") as download:
                resolve_hf_dataset_root(directory, repo_id="fixture/data", required_subdir="eqtl",
                                        revision="pinned")
            self.assertEqual(download.call_args.kwargs["revision"], "pinned")


if __name__ == "__main__":
    unittest.main()
