"""Strict-load a staged pretrained model and check its configured routing on CUDA."""

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import argparse
import json
from pathlib import Path

import torch

from src.genezip.hnet_train import build_model
from src.utils import RunConfig, resolve_hydra_cfg_path
from src.downstream.utils import load_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    cfg = RunConfig.from_path(resolve_hydra_cfg_path(str(args.checkpoint)))
    device = torch.device("cuda")
    model, _ = build_model(cfg, device)
    model.load_state_dict(load_state_dict(str(args.checkpoint / "pytorch_model.bin"),
                                         torch.device("cpu")), strict=True)
    model = model.to(dtype=torch.bfloat16).eval()
    torch.manual_seed(0)
    tokens = torch.randint(4, 8, (1, 2048), device=device)
    with torch.no_grad():
        logits = model(input_ids=tokens, mask=torch.ones_like(tokens, dtype=torch.bool)).logits
    if not torch.isfinite(logits).all():
        raise AssertionError("Nonfinite logits")
    print(json.dumps({"model": cfg.model.name, "status": "passed", "strict_load": True,
                      "sequence_length": 2048, "routing_class": type(model).__name__,
                      "finite_logits": True}))


if __name__ == "__main__":
    main()
