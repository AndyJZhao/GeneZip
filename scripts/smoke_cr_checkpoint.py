"""Strict-load and training smoke for an explicit local GeneZip-70M source checkpoint."""

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import argparse
import json
from pathlib import Path
import tempfile

import torch

from src.scripts.finetune_dlb_binary_classification import (
    HNetForSequenceClassification,
    build_hnet_backbone,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    directory = args.checkpoint.resolve(strict=True)
    weights = directory / "pytorch_model.bin"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not torch.cuda.is_available():
        raise RuntimeError("Run this smoke on an allocated CUDA node")
    torch.manual_seed(1)
    backbone, _ = build_hnet_backbone(
        checkpoint_dir=str(directory), checkpoint_path=str(weights),
        device=torch.device("cuda"), dtype=torch.bfloat16, strict=True,
        routing_floor_tokens=[0, 0], routing_ceiling_tokens=[70000, 70000],
    )
    model = HNetForSequenceClassification(backbone=backbone, num_labels=2,
                                          pooling_method="mean", pad_token_id=0)
    model.freeze_encoder()
    frozen = [p for p in model.parameters() if not p.requires_grad]
    if not frozen or model.hnet.embeddings.weight.requires_grad:
        raise AssertionError("Expected frozen embeddings, encoder and router")
    batch = {"input_ids": torch.randint(4, 8, (1, args.sequence_length), device="cuda"),
             "attention_mask": torch.ones((1, args.sequence_length), device="cuda", dtype=torch.bool),
             "labels": torch.tensor([1], device="cuda")}
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    output = model(**batch)
    if not torch.isfinite(output["loss"]) or not torch.isfinite(output["logits"]).all():
        raise AssertionError("Nonfinite forward output")
    output["loss"].backward()
    if any(p.grad is not None for p in frozen):
        raise AssertionError("Frozen parameter received a gradient")
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise AssertionError("Missing or nonfinite gradients")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(**batch)["logits"].clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.pt"
        torch.save(model.state_dict(), path)
        with torch.no_grad():
            model.classifier.weight.zero_()
        model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True), strict=True)
        with torch.no_grad():
            actual = model(**batch)["logits"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(json.dumps({"status": "passed", "checkpoint": str(args.checkpoint),
                      "strict_load": True, "sequence_length": args.sequence_length,
                      "loss": float(output["loss"].detach()),
                      "frozen_parameter_tensors": len(frozen), "save_reload_equal": True}))


if __name__ == "__main__":
    main()
