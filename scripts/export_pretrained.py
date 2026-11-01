"""Stage pinned GeneZip pretrained weights locally; never upload to the Hub."""

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((root / "results/checkpoints.json").read_text())
    if args.model not in manifest["release_models"]:
        raise ValueError(f"Not a GeneZip release model: {args.model}")
    entry = manifest["models"][args.model]
    source = entry["source"]
    download = dict(repo_id=source["repo_id"], revision=source["revision"])
    weights = Path(hf_hub_download(filename=f"{source['subfolder']}/{source['weights_file']}", **download))
    checksum = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != source["weights_sha256"]:
        raise ValueError("Source weight checksum mismatch")
    cfg = OmegaConf.load(hf_hub_download(filename=source["config_file"], **download))
    architecture = OmegaConf.to_container(cfg.model_cfg, resolve=True)
    # Export only inference configuration; source configs may contain private runtime fields.
    model_cfg = {key: architecture[key] for key in (
        "arch_layout", "d_model", "d_intermediate", "vocab_size", "ssm_cfg",
        "attn_cfg", "tie_embeddings",
    )}
    routing = entry["pretraining_routing"]
    model_cfg.update(k_min_list=routing["floor_tokens"], k_max_list=routing["ceiling_tokens"])
    clean = dict(model_cfg=model_cfg, model={"name": args.model, "arch": "hnet"},
                 use_routing_floor=routing["use_floor"], use_routing_ceiling=routing["use_ceiling"],
                 bp_per_token=entry["base_bpt_anchor"], region_info=str(cfg.region_info),
                 max_length=entry["pretraining_context_bp"], tokenizer="fast")
    target = args.output / args.model
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, target / "pytorch_model.bin")
    OmegaConf.save(OmegaConf.create(clean), target / "hydra_cfg.yaml")
    (target / "provenance.json").write_text(json.dumps({
        "model": args.model, "source": source, "weights_sha256": checksum.hexdigest(),
        "configuration": "architecture and explicit routing only; no runtime credentials",
    }, indent=2) + "\n")
    print(json.dumps({"model": args.model, "directory": str(target),
                      "weights_sha256": checksum.hexdigest(), "status": "staged_not_published"}))


if __name__ == "__main__":
    main()
