<div align="center">

# GeneZip: Region-Aware Compression for Long-Context DNA Modeling

</div>

Camera Ready code review candidate on `cr_release`. The main benchmark mapping
was checked against the current **V7.13** paper source tables on 2026-11-01.
The seven CR pretrained models are published on Hugging Face. Current validation coverage and limits are recorded in
[the release validation record](results/reproduce_release.md).

GeneZip combines hierarchical dynamic routing, a Region-Aware Ratio (RAR)
training objective and bounded routing. Gene annotations supervise compression
training; inference requires raw DNA only. GeneZip-70M reaches 137.6 BPT with a
0.31 PPL increase relative to uncompressed Mamba2-84M. A common pretrained
GeneZip-70M checkpoint supports four downstream tasks, outperforming H-Net and
U-Net on their primary metrics. Its average rank is 3.13; NTv3 has the best
rank across all pretraining groups. In the paper's single-H100 benchmark of
100 real 450-kb sequences, inference takes 6.34 seconds versus JanusDNA's
228.63 seconds (36.1×). This is an inference measurement, not training speed.

![GeneZip overview](asset/GeneZip.png)

## Model and result mapping

See [the complete checkpoint/task matrix](results/CHECKPOINTS.md), its
[machine-readable manifest](results/checkpoints.json), and the
[paper result CSV](results/tables/dnalongbench.csv).

### Main models

| Canonical model | Pretraining context | Use |
| --- | ---: | --- |
| [GeneZip-70M](https://huggingface.co/andyjzhao/GeneZip-70M) | 12,800 | Default; all four main tasks |
| [GeneZip-70M-128K](https://huggingface.co/andyjzhao/GeneZip-70M-128K) | 128,000 | Pretraining context scaling |
| [GeneZip-636M](https://huggingface.co/andyjzhao/GeneZip-636M) | 12,800 | Pretraining model scaling |
| [GeneZip-636M-128K](https://huggingface.co/andyjzhao/GeneZip-636M-128K) | 128,000 | Model/context scaling |

### Ablation models

These checkpoints support the regional-prior ablation in the paper appendix.

| Canonical model | Pretraining context | Use |
| --- | ---: | --- |
| [GeneZip-Regulome](https://huggingface.co/andyjzhao/GeneZip-Regulome) | 12,800 | N=64 regional prior |
| [GeneZip-PromoterDistal](https://huggingface.co/andyjzhao/GeneZip-PromoterDistal) | 12,800 | N=64 regional prior |
| [GeneZip-Intergenic](https://huggingface.co/andyjzhao/GeneZip-Intergenic) | 12,800 | N=32 regional prior |

Historical names are retained as source provenance. This release covers GeneZip
pretrained weights and pretraining/CMP/eQTL/ETGP/TISP code. Downstream task
checkpoints are not distributed.

## Environment

Linux and an NVIDIA GPU are required for model execution. The inherited lockfile
uses Python 3.12, PyTorch 2.8/CUDA 12.6, flash-attn 2.8.3, mamba-ssm 2.3.0 and
causal-conv1d 1.6.0. This incremental release reuses the existing environment; a fresh-install audit
is outside the agreed validation scope.

```sh
uv venv --python 3.12
uv sync
source .venv/bin/activate
```

Run shell entrypoints from the repository root. W&B is optional for downstream
commands (`USE_WANDB=true`). HF uploads are disabled by default. `HF_USER` is the
checkpoint namespace, not a requirement to create a personal HF account.

## Main downstream recipes

All four wrappers default to the public `andyjzhao/GeneZip-70M` weights.
Their default commands download the checkpoint automatically:

```sh
bash shell/dnalongbench/cmp.sh
bash shell/dnalongbench/eqtl.sh
bash shell/dnalongbench/etgp.sh
bash shell/dnalongbench/tisp.sh
```

For an immutable download, use `snapshot_download` with the model's
`release_revision` from `results/checkpoints.json`, then pass its returned path
as `CKPT`. To reconstruct the same weights from their historical source:

```sh
python scripts/export_pretrained.py --model GeneZip-70M --output ./pretrained
CKPT=./pretrained/GeneZip-70M bash shell/dnalongbench/etgp.sh
```

The export verifies the source SHA-256 and writes a minimal `hydra_cfg.yaml`
with explicit routing settings. It never uploads files.

CMP requires all five subsets, not only the wrapper's HFF default:

```sh
for SUBSET in HFF H1hESC GM12878 IMR90 HCT116; do
  SUBSET="$SUBSET" CKPT=/path/to/local/GeneZip-70M bash shell/dnalongbench/cmp.sh
done
```

eQTL requires all nine tissues listed in `results/checkpoints.json`; the wrapper
starts with Artery_Tibial. Both eQTL and ETGP use 450,000 bp, seed 1, no LoRA,
frozen embeddings/encoder/router, LR 2e-4, three epochs and best validation AUROC.
Their compression regularizer targets 128 BPT (alpha 0.03), independently of
the pretraining N=32 anchor. TISP uses WD 0.01 and routing caps `[60000,30000]`. CMP evaluates every 200 steps. Alternative-prior tasks require
their own recipes; changing
only `CKPT` is not a reproduction of Table 12.

Data loaders download the revisions pinned in `configs/data.yaml` and
`configs/main.yaml`. GENCODE uses Parquet with its original `train`/`valid` split;
DNALongBench uses its published task directories. Set `offline_mode=true` only
after the required data and model files are available locally.

## Pretraining templates

```sh
bash shell/pretrain/GeneZip-70M-12.8K-template.sh
bash shell/pretrain/GeneZip-70M-128K-template.sh
bash shell/pretrain/GeneZip-636M-12.8K-template.sh
bash shell/pretrain/GeneZip-636M-128K-template.sh
```

128K training continues from the same-size 12.8K checkpoint. Each stage uses its
own training budget. The inherited templates use eight GPUs; they do not by
themselves demonstrate the paper's single-GPU feasibility claim. Regional targets
are declared in `configs/prior/`, with N=64 for Regulome and PromoterDistal.
Regulome, PromoterDistal and Intergenic use routing floors `[8,8]` and caps
`[5000,2000]`. The 636M templates use `[3000,3000]` at 12.8K and `[30000,10000]`
at 128K, matching their pinned source configurations.

Evaluate the staged 70M checkpoint on the pretraining validation split:

```sh
python src/scripts/pretrain_genezip.py task=pretrain \
  pretrained_ckpt=./pretrained/GeneZip-70M training.do_train=false training.do_eval=true
```

For a short training check, set `max_train_steps=2 num_train_samples=4
num_valid_samples=2 eval_steps=1 save_steps=1`. These sample and step caps are
validation overrides, not the paper's training budget.

## Validation

Run CPU contract checks and GPU smoke in an allocated compute environment:

```sh
python -m unittest discover -s tests -v
python scripts/smoke_pretrained.py --checkpoint ./pretrained/GeneZip-70M
python scripts/smoke_cr_checkpoint.py --checkpoint /path/to/local/GeneZip-70M
python scripts/smoke_cr_checkpoint.py --checkpoint /path/to/local/GeneZip-70M --sequence-length 450000
```

The GPU smoke checks strict loading, finite forward/backward, frozen-parameter
behavior and save/reload equivalence. It is not a full benchmark reproduction.

Code is Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
