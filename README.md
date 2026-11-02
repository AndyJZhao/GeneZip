<div align="center">

# GeneZip: Region-Aware Compression for Long-Context DNA Modeling

</div>

Long-context DNA models are limited by token-mixing cost and by how compression allocates representational budget across the genome. Existing approaches operate close to base-pair resolution, apply fixed downsampling, or learn content-dependent chunks without an explicit genomic budget, making long-context pretraining expensive and difficult to control. We introduce GeneZip, a region-aware DNA compression framework that combines H-Net-style dynamic routing with a Region-Aware Ratio (RAR) objective and bounded routing. GeneZip uses static gene-structure annotations during compression training to specify region-wise base-pairs-per-token (BPT) targets; at inference time, it compresses raw unseen DNA without annotations. GeneZip provides three main benefits. First, it is effective: GeneZip variants achieve the best validation perplexity (PPL) among high-compression encoder-based models. At 137.6 BPT, GeneZip-70M incurs only a +0.31 perplexity increase relative to the 84M Mamba2 baseline without compression. Across four DNALongBench tasks—contact map prediction, eQTL prediction, enhancer–target gene prediction, and transcription-initiation signal prediction—GeneZip outperforms all compression baselines under matched leakage-controlled pre-training. Second, it is redundancy-aware: a post-hoc RepeatMasker/TRF analysis shows that, without repeat supervision, GeneZip assigns higher local BPT to selected TE-derived interspersed repeats and tandem repeats, two major classes of repetitive DNA sequence redundancy. Third, it is efficient: on 450-kb inputs, 70M GeneZip achieves a 36.1× inference speedup over the uncompressed JanusDNA-7.7M model. GeneZip also supports 128K-context and 636M-parameter pretraining on a single H100 80GB GPU. These results establish GeneZip as an effective, redundancy-aware, and efficient compression interface for long-context DNA modeling.

![GeneZip overview](asset/GeneZip.png)

## Pretrained models

See [model details and benchmark results](results/CHECKPOINTS.md) and the
[paper result CSV](results/tables/dnalongbench.csv).

The regional multipliers $\mu_r$ are listed in the order **promoter, CDS, UTR,
exon, intron, NIG, DIG**. The target BPT for each region is $N \mu_r$.

### Main models

| Model | Pretraining context | $\mu_r$ | Use |
| --- | ---: | --- | --- |
| [GeneZip-70M](https://huggingface.co/andyjzhao/GeneZip-70M) | 12,800 | (1, 1, 2, 2, 8, 8, 16) | Default; all four main tasks |
| [GeneZip-70M-128K](https://huggingface.co/andyjzhao/GeneZip-70M-128K) | 128,000 | (1, 1, 2, 2, 8, 8, 16) | Pretraining context scaling |
| [GeneZip-636M](https://huggingface.co/andyjzhao/GeneZip-636M) | 12,800 | (1, 1, 2, 2, 8, 8, 16) | Pretraining model scaling |
| [GeneZip-636M-128K](https://huggingface.co/andyjzhao/GeneZip-636M-128K) | 128,000 | (1, 1, 2, 2, 8, 8, 16) | Model/context scaling |

### Ablation models

These checkpoints support the regional-prior ablation in the paper appendix.

| Model | Pretraining context | $\mu_r$ | Use |
| --- | ---: | --- | --- |
| [GeneZip-Regulome](https://huggingface.co/andyjzhao/GeneZip-Regulome) | 12,800 | (1, 16, 2, 4, 2, 2, 4) | N=64 regional prior |
| [GeneZip-PromoterDistal](https://huggingface.co/andyjzhao/GeneZip-PromoterDistal) | 12,800 | (1, 16, 8, 8, 2, 2, 4) | N=64 regional prior |
| [GeneZip-Intergenic](https://huggingface.co/andyjzhao/GeneZip-Intergenic) | 12,800 | (32, 16, 8, 4, 2, 4, 1) | N=32 regional prior |

This repository provides pretrained weights, pretraining code, and code for
training and evaluating CMP, eQTL, ETGP and TISP on DNALongBench. Each task
starts from the pretrained weights using the commands below.

## Environment

Linux and an NVIDIA GPU are required for model execution. The lockfile
uses Python 3.12, PyTorch 2.8/CUDA 12.6, flash-attn 2.8.3, mamba-ssm 2.3.0 and
causal-conv1d 1.6.0.

```sh
uv venv --python 3.12
uv sync
source .venv/bin/activate
```

Run shell entrypoints from the repository root. W&B is optional for downstream
commands (`USE_WANDB=true`). `HF_USER` selects the Hugging Face namespace and defaults to `andyjzhao`.

## DNALongBench training and evaluation

All four wrappers default to the public `andyjzhao/GeneZip-70M` weights.
Their default commands download the checkpoint automatically:

```sh
bash shell/dnalongbench/cmp.sh
bash shell/dnalongbench/eqtl.sh
bash shell/dnalongbench/etgp.sh
bash shell/dnalongbench/tisp.sh
```

To download a fixed model revision into a local directory:

```sh
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="andyjzhao/GeneZip-70M",
    revision="15e46711bd6fa23fe5922dfe22e8893c9f01bec5",
    local_dir="./pretrained/GeneZip-70M",
)
PYTHON
CKPT=./pretrained/GeneZip-70M bash shell/dnalongbench/etgp.sh
```

The CMP command defaults to HFF. To evaluate all five cell lines:

```sh
for SUBSET in HFF H1hESC GM12878 IMR90 HCT116; do
  SUBSET="$SUBSET" CKPT=/path/to/local/GeneZip-70M bash shell/dnalongbench/cmp.sh
done
```

The eQTL command defaults to Artery_Tibial. Use `CELL_TYPE` to select one of the
[nine tissues](results/CHECKPOINTS.md#task-subsets). Both eQTL and ETGP use 450,000 bp, seed 1, no LoRA,
frozen embeddings/encoder/router, LR 2e-4, three epochs and best validation AUROC.
Their compression regularizer targets 128 BPT (alpha 0.03), independently of
the pretraining N=32 anchor. TISP uses WD 0.01 and routing caps `[60000,30000]`. CMP evaluates every 200 steps. Alternative-prior tasks require
task-specific settings described in [the ablation results](results/CHECKPOINTS.md#regional-prior-ablations). The wrappers above use the main-model settings.

To repeat eQTL or ETGP evaluation after training, reuse the same model, subset
and checkpoint directory. These examples use the wrappers' default settings
and restore the checkpoint selected by validation AUROC:

```sh
python src/scripts/finetune_dlb_binary_classification.py task=eqtl \
  cell_type=Artery_Tibial save_dir=./temp/eQTL_ckpt training.do_train=false test_load=best
python src/scripts/finetune_dlb_binary_classification.py task=etgp \
  cell_type=CRISPRi_EPI_K562_hg19 save_dir=./temp/ETGP_ckpt training.do_train=false test_load=best
```

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
own training budget. The templates default to eight GPUs. Regional targets
are declared in `configs/prior/`, with N=64 for Regulome and PromoterDistal.
Regulome, PromoterDistal and Intergenic use routing floors `[8,8]` and caps
`[5000,2000]`. The 636M templates use `[3000,3000]` at 12.8K and `[30000,10000]`
at 128K.

Evaluate the downloaded 70M checkpoint on the pretraining validation split:

```sh
python src/scripts/pretrain_genezip.py task=pretrain \
  pretrained_ckpt=./pretrained/GeneZip-70M training.do_train=false training.do_eval=true
```

For a short training check, set `max_train_steps=2 num_train_samples=4
num_valid_samples=2 eval_steps=1 save_steps=1`. These sample and step caps are
validation overrides, not the paper's training budget.

## Quick checks

Run the configuration/data-loader checks and short GPU checks:

```sh
python -m unittest discover -s tests -v
python scripts/smoke_pretrained.py --checkpoint ./pretrained/GeneZip-70M
python scripts/smoke_cr_checkpoint.py --checkpoint /path/to/local/GeneZip-70M
python scripts/smoke_cr_checkpoint.py --checkpoint /path/to/local/GeneZip-70M --sequence-length 450000
```

The GPU checks cover strict loading, finite forward/backward outputs, frozen
parameters and save/reload equivalence. Short checks validate execution; benchmark
scores require the full training and evaluation recipes. See
[validation coverage](results/reproduce_release.md) for details.

Code is Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
