# Reproducing GeneZip results

This guide describes the experiments in the [paper](https://openreview.net/pdf?id=E37ysCRa5d).
The repository provides seven GeneZip pretrained checkpoints, pretraining
templates, and CMP/eQTL/ETGP/TISP training and evaluation. See the
[README](../README.md#environment) for installation and the [result tables](CHECKPOINTS.md)
for benchmark scores.

## Training commands

Run shell entrypoints from the repository root. W&B is optional for downstream
commands (`USE_WANDB=true`). `HF_USER` selects the Hugging Face namespace and defaults to `andyjzhao`.

All four wrappers default to `andyjzhao/GeneZip-70M` and download the checkpoint
automatically:

```sh
bash shell/dnalongbench/cmp.sh
bash shell/dnalongbench/eqtl.sh
bash shell/dnalongbench/etgp.sh
bash shell/dnalongbench/tisp.sh
```

### Model downloads

To download the model into a local directory:

```sh
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="andyjzhao/GeneZip-70M",
    local_dir="./pretrained/GeneZip-70M",
)
PYTHON
CKPT=./pretrained/GeneZip-70M bash shell/dnalongbench/etgp.sh
```

### Task selection

The CMP command defaults to HFF. To evaluate all five cell lines:

```sh
for SUBSET in HFF H1hESC GM12878 IMR90 HCT116; do
  SUBSET="$SUBSET" bash shell/dnalongbench/cmp.sh
done
```

The eQTL command defaults to Artery_Tibial. Use `CELL_TYPE` to select one of the
[nine tissues](#task-subsets).

Select a regional-prior model with `CKPT=GeneZip-Regulome`,
`CKPT=GeneZip-PromoterDistal` or `CKPT=GeneZip-Intergenic` in the corresponding
eQTL/ETGP/TISP wrapper.

### Task subsets

CMP uses HFF, H1hESC, GM12878, IMR90 and HCT116.

eQTL uses the following values for `CELL_TYPE`:

- `Adipose_Subcutaneous`
- `Artery_Tibial`
- `Cells_Cultured_fibroblasts`
- `Muscle_Skeletal`
- `Nerve_Tibial`
- `Skin_Not_Sun_Exposed_Suprapubic`
- `Skin_Sun_Exposed_Lower_leg`
- `Thyroid`
- `Whole_Blood`

ETGP uses K562. TISP covers FC, EC, ER, GC and PC.

## Evaluating task checkpoints

To repeat eQTL or ETGP evaluation after training, reuse the same model, subset
and checkpoint directory. These examples use the wrappers' default settings
and restore the checkpoint selected by validation AUROC:

```sh
python src/scripts/finetune_dlb_binary_classification.py task=eqtl \
  cell_type=Artery_Tibial save_dir=./temp/eQTL_ckpt training.do_train=false test_load=best
python src/scripts/finetune_dlb_binary_classification.py task=etgp \
  cell_type=CRISPRi_EPI_K562_hg19 save_dir=./temp/ETGP_ckpt training.do_train=false test_load=best
```

To evaluate a trained TISP checkpoint, use the same pretrained model and task
settings with a fresh output directory. For a completed 5,000-step run:

```sh
python src/scripts/finetune_dlb_tisp.py task=tisp \
  evaluation_checkpoint=/path/to/training/best_model.pt \
  completed_training_steps=5000 train_steps=5000 dirs.output=./output/tisp-evaluation
```

Set `completed_training_steps` and `train_steps` to the training run's completed
step count. The command evaluates the validation and test splits.

Training produces the task checkpoints used for evaluation.
When changing the eQTL tissue or pretrained model, use a distinct `SAVE_DIR`
(e.g. `CELL_TYPE=Whole_Blood SAVE_DIR=./temp/eqtl_Whole_Blood bash shell/dnalongbench/eqtl.sh`)
and pass that same directory and `ckpt` to independent evaluation.

## Task protocols

All main-task commands start from `andyjzhao/GeneZip-70M`. eQTL/ETGP/TISP training includes
validation selection and final held-out evaluation. CMP reports validation
and test metrics every 200 head updates.

| Task | Input (bp) | Seed | Batch / accumulation | Budget | Selection |
| --- | ---: | ---: | --- | --- | --- |
| CMP | 1,048,576, encoded in 12,800-bp chunks | 0 | 32 / 1 | 5,000 head updates | Validation/test reporting every 200 updates |
| eQTL | 450,000 | 1 | 1 / 8 | 3 epochs | Validation AUROC, every 100 updates |
| ETGP | 450,000 | 1 | 1 / 8 | 3 epochs | Validation AUROC, every 100 updates |
| TISP | 100,000 | 0 | 1 / 4 | 5,000 updates | Chromosome-level validation mean PCC, every 1,000 updates |

CMP freezes the complete sequence model, uses mean pooling and trains a contact-map head with
LR 2e-4 and weight decay 0. The other tasks freeze embeddings, encoders and
routers while training the subsequent backbone modules and task head with
LR 2e-4 and weight decay 0.1. eQTL/ETGP use no LoRA and retain the 128-BPT
compression regularizer (alpha 0.03), independently of the pretraining N=32 anchor. TISP uses pseudo-Poisson KL without a
ratio-loss term. All use BF16 and gradient clipping at 1.0.

For eQTL, ETGP and TISP, all four GeneZip priors use a ceiling of 70,000
tokens at each routing stage, with no additional floor, in both training
and validation/test. Content-dependent routing and the first-valid-token
rule remain enabled. These settings differ from pretraining, which uses
a floor of eight tokens per stage. CMP retains the pretrained routing floor.

Aggregate eQTL scores average all nine tissues; CMP averages all five cell
lines. TISP averages FC/EC/ER/GC/PC PCC computed on the full held-out
chromosomes. Checkpoints are selected using validation metrics.

The same eQTL/ETGP/TISP adaptation protocol applies to the regional-prior
models. The paper compares H-Net and U-Net under this common protocol;
external-pretraining and official benchmark rows retain their reported protocols.

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

## Inference throughput

The main table times 100 alternate sequences from the Artery_Tibial test
split, each 450,000 bp, on one NVIDIA H100 80GB with BF16. It uses pretrained
backbones with mean pooling and newly initialized binary heads, independently
of task-specific adaptation. Inputs retain orientation, variant masking and
N padding and are pretokenized in pinned CPU memory.

Timing includes host-to-device transfer, mask construction, backbone forward,
pooling and classification. It excludes loading, tokenization, compilation,
warmup, output transfer to CPU and numerical checks. Reported times are the
median of three complete passes after warmup, with CUDA synchronization.
Each model uses its largest batch verified across all 100 inputs, including
the final partial batch. Outputs are checked against batch-1 inference with
absolute and relative tolerances of 0.02.

GeneZip takes 6.34 s at batch 7 (72.80 GiB peak allocated memory), versus
JanusDNA's 228.63 s at batch 9 (41.11 GiB), giving 36.1-fold higher throughput.
GeneZip uses ceilings `[70000,70000]`; the H-Net timing run has no routing
limits, separately from the common downstream adaptation protocol.
NTv3 includes padding to 450,048 positions in its timing and excludes the
added positions from pooling. Evo2-1B runs out of memory on one GPU.

## Public data and checkpoints

The loaders download these public datasets automatically:

- [DNALongBench](https://huggingface.co/datasets/andyjzhao/dnalongbench)
- [GENCODE 12.8K](https://huggingface.co/datasets/andyjzhao/gencode_human_12.8k)
- [GENCODE 128K](https://huggingface.co/datasets/andyjzhao/gencode_human_128k)

GENCODE uses Parquet and the original `train`/`valid` split. DNALongBench
uses its published task directories. Dataset paths are configured in
`configs/data.yaml` and `configs/main.yaml`; see [model downloads](#model-downloads)
and [task selection](#task-selection) for checkpoint choices.
Use `offline_mode=true` only after the required files are cached locally.

## Quick pipeline check

Run two TISP optimization steps using eight training windows and two windows
per validation/test split:

```sh
TRAIN_STEPS=2 EVAL_STEPS=1 LOG_EVERY=1 \
  NUM_TRAIN_SAMPLES=8 NUM_VALID_SAMPLES=2 NUM_TEST_SAMPLES=2 \
  bash shell/dnalongbench/tisp.sh
```
