# Reproducing the camera-ready results

The [README](../README.md) and [result tables](CHECKPOINTS.md) follow the
accepted camera-ready manuscript. The release provides seven GeneZip
pretrained checkpoints, pretraining templates, and CMP/eQTL/ETGP/TISP
training and evaluation. Baseline scores are comparisons; their training
implementations and downstream task checkpoints are not distributed here.

## Task protocols

All main-task commands start from `andyjzhao/GeneZip-70M`. eQTL/ETGP/TISP training includes
validation selection and final held-out evaluation. CMP reports validation
and test metrics every 200 head updates; it does not expose a standalone
head-checkpoint evaluation command. Use the README commands
for independent binary evaluation and interrupted TISP evaluation recovery.

| Task | Input (bp) | Seed | Batch / accumulation | Budget | Selection |
| --- | ---: | ---: | --- | --- | --- |
| CMP | 1,048,576, encoded in 12,800-bp chunks | 0 | 32 / 1 | 5,000 head updates | Validation/test reporting every 200 updates |
| eQTL | 450,000 | 1 | 1 / 8 | 3 epochs | Validation AUROC, every 100 updates |
| ETGP | 450,000 | 1 | 1 / 8 | 3 epochs | Validation AUROC, every 100 updates |
| TISP | 100,000 | 0 | 1 / 4 | 5,000 updates | Chromosome-level validation mean PCC, every 1,000 updates |

CMP freezes the complete sequence model and trains a contact-map head with
LR 2e-4 and weight decay 0. The other tasks freeze embeddings, encoders and
routers while training the subsequent backbone modules and task head with
LR 2e-4 and weight decay 0.1. eQTL/ETGP use no LoRA and retain the 128-BPT
compression regularizer (alpha 0.03). TISP uses pseudo-Poisson KL without a
ratio-loss term. All use BF16 and gradient clipping at 1.0.

For eQTL, ETGP and TISP, all four GeneZip priors use a ceiling of 70,000
tokens at each routing stage, with no additional floor, in both training
and validation/test. Content-dependent routing and the first-valid-token
rule remain enabled. These settings differ from pretraining, which uses
a floor of eight tokens per stage. CMP retains the pretrained routing floor.

Aggregate eQTL scores average all nine tissues; CMP averages all five cell
lines. TISP averages FC/EC/ER/GC/PC PCC computed on the full held-out
chromosomes, not an average over a small sample of windows. Checkpoints are
selected using validation metrics. Single-seed results and GPU kernels do
not imply bitwise equality on reruns.

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
This is not a training-speed or single-sequence-latency comparison.
GeneZip uses ceilings `[70000,70000]`; the H-Net timing run has no routing
limits, separately from the common downstream adaptation protocol.
NTv3 includes padding to 450,048 positions in its timing and excludes the
added positions from pooling. Evo2-1B runs out of memory on one GPU.

## Public data and checkpoints

The loaders download these public dataset revisions automatically:

| Dataset | Revision |
| --- | --- |
| [DNALongBench](https://huggingface.co/datasets/andyjzhao/dnalongbench) | `4764bc626b4f4187cefb9e1c31355f62f4d3ee66` |
| [GENCODE 12.8K](https://huggingface.co/datasets/andyjzhao/gencode_human_12.8k) | `af222ba30cb3bd3e97c1b8b678b4f8c2ef71006b` |
| [GENCODE 128K](https://huggingface.co/datasets/andyjzhao/gencode_human_128k) | `883f3df895d6de4e5575c2f9b581c5f7c71058f1` |

GENCODE uses Parquet and the original `train`/`valid` split. DNALongBench
uses its published task directories. The default model's verified revision
is `15e46711bd6fa23fe5922dfe22e8893c9f01bec5`; the README shows how to download
that exact revision. Direct Python entrypoints also accept
`ckpt=andyjzhao/GeneZip-70M@15e46711bd6fa23fe5922dfe22e8893c9f01bec5`.
Use `offline_mode=true` only after the required files are cached locally.

## Validation scope

The released environment and seven pretrained checkpoints have been checked
on Linux/A100 80GB, including anonymous downloads, strict weight loading,
finite model outputs, and four-task training/validation/test smoke checks at
their real input lengths. Prior validation also covered short 70M/636M
pretraining and 128K continuation. Full eQTL/ETGP/TISP reruns were used to
check the execution paths; their scores are not substituted for the adopted
paper results. CMP uses its adopted full-run evidence and a real-length smoke.
A fresh dependency installation and complete multi-GPU pretraining are not
part of this release acceptance.

A short TISP pipeline check uses the same routing and freezing defaults:

```sh
TRAIN_STEPS=2 EVAL_STEPS=1 LOG_EVERY=1 \
  NUM_TRAIN_SAMPLES=8 NUM_VALID_SAMPLES=2 NUM_TEST_SAMPLES=2 \
  bash shell/dnalongbench/tisp.sh
```

These tiny-window scores validate execution only; they are not chromosome-level
benchmark estimates. The camera-ready update replaces the former TISP routing
recipe and associated results, updates the regional-prior eQTL results, and
recomputes main-table ranks from the displayed three-decimal task scores.
