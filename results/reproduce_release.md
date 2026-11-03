# Validation and reproduction

The reported benchmark results are listed in
[model details and benchmark results](CHECKPOINTS.md). Short checks validate the
training and evaluation pipelines; their scores are not benchmark estimates.

## Validation coverage

Model checks ran on NVIDIA A100 GPUs; configuration and regression checks
ran on CPU compute nodes.

| Component | Coverage |
| --- | --- |
| Pretrained weights | Anonymous downloads at fixed revisions, SHA-256 and configuration verification, strict loading and finite CUDA outputs for every published model |
| Configuration and data loading | Checks covering shell entrypoints, Hydra composition, Parquet loading, preservation of validation splits and partial-download recovery |
| 12.8K pretraining | Two training steps and evaluation for 70M and 636M; independent evaluation for 70M |
| 128K pretraining | One continuation step and evaluation for 70M and 636M |
| Regional priors | Two 12.8K training steps and evaluation for Regulome and PromoterDistal |
| CMP | 1,048,576-bp inputs, representation extraction, two head updates and validation/test evaluation |
| eQTL and ETGP | 450,000-bp inputs, two optimizer steps, validation-based selection, checkpoint reload and test evaluation |
| TISP | 100,000-bp inputs, two optimizer steps, validation-based selection, checkpoint reload and window-level test evaluation |
| Frozen TISP and evaluation recovery | CPU checks verify the shell/Hydra defaults, checkpoint restoration, validation/test evaluation without optimizer steps or checkpoint replacement, and rejection of missing checkpoints or incomplete training evidence |
| Main-task results | Configurations, scoring code and held-out metrics checked against all 16 experiment records |

The dataset checks used separate public-data caches. GENCODE preserves its
original train/valid splits: 950,000/50,000 records at 12.8K and 99,000/1,000 at
128K. CMP HFF contains 7,008/419/413 train/validation/test records.

The 2026-11-03 CPU regression suite passed all 11 tests. Full optimizer-state
resume, fresh installation, long-duration training and distributed training
were not tested. These checks did not rerun full pretraining or the
full downstream benchmarks. The TISP short check uses window-level scoring;
reported benchmark results use the chromosome-level evaluator.

## Fixed-protocol result sources

The updated GeneZip ETGP/TISP results use frozen embeddings, encoders and
routers, LR 2e-4 and WD 0.1. Checkpoints were selected only on validation
metrics; all fixed conditions were retained, without test-score selection.
These are single-seed results. eQTL and CMP retain their recorded sources;
the other historical baseline rows are not all audited as sharing this
adaptation protocol.

| Task/model | W&B source | Mila provenance |
| --- | --- | --- |
| TISP/GeneZip | [028d2db1](https://wandb.ai/jzshared/GeneZip-TISP/runs/028d2db1) | long, cn-g003, job 11506588 |
| TISP/Regulome | [training 8b1be5f0](https://wandb.ai/jzshared/GeneZip-TISP/runs/8b1be5f0), [evaluation f9a96861](https://wandb.ai/jzshared/GeneZip-TISP/runs/f9a96861) | training job 11506589; evaluation main, cn-i001, job 11508833 |
| TISP/PromoterDistal | [180acab4](https://wandb.ai/jzshared/GeneZip-TISP/runs/180acab4) | long, cn-g014, job 11506591 |
| TISP/Intergenic | [e88e8ae4](https://wandb.ai/jzshared/GeneZip-TISP/runs/e88e8ae4) | long, cn-g010, job 11506590 |
| ETGP/Regulome | [b74562af](https://wandb.ai/jzshared/DNALLM_ETGP/runs/b74562af) | short-unkillable, cn-g006, job 11506441 |
| ETGP/PromoterDistal | [53fa1098](https://wandb.ai/jzshared/DNALLM_ETGP/runs/53fa1098) | short-unkillable, cn-g006, job 11506441 |
| ETGP/Intergenic | [b39ffe83](https://wandb.ai/jzshared/DNALLM_ETGP/runs/b39ffe83) | short-unkillable, cn-g006, job 11506441 |

Regulome completed 5000 training steps before preemption during testing.
Recovery reused its original validation-selected checkpoint; the evaluation
run is not a second training run. Partition names above describe historical
execution, not requirements for reproducing the experiments.

Download GeneZip-70M into `./pretrained/GeneZip-70M` using the fixed-revision
[download example](../README.md#dnalongbench-training-and-evaluation) before
running the commands below.

## Commands

Run on an allocated NVIDIA GPU compute node, from the release repository root.
Public dataset revisions are pinned in `configs/data.yaml` and `configs/main.yaml`.

```sh
python -m unittest discover -s tests -v
python scripts/smoke_pretrained.py --checkpoint ./pretrained/GeneZip-70M
python scripts/smoke_cr_checkpoint.py --checkpoint ./pretrained/GeneZip-70M

python src/scripts/pretrain_genezip.py task=pretrain \
  num_train_samples=4 num_valid_samples=2 max_train_steps=2 \
  eval_steps=1 save_steps=1 training.warmup_steps=0 use_wandb=false

python src/scripts/pretrain_genezip.py task=pretrain \
  pretrained_ckpt=./pretrained/GeneZip-70M \
  training.do_train=false training.do_eval=true num_valid_samples=2 use_wandb=false

python src/scripts/finetune_dlb_binary_classification.py task=eqtl \
  ckpt=./pretrained/GeneZip-70M max_train_records=16 max_valid_records=16 \
  max_test_records=16 +training.max_steps=2 eval_steps=1 save_steps=1 use_wandb=false

python src/scripts/finetune_dlb_binary_classification.py task=etgp \
  ckpt=./pretrained/GeneZip-70M max_train_records=16 max_valid_records=16 \
  max_test_records=64 +training.max_steps=2 eval_steps=1 save_steps=1 use_wandb=false

python src/scripts/finetune_dlb_cmp.py task=cmp \
  ckpt=./pretrained/GeneZip-70M num_train_samples=2 num_valid_samples=1 \
  num_test_samples=1 batch_size=1 eval_batch_size=1 train_steps=2 eval_steps=1 use_wandb=false

python src/scripts/finetune_dlb_tisp.py task=tisp \
  ckpt=./pretrained/GeneZip-70M num_train_samples=8 num_valid_samples=2 \
  num_test_samples=2 train_steps=2 eval_steps=1 use_wandb=false

python src/scripts/pretrain_genezip.py task=pretrain \
  data=gencode_human_128k max_len=128000 pretrained_ckpt=./pretrained/GeneZip-70M \
  num_train_samples=2 num_valid_samples=1 max_train_steps=1 eval_steps=1 \
  training.save_strategy=no training.warmup_steps=0 use_wandb=false

```

The first positive ETGP test example occurs at index 53 in this pinned dataset;
the smoke uses 64 test records so AUROC/AUPRC are defined. Complete benchmark
recipes keep the full splits. Short-run caps and initialization losses must not be
reported as reproduced benchmark scores.
