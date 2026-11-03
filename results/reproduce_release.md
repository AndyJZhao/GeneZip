# Reproducing the published results

Model scores and task protocols are listed in
[model details and benchmark results](CHECKPOINTS.md). Use the full training
and evaluation commands in the [README](../README.md#dnalongbench-training-and-evaluation).
The wrappers default to the published training budgets and task splits.

Public dataset revisions are pinned in `configs/data.yaml` and `configs/main.yaml`.
TISP benchmark results use chromosome-level PCC across FC, EC, ER, GC and PC.
For interrupted final evaluation, reuse the validation-selected task checkpoint
with the [TISP evaluation recovery command](../README.md#dnalongbench-training-and-evaluation).

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
