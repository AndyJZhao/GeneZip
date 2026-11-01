# Pretrained models and benchmark results

The [model manifest](checkpoints.json) records model configurations, fixed download
revisions, weight checksums and the experiments behind the results below.

## Pretrained models

| Model | Context (bp) | BPT anchor (N) | Purpose |
| --- | ---: | ---: | --- |
| [GeneZip-70M](https://huggingface.co/andyjzhao/GeneZip-70M) | 12,800 | 32 | Default model for the four downstream tasks |
| [GeneZip-70M-128K](https://huggingface.co/andyjzhao/GeneZip-70M-128K) | 128,000 | 32 | Context scaling |
| [GeneZip-636M](https://huggingface.co/andyjzhao/GeneZip-636M) | 12,800 | 32 | Model scaling |
| [GeneZip-636M-128K](https://huggingface.co/andyjzhao/GeneZip-636M-128K) | 128,000 | 32 | Model and context scaling |
| [GeneZip-Regulome](https://huggingface.co/andyjzhao/GeneZip-Regulome) | 12,800 | 64 | Regional-prior ablation |
| [GeneZip-PromoterDistal](https://huggingface.co/andyjzhao/GeneZip-PromoterDistal) | 12,800 | 64 | Regional-prior ablation |
| [GeneZip-Intergenic](https://huggingface.co/andyjzhao/GeneZip-Intergenic) | 12,800 | 32 | Regional-prior ablation |

Use `release_revision` in the manifest to download an exact model version and
`weights_sha256` to verify its weights. Download commands are in the
[README](../README.md#dnalongbench-training-and-evaluation).

`GeneZip` in the paper refers to `GeneZip-70M`. Region-wise targets follow the
order promoter, CDS, UTR, exon, intron, NIG and DIG.

## DNALongBench results

Every row starts from the same `GeneZip-70M` pretrained weights; task heads and adapted weights differ.

| Task | Input bp | Held-out result | Protocol |
| --- | ---: | --- | --- |
| CMP | 1,048,576 (12,800 encoder chunks) | SCC 0.1332 / Corr 0.1791 | Frozen representations, mean pooling, 5,000 head updates, evaluation every 200 steps; all five cell lines |
| eQTL | 450,000 | AUROC 0.8172332642 / AUPRC 0.1995326131 | Seed 1; frozen embeddings/encoder/router; no LoRA; 3 epochs; LR 2e-4; best validation AUROC; nine tissues |
| ETGP | 450,000 | AUROC 0.8228846154 / AUPRC 0.3746280992 | Same adaptation protocol as eQTL; K562; best step 300 |
| TISP | 100,000 | Mean PCC 0.2741 | Seed 0; full fine-tuning; WD 0.01; routing caps [60000,30000]; 5,000 steps; chromosome-level validation selection; five assays |

The eQTL/ETGP recipes disable the routing floor and use caps of `[70000,70000]`.
Their compression regularizer targets **128 BPT**, with alpha 0.03. These are
fine-tuning settings; the pretrained model uses an N=32 anchor and routing floors
of 8 tokens.

### Experiment records

| Task | Subset | W&B run |
| --- | --- | --- |
| eqtl | Adipose_Subcutaneous | https://wandb.ai/jzshared/DNALLM_eQTL/runs/37nwc0hq |
| eqtl | Artery_Tibial | https://wandb.ai/jzshared/DNALLM_eQTL/runs/9syr5awo |
| eqtl | Cells_Cultured_fibroblasts | https://wandb.ai/jzshared/DNALLM_eQTL/runs/tp8fplr0 |
| eqtl | Muscle_Skeletal | https://wandb.ai/jzshared/DNALLM_eQTL/runs/bruz076f |
| eqtl | Nerve_Tibial | https://wandb.ai/jzshared/DNALLM_eQTL/runs/9t4u2w5p |
| eqtl | Skin_Not_Sun_Exposed_Suprapubic | https://wandb.ai/jzshared/DNALLM_eQTL/runs/xl2b1h0w |
| eqtl | Skin_Sun_Exposed_Lower_leg | https://wandb.ai/jzshared/DNALLM_eQTL/runs/20a8poj5 |
| eqtl | Thyroid | https://wandb.ai/jzshared/DNALLM_eQTL/runs/5avmvngd |
| eqtl | Whole_Blood | https://wandb.ai/jzshared/DNALLM_eQTL/runs/1qa7ifsh |
| etgp | K562 | https://wandb.ai/jzshared/DNALLM_ETGP/runs/8a4nrecm |
| cmp | HFF | https://wandb.ai/jzshared/DLB-CMP/runs/8pbzcpqg |
| cmp | H1hESC | https://wandb.ai/jzshared/DLB-CMP/runs/j508744t |
| cmp | GM12878 | https://wandb.ai/jzshared/DLB-CMP/runs/6516bst3 |
| cmp | IMR90 | https://wandb.ai/jzshared/DLB-CMP/runs/2krrvu3o |
| cmp | HCT116 | https://wandb.ai/jzshared/DLB-CMP/runs/1ved4mm9 |
| tisp | FC / EC / ER / GC / PC | https://wandb.ai/jzshared/GeneZip-TISP/runs/m6tlxgu1 |

The [experiment index](tables/checkpoint_task_map.csv) links each task and subset
to its pretrained model and W&B run. Train task-specific checkpoints from the
pretrained weights with the commands in the README.

## Regional-prior ablations

| Prior | eQTL AUROC/AUPRC | ETGP AUROC/AUPRC | CMP SCC/Corr | TISP PCC |
| --- | --- | --- | --- | --- |
| GeneZip-Regulome | 0.8343 / 0.2041 | 0.7888 / 0.3395 | 0.1013 / 0.1482 | 0.2637 |
| GeneZip-PromoterDistal | 0.8142 / 0.1743 | 0.7523 / 0.2586 | 0.0932 / 0.1416 | 0.2507 |
| GeneZip-Intergenic | 0.7865 / 0.2058 | 0.8225 / 0.4464 | 0.1280 / 0.1735 | 0.2238 |

These results use task-specific fine-tuning settings. The main-model wrappers
do not reproduce these experiments by changing `CKPT` alone. In particular:

- Regulome and PromoterDistal TISP use weight decay 0.001.
- Regulome ETGP uses seed 1, full fine-tuning and final-checkpoint evaluation.
- PromoterDistal ETGP uses 12.8K inputs, seed 0, LoRA, learning rate 1e-3,
  15 epochs and best validation AUPRC selection.
- Intergenic ETGP uses seed 1, full fine-tuning and final-checkpoint evaluation.

The [ablation experiment records](sources/regional_task_run_inventory.json)
contain configurations for each task and subset. The [result CSV](tables/regional_priors.csv)
contains the aggregate scores.

## Reproducibility

[Validation coverage and quick-check commands](reproduce_release.md) describe
which execution paths have been tested. The [main-task experiment records](sources/adopted_main_run_audit.json)
contain the configurations and held-out metrics behind the reported results.
The BPT values 137.59 in the main table and 139.35 in the matched-panel table
come from different measurement protocols.
