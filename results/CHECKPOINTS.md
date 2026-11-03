# Pretrained models and benchmark results

The models below provide the pretrained weights for GeneZip and its regional-prior
ablations. Training and evaluation commands are in the [README](../README.md).

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

Download commands, including a fixed-revision example, are in the
[README](../README.md#dnalongbench-training-and-evaluation).

`GeneZip` in the paper refers to `GeneZip-70M`. Region-wise targets follow the
order promoter, CDS, UTR, exon, intron, NIG and DIG.

## DNALongBench results

Every row starts from the same `GeneZip-70M` pretrained weights; task heads and adapted weights differ.

| Task | Input bp | Held-out result | Protocol |
| --- | ---: | --- | --- |
| CMP | 1,048,576 (12,800 encoder chunks) | SCC 0.1332 / Corr 0.1791 | Frozen representations, mean pooling, 5,000 head updates, evaluation every 200 steps; all five cell lines |
| eQTL | 450,000 | AUROC 0.8172332642 / AUPRC 0.1995326131 | Seed 1; frozen embeddings/encoder/router; no LoRA; 3 epochs; LR 2e-4; WD 0.1; best validation AUROC; nine tissues |
| ETGP | 450,000 | AUROC 0.8228846154 / AUPRC 0.3746280992 | Same adaptation protocol as eQTL; K562; best step 300 |
| TISP | 100,000 | Mean PCC 0.3391655900 | Seed 0; frozen embeddings/encoder/router; WD 0.1; native routing floor, no routing ceiling; 5,000 steps; chromosome-level validation selection; five assays |

The eQTL/ETGP recipes disable the routing floor and use caps of `[70000,70000]`.
Their compression regularizer targets **128 BPT**, with alpha 0.03. These are
fine-tuning settings; the pretrained model uses an N=32 anchor and routing floors
of 8 tokens.

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

## Regional-prior ablations

| Prior | eQTL AUROC/AUPRC | ETGP AUROC/AUPRC | CMP SCC/Corr | TISP PCC |
| --- | --- | --- | --- | --- |
| GeneZip-Regulome | 0.8343 / 0.2041 | 0.8290384615 / 0.5049418029 | 0.1013 / 0.1482 | 0.3277016163 |
| GeneZip-PromoterDistal | 0.8142 / 0.1743 | 0.8086538462 / 0.4893137255 | 0.0932 / 0.1416 | 0.3378432474 |
| GeneZip-Intergenic | 0.7865 / 0.2058 | 0.8184615385 / 0.4230425843 | 0.1280 / 0.1735 | 0.2706577646 |

ETGP and TISP now use the same fixed adaptation protocol as GeneZip-70M:
LR 2e-4, WD 0.1 and frozen embeddings, encoders and routers. ETGP uses 450K
inputs, seed 1, three epochs, caps `[70000,70000]`, no routing floor and best
validation AUROC. TISP uses 100K inputs, seed 0, 5000 steps, the checkpoint's
native routing floor, no ceiling and best chromosome-level validation PCC.
Select the regional model with `CKPT=GeneZip-Regulome`,
`CKPT=GeneZip-PromoterDistal` or `CKPT=GeneZip-Intergenic` in the corresponding
ETGP/TISP wrapper. eQTL and CMP entries retain their previous experiment
protocols. Other historical baseline rows are not all certified as sharing
the fixed adaptation protocol.

The [result CSV](tables/regional_priors.csv) contains the aggregate scores.

## Reproducibility

[Validation coverage and quick-check commands](reproduce_release.md) describe
which execution paths have been tested.
The BPT values 137.59 in the main table and 139.35 in the matched-panel table
come from different measurement protocols.

The main-table average rank uses CMP SCC, eQTL AUROC, ETGP AUPRC and TISP
PCC across all ten displayed models, with average ranks for ties at the
three-decimal reporting precision. GeneZip's average rank is 3.13, compared
with U-Net 4.25, H-Net 5.25, Mamba2 6.25 and NTv3 3.00. The claim is restricted
to compression models with leakage-controlled pretraining. eQTL AUPRC is
an auxiliary metric. The U-Net and H-Net scores are comparison results;
this repository does not provide their baseline training implementations.
