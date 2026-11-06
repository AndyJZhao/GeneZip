# Pretrained models and benchmark results

The models below provide the pretrained weights for GeneZip and its regional-prior
ablations. Commands and protocols are in the [reproduction guide](reproduce_release.md).

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

Download commands are in the
[reproduction guide](reproduce_release.md#model-downloads).

`GeneZip` in the paper refers to `GeneZip-70M`. Region-wise targets follow the
order promoter, CDS, UTR, exon, intron, NIG and DIG.

## DNALongBench results

| Model | BPT | Inference (s) | CMP SCC | eQTL AUROC | ETGP AUPRC | TISP PCC | Avg. rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HyenaDNA-6.6M | 1.0 | 22.86 | 0.115 | 0.514 | 0.249 | 0.132 | 8.00 |
| Caduceus-Ph-7.7M | 1.0 | 35.06 | 0.133 | 0.566 | 0.380 | 0.109 | 5.63 |
| Caduceus-PS-7.7M | 1.0 | 68.36 | 0.127 | 0.538 | 0.376 | 0.108 | 7.00 |
| JanusDNA-7.7M | 1.0 | 228.63 | 0.159 | 0.840 | 0.438 | 0.002 | 3.25 |
| NTv3-100M-pre | 127.99 | 13.27 | 0.136 | 0.792 | 0.440 | 0.194 | 2.75 |
| Evo2-1B | 1.0 | OOM | 0.018 | 0.754 | 0.095 | -0.002 | 9.25 |
| Mamba2-84M | 1.0 | 40.80 | 0.111 | 0.762 | 0.322 | 0.294 | 6.25 |
| U-Net-70M | 127.99 | 4.85 | 0.131 | 0.807 | 0.390 | 0.178 | 4.00 |
| H-Net-70M | 122.19 | 11.78 | 0.118 | 0.791 | 0.225 | 0.303 | 5.75 |
| GeneZip-70M | 137.59 | 6.34 | 0.133 | 0.817 | 0.375 | 0.336 | 3.13 |

Inference times cover 100 × 450-kb inputs on one H100 80GB with model-specific
batches; see the [timing protocol](reproduce_release.md#inference-throughput).
The CSV records the three pretraining groups used in the paper.

All four tasks start from the same `GeneZip-70M` pretrained weights; task heads
and adapted weights differ. See [task protocols](reproduce_release.md#task-protocols)
and [task subsets](reproduce_release.md#task-subsets).

| Task | Held-out result |
| --- | --- |
| CMP | SCC 0.1332 / Corr 0.1791 |
| eQTL | AUROC 0.8172332642 / AUPRC 0.1995326131 |
| ETGP | AUROC 0.8228846154 / AUPRC 0.3746280992 |
| TISP | Mean PCC 0.3355521527 |

The reported ETGP checkpoint was selected at step 300.

## Regional-prior ablations

| Prior | eQTL AUROC/AUPRC | ETGP AUROC/AUPRC | CMP SCC/Corr | TISP PCC |
| --- | --- | --- | --- | --- |
| GeneZip-Regulome | 0.8229 / 0.1978 | 0.8290384615 / 0.5049418029 | 0.1013 / 0.1482 | 0.3278238692 |
| GeneZip-PromoterDistal | 0.8241 / 0.1847 | 0.8086538462 / 0.4893137255 | 0.0932 / 0.1416 | 0.3376757999 |
| GeneZip-Intergenic | 0.7865 / 0.2058 | 0.8184615385 / 0.4230425843 | 0.1280 / 0.1735 | 0.2654333989 |

See the [reproduction guide](reproduce_release.md#task-protocols) for the common
adaptation protocol and [model selection](reproduce_release.md#task-selection).

The [result CSV](tables/regional_priors.csv) contains the aggregate scores.

## Reproducibility

[Reproduction instructions](reproduce_release.md) describe
the published task protocols.
The BPT values 137.59 in the main table and 139.35 in the matched-panel table
come from different measurement protocols.

The main-table average rank uses CMP SCC, eQTL AUROC, ETGP AUPRC and TISP
PCC across all ten displayed models, with average ranks for ties at the
three-decimal reporting precision. GeneZip's average rank is 3.13, compared
with U-Net 4.00, H-Net 5.75, Mamba2 6.25 and NTv3 2.75. GeneZip has the lowest
average rank among compression models with leakage-controlled pretraining.
eQTL AUPRC is an auxiliary metric.
