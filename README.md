<div align="center">

# GeneZip: Region-Aware Compression for Long-Context DNA Modeling

[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![arXiv 2602.17739](https://img.shields.io/badge/arXiv-2602.17739-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.17739)
[![Config: Hydra 1.3](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Paper: NeurIPS 2026](https://img.shields.io/badge/Paper-NeurIPS_2026-5067b7)](https://openreview.net/pdf?id=E37ysCRa5d)
[![Hugging Face models](https://img.shields.io/badge/🤗_Models-Hugging_Face-ffd21e)](#pretrained-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[Project page](https://andyjzhao.github.io/GeneZip/) · [Paper](https://openreview.net/pdf?id=E37ysCRa5d) · [Models](#pretrained-models) · [Getting started](#environment) · [Citation](#citation)

</div>

Official PyTorch implementation of GeneZip, a framework for region-aware DNA compression.

Jianan Zhao, Xixian Liu, Zhihao Zhan, Xinyu Yuan, Hongyu Guo, and Jian Tang.

## Abstract

Long-context DNA modeling is computationally expensive, motivating existing approaches to reduce sequence length through uniform downsampling. However, functional information and sequence redundancy are unevenly distributed across the genome, calling for finer resolution in functionally dense regions and stronger compression of redundant sequence. We introduce GeneZip, a region-aware DNA compression framework that learns where to allocate a limited token budget. Building on H-Net-style dynamic routing, GeneZip combines a Region-Aware Ratio objective, which uses regional labels derived from GENCODE gene annotations to specify compression targets during training, with bounded routing for stable token counts. At inference, it compresses raw DNA without annotations. GeneZip provides three main benefits. First, it preserves predictive utility under high compression: GeneZip-70M achieves 137.6-fold compression with a validation perplexity increase of 0.31 relative to uncompressed Mamba2-84M, and the best average rank across four DNALongBench tasks among compression models with leakage-controlled pretraining. Second, it learns region-aware allocation with emergent repeat-aware compression. Alternative regional priors yield task-dependent performance trade-offs. Without repeat-specific supervision, GeneZip compresses transposable-element-derived and tandem repeats more strongly than matched non-repeat sequence; its tandem-repeat compression preference transfers to a nine-species evaluation panel without further training. Third, it enables efficient long-context modeling: on 450-kb inputs, GeneZip-70M achieves 36.1-fold higher inference throughput than JanusDNA-7.7M on a single H100 GPU.

![GeneZip overview](asset/GeneZip.png)

## Results

GeneZip-70M uses the same pretrained weights across all four tasks, with
separately trained task heads and adapted weights.

| CMP SCC | eQTL AUROC | ETGP AUPRC | TISP PCC | Average rank |
| ---: | ---: | ---: | ---: | ---: |
| 0.1332 | 0.8172 | 0.3746 | 0.3356 | 3.13 |

The average rank is computed across all ten models using the four task
scores rounded to three decimals, with average ranks for ties. GeneZip leads
among **compression models with leakage-controlled pretraining**; the
[complete comparison](results/CHECKPOINTS.md#dnalongbench-results) includes
external-pretraining models.

The **36.1× batched inference throughput** result processes 100 sequences of 450,000 bp
on one H100 80GB with BF16: 6.34 s for GeneZip-70M (batch 7), versus 228.63 s
for JanusDNA-7.7M (batch 9). See the [timing protocol](results/reproduce_release.md#inference-throughput).

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

Linux x86_64, an NVIDIA GPU and a CUDA 12.6-compatible driver/toolkit are
required for model execution. The software has been tested on A100 80GB GPUs.
The lockfile uses Python 3.12, PyTorch 2.8/CUDA 12.6, flash-attn 2.8.3, mamba-ssm 2.3.0 and
causal-conv1d 1.6.0.

```sh
git clone https://github.com/AndyJZhao/GeneZip.git
cd GeneZip
uv venv --python 3.12
uv sync
source .venv/bin/activate
```

Run shell entrypoints from the repository root.

## DNALongBench training and evaluation

All four wrappers default to the public `andyjzhao/GeneZip-70M` weights.
Their default commands download the checkpoint automatically:

```sh
bash shell/dnalongbench/cmp.sh
bash shell/dnalongbench/eqtl.sh
bash shell/dnalongbench/etgp.sh
bash shell/dnalongbench/tisp.sh
```

See the [reproduction guide](results/reproduce_release.md) for model downloads,
task subsets, training protocols, checkpoint evaluation and
[pretraining templates](results/reproduce_release.md#pretraining-templates).

## Citation

If you use GeneZip in your research, please cite:

```bibtex
@inproceedings{zhao2026genezip,
  title = {GeneZip: Region-Aware Compression for Long-Context DNA Modeling},
  author = {Zhao, Jianan and Liu, Xixian and Zhan, Zhihao and Yuan, Xinyu and Guo, Hongyu and Tang, Jian},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://openreview.net/pdf?id=E37ysCRa5d}
}
```

## License

GeneZip-authored code is licensed under [MIT](LICENSE).
