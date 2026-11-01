# Camera Ready checkpoint and task mapping

Numerical provenance snapshot: **V6.13**. The main task values were cross-checked
against the current **V7.13** source tables on 2026-11-01 and agree. The code is a `cr_release` review candidate; the seven pretrained models are now public on Hugging Face.
The machine-readable source is [`checkpoints.json`](checkpoints.json). Historical repo IDs below are immutable source provenance; they are not runtime compatibility aliases.

## Pretrained models

| Public name | Context | N | Source checkpoint | Original pin evidence |
| --- | ---: | ---: | --- | --- |
| `GeneZip-70M` | 12,800 | 32 | `jzshared/GeneZip-7R-LT-32BPT-100B@40a4039a589288939c551b3c330a1c3562b2b15a/checkpoint-30600` | paper_run_pin |
| `GeneZip-70M-128K` | 128,000 | 32 | `jzshared/GeneZip-7R-LT-32BPT-128K-100B@08b76c09d7bd5e76cf30ade326838cee7b32d677/checkpoint-24414` | paper_run_pin |
| `GeneZip-636M` | 12,800 | 32 | `jzshared/GeneZip-7R-LT-32BPT-100B-650M@4bff03564e32f77698630eb64d5361e6eccb1822/checkpoint-30600` | hub_metadata_pin_pending_tensor_audit |
| `GeneZip-636M-128K` | 128,000 | 32 | `jzshared/GeneZip-7R-LT-32BPT-128K-100B-650M@baf1cdefd5e7ef751191a1440a5b61d018cf6458/checkpoint-24414` | hub_metadata_pin_pending_tensor_audit |
| `GeneZip-Regulome` | 12,800 | 64 | `jzshared/GeneZip-CisReg-64BPT-100B@e8c84abc9ae814f296f04efbdc09410c18c92a81/checkpoint-30600` | paper_run_pin |
| `GeneZip-PromoterDistal` | 12,800 | 64 | `jzshared/GeneZip-PromoterDistal-64BPT-100B@1520f073a6765d0ae338ede823e32e4b501da67f/checkpoint-30600` | paper_run_pin |
| `GeneZip-Intergenic` | 12,800 | 32 | `jzshared/GeneZip-AntiBioPrior-BPT32-12.8K@3b9c82577ee4159ef6ab3421f4bca45ab2fee5f8/checkpoint-30600` | hub_metadata_pin_pending_tensor_audit |

Only GeneZip pretrained weights are in the release scope; task checkpoints are not published.

All seven CR model repositories are public under `andyjzhao/`, using the exact
names above. `release_revision` pins the published commit in `checkpoints.json`;
`source.weights_sha256` and `weights_sha256` verify unchanged source/export bytes.
The original pin-evidence column records provenance from before publication.
The [publication record](sources/published_pretrained_models.json) lists all seven
public URLs, immutable revisions and anonymous-download verification results.
A Hub metadata pin alone does not certify equivalence to a historical paper run.

`GeneZip` in paper prose means `GeneZip-70M`. Region order is promoter, CDS, UTR, exon, intron, NIG, DIG. Regulome and PromoterDistal require the **N=64** sources above; renaming their old N=32 weights is incorrect.

The [64-row task/checkpoint CSV](tables/checkpoint_task_map.csv) gives one row per adopted task run, with canonical identity, pretrained source, W&B run, and verified weight path where available.

## Main task matrix

Every row starts from the same `GeneZip-70M` pretrained weights; task heads and adapted weights differ.

| Task | Input bp | Held-out target | Protocol |
| --- | ---: | --- | --- |
| CMP | 1,048,576 (12,800 encoder chunks) | SCC 0.1332 / Corr 0.1791 | Frozen representations, mean pooling, 5,000 head updates, evaluation every 200 steps; all five cell lines |
| eQTL | 450,000 | AUROC 0.8172332642 / AUPRC 0.1995326131 | Seed 1; frozen embeddings/encoder/router; no LoRA; 3 epochs; LR 2e-4; best validation AUROC; nine tissues |
| ETGP | 450,000 | AUROC 0.8228846154 / AUPRC 0.3746280992 | Same adaptation protocol as eQTL; K562; best step 300 |
| TISP | 100,000 | Mean PCC 0.2741 | Seed 0; full fine-tuning; WD 0.01; routing caps [60000,30000]; 5,000 steps; chromosome-level validation selection; five assays |

The main eQTL/ETGP recipe explicitly uses no routing floor and caps of `[70000,70000]`, matching the adopted `legacy_ceiling_only` execution. This is a task-specific setting, not the pretraining floor of 8. Changing it requires a new numerical comparison. The downstream compression regularizer targets **128 BPT** with alpha 0.03 (verified at source commit `edc537a`); the pretraining N=32 anchor must not be reused as this task target.

### Adopted run inventory

| Task | Subset | Evidence run |
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

The JSON retains recorded code and execution provenance for eQTL/ETGP, with
historical source records in `sources/`. Historical task IDs and saved-weight
inventories identify adopted runs only; they are not publication targets or
required files for this release. Reproduction trains task heads from the released
pretrained weights using the recipes above. No downstream checkpoint recovery or
publication is required.

## Regional-prior matrix

Paper Table 12 is transcribed in [`tables/regional_priors.csv`](tables/regional_priors.csv). The JSON includes every alternative-prior × task cell, its pretrained source, expected metrics and adopted run/sweep evidence. All **48** regional-prior runs have been read from live W&B and are indexed in [`sources/regional_task_run_inventory.json`](sources/regional_task_run_inventory.json), including per-run task/subset IDs and recorded configuration. Together with the 16 main runs, this covers 64 adopted task runs. This establishes result provenance; recipe execution remains a separate check.

| Prior | eQTL AUROC/AUPRC | ETGP AUROC/AUPRC | CMP SCC/Corr | TISP PCC |
| --- | --- | --- | --- | --- |
| GeneZip-Regulome | 0.8343 / 0.2041 | 0.7888 / 0.3395 | 0.1013 / 0.1482 | 0.2637 |
| GeneZip-PromoterDistal | 0.8142 / 0.1743 | 0.7523 / 0.2586 | 0.0932 / 0.1416 | 0.2507 |
| GeneZip-Intergenic | 0.7865 / 0.2058 | 0.8225 / 0.4464 | 0.1280 / 0.1735 | 0.2238 |

These are separately configured experiments, not a loop that changes only `CKPT`. In particular:

- N=64 TISP uses WD 0.001; the default main-model recipe uses 0.01.
- Regulome ETGP uses seed 1, full fine-tuning and the final checkpoint (`kcgnr42g`), not the later LoRA tuning winner.
- N=64 CMP adopts the final fresh-cache runs from `v4udhiv6`.
- Regulome eQTL: all nine runs from `ey3eta02` are verified against live W&B; mean AUROC 0.834264663 and AUPRC 0.204068027 match the paper.
- PromoterDistal ETGP: `24qit8ln`, 12.8K input, seed 0, LoRA, LR 1e-3, 15 epochs, best validation AUPRC.
- Intergenic ETGP: `css2g9wf`, seed 1, full fine-tuning, no LoRA, final checkpoint.
- PromoterDistal/Intergenic eQTL: all 18 adopted runs from `vpucss4z` now match the paper; means are 0.814182431/0.174298521 and 0.786497922/0.205832172. Each tissue is recorded in the JSON. Historical sweeps with different means are excluded.

## Validation boundaries

- All seven canonical GeneZip models are published with verified source SHA-256. Use the public repo IDs or an immutable revision from the manifest; historical-source export remains available.
- GeneZip-70M source strict loading passed on Mila A100 (job `11414821`, `cn-g025`, `long`): finite forward/backward at 2,048 bp, 43 frozen tensors without gradients, exact save/reload output equality. All seven staged models subsequently passed strict loading and finite 2048-bp forward checks with their configured routing classes (job 11475019, cn-g006, unkillable).
- Seven contract checks passed in job `11475103`, including all eight shell entrypoints, Hydra composition, public Parquet loading and partial-download recovery. See [release validation](reproduce_release.md) for GPU and public-data evidence. The user waived clean installation and formal-scale runtime validation; full retraining is replaced by adopted-run consistency checking.
- The 16 adopted main-task runs were re-read from live W&B. Config, metadata argv, source scoring protocol and held-out metrics agree after the CMP cadence and TISP routing fixes. See [the audit](sources/adopted_main_run_audit.json). This is a provenance/protocol check, not a new full training reproduction.
- The pinned 636M/12.8K source config confirms caps `[3000,3000]`; 128K uses `[30000,10000]`. The templates retain these source settings. Regulome, PromoterDistal and Intergenic use `[5000,2000]`.
- Source metadata and public API names can differ without maintaining runtime aliases.
- BPT 137.59 in the main table and 139.35 in the matched-panel table are different measurement protocols.
- No withdrawn downstream length-ablation, RSAP, or unverified task-checkpoint claim is part of this release.
