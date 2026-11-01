# Camera Ready incremental release validation

Base release: `d3b7d9d74b10ede08eedc2a6300feb38b4c5aaed`.
Code candidate: `cr_release`; this record is versioned with the reviewed code. Pretrained weight
publication was authorized on 2026-11-01; published revisions are recorded in
`checkpoints.json`.
Only GeneZip pretrained weights and pretraining/CMP/eQTL/ETGP/TISP code are in scope.

## Verified artifacts

- All seven models listed in `checkpoints.json:release_models` were staged on Mila
  and matched the pinned source weight SHA-256 (job `11474942`). The export keeps
  only architecture, routing and scientific metadata, without source runtime fields.
- All seven staged models passed strict loading and finite CUDA forward checks
  at 2,048 bp with their configured routing classes (`11475019`, A100, `cn-g006`).
- GeneZip-70M passed the frozen-parameter/backward/save-reload smoke (`11474942`).
- Seven CPU contract checks passed in 40.5 seconds (`11475103`). Its subsequent
  shared-128K-data probe failed; that job is not an end-to-end data success.
- 70M completed two 12.8K training steps in the first training batch (`11475012`).
  A launcher log-close I/O error interrupted its sibling 636M run. The isolated
  636M rerun completed two steps and evaluation (`11475031`). Both checks used
  batch 1, four training records and two validation records.
- Independent evaluation of staged GeneZip-70M completed on two validation
  records (`11475039`): PPL 2.7793 and 175.3288 average bp/token. These two-record
  values are smoke evidence, not estimates of the paper's aggregate results.

- Real-length eQTL/ETGP 450K smoke completed (`11475160`, two A100s, `cn-g026`):
  two optimizer steps, validation AUROC selection, checkpoint reload and test evaluation.
  eQTL used 16/16/16 train/valid/test records (test AUROC 0.5, AUPRC 0.1833333).
  ETGP used 16/16/64 records (test AUROC 0.0634921, AUPRC 0.0166667). These tiny,
  two-step runs verify the pipeline only.

- Initial TISP 100K smoke completed (`11475219`, A100; superseded by the
  protocol-corrected check below): two updates, validation selection
  and test evaluation, using eight training windows and two windows per held-out
  split. Test average PCC was -0.0015. Subsetting invokes the retained window-level
  smoke evaluator; the full chromosome-level benchmark has not been rerun here.

- Both 70M and 636M completed 128K continuation from their same-size 12.8K
  staged checkpoints: one optimizer step plus evaluation, two training records
  and one validation record. N=64 Regulome and PromoterDistal also completed two
  12.8K training steps plus evaluation (`11475239`, four independent A100 workers,
  `cn-g017`, 1 minute 49 seconds). These checks did not save new pretrained weights.
- Public dataset acceptance completed without the shared research data cache
  (`11475147`): CMP 7,008/419/413 train/validation/test records; GENCODE 12.8K
  950,000 train and 50,000 valid; GENCODE 128K 99,000 train and 1,000 valid.
  eQTL, ETGP and TISP public task files were downloaded and consumed by their
  GPU smoke runs. The original GENCODE valid splits were preserved.

- CMP HFF completed real 1,048,576-bp input embedding extraction, two head
  updates and validation/test evaluation (`11475246`, A100, `cn-g023`, 2 minutes
  48 seconds). The test used one record: SCC -0.0339, Corr -0.0165. The checkpoint
  directory was deliberately named `routing-check`, verifying that routing no
  longer depends on a `GeneZip` basename. CMP retains its existing in-run
  evaluation protocol; this smoke does not claim saved-head reload coverage.

## Commands

Run on an allocated NVIDIA GPU compute node, from the release repository root.
Public dataset revisions are pinned in `configs/data.yaml` and `configs/main.yaml`.

```sh
python -m unittest discover -s tests -v
python scripts/export_pretrained.py --model GeneZip-70M --output ./pretrained
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

## Adopted-run consistency audit

All 16 main-task W&B runs were re-read (`11476062`); their held-out metrics match
this release's adopted results. Full retraining is not required for this release.
The [audit snapshot](sources/adopted_main_run_audit.json) records configs, metadata
argv, source commits, machine provenance and metric values.

Two protocol mismatches were corrected: CMP evaluates every 200 steps, and TISP
uses floor plus ceiling with caps `[60000,30000]`. TISP's extra float cast before
post-hoc exponentiation was removed to retain the adopted evaluation dtype. The
chromosome segment/mask/accumulation logic matches the adopted implementation.

The corrected TISP 100K pipeline passed two optimizer steps, validation-based
checkpoint selection/reload and test evaluation in job `11476130` (A100,
`cn-g001`). Window-level test average PCC was -0.0016 on two test windows.
The seven contract checks passed after the protocol updates (`11476117`, test
phase); that job's subsequent TISP script failed on an indentation error, which
was corrected before the successful `11476130` rerun.

## Validation limits

All jobs above have reached terminal states; no validation allocation remains
running. The 1+2+4 A100 resource plan was used in separate short allocations,
with independent single-GPU workers rather than seven-way distributed training.

The candidate used the existing Mila environment with isolated code and public
data caches. The user explicitly waived fresh dependency installation and formal-scale
runtime checks. Full retraining is replaced by consistency checks against adopted
Rebuttal runs; the smoke results are not newly reproduced benchmark scores. Full TISP chromosome
scoring, all CMP cell lines and all eQTL tissues remain covered by inherited
recipes and historical result provenance, not newly completed full runs.
This candidate is intended for branch review before the public code release.
Pretrained weight publication and its anonymous-download/load evidence are
tracked separately below.

## Public pretrained weights

All seven canonical model repositories were published in job `11476123`.
Anonymous (`token=False`) downloads from a new HTTP cache matched every source
SHA-256 and the exact exported Hydra configuration. Published repository IDs and
immutable revisions are recorded in `checkpoints.json` and
[the publication record](sources/published_pretrained_models.json).

Job `11476167` then resolved each public repo at its pinned revision through the
release checkpoint resolver and strict-loaded the downloaded files on two A100s
(`main`, `cn-g006`). All seven produced finite 2,048-bp CUDA forward outputs with
their configured routing classes. The job completed successfully in 4 minutes
39 seconds. All publication and verification jobs have ended.
