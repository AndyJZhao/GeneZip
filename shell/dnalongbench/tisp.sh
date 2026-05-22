#!/bin/bash
# Canonical entrypoint: `bash shell/dnalongbench/tisp.sh`.
# NOTE: The SBATCH lines below are examples for Slurm; edit/remove for your cluster.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PROJECT_ROOT

export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/temp/wandb/}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_ROOT}/temp/wandb_cache/}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/temp/hf_home/}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/temp/xdg_cache/}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_ROOT}/temp/triton_cache/}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/temp/tmp/}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PROJECT_ROOT}/temp/torch_extensions/}"

mkdir -p \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TMPDIR}" \
  "${TORCH_EXTENSIONS_DIR}"
cd "${PROJECT_ROOT}"

WANDB_PROJ="${WANDB_PROJ:-GeneZip-TISP}"
USE_WANDB="${USE_WANDB:-false}"

CKPT="${CKPT:-GeneZip-70M-Promoter-distal-regulatory}"
HF_USER="${HF_USER:-andyjzhao}"
DATA_ROOT="${DATA_ROOT:-./data/dnalongbench}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-50}"
NUM_TRAIN_SAMPLES="${NUM_TRAIN_SAMPLES:-0}"
NUM_VALID_SAMPLES="${NUM_VALID_SAMPLES:-0}"
NUM_TEST_SAMPLES="${NUM_TEST_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-4}"
LR="${LR:-2e-4}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  RUNNER="accelerate launch --num_processes ${NUM_PROCESSES} --main_process_port 0 --mixed_precision ${MIXED_PRECISION}"
else
  RUNNER="python"
fi

WANDB_ARGS="use_wandb=false"
if [[ "${USE_WANDB}" == "true" ]]; then
  WANDB_ARGS="use_wandb=true wandb.project=${WANDB_PROJ}"
fi

CMD_TO_RUN="${RUNNER} src/scripts/finetune_dlb_tisp.py \
  task=tisp root=${DATA_ROOT} \
  ckpt=${CKPT} hf_user=${HF_USER} \
  train_steps=${TRAIN_STEPS} eval_steps=${EVAL_STEPS} \
  num_train_samples=${NUM_TRAIN_SAMPLES} num_valid_samples=${NUM_VALID_SAMPLES} num_test_samples=${NUM_TEST_SAMPLES} \
  batch_size=${BATCH_SIZE} eval_batch_size=${EVAL_BATCH_SIZE} grad_acc_steps=${GRAD_ACC_STEPS} \
  learning_rate=${LR} log_every=${LOG_EVERY} freeze_encoder=${FREEZE_ENCODER} \
  ${WANDB_ARGS}"

echo "Running command:"
echo "${CMD_TO_RUN}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi
eval "${CMD_TO_RUN}"
