#!/bin/bash
# Canonical entrypoint: `bash shell/dnalongbench/cmp.sh`.
# NOTE: The SBATCH lines below are examples for Slurm; edit/remove for your cluster.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

# -----------------------------
# Environment (cache dirs)
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PROJECT_ROOT

# Cache / runtime directories (override via env)
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


mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" \
  "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}"
cd "${PROJECT_ROOT}"

# -----------------------------
# Local vars (override via env)
# -----------------------------
WANDB_PROJ="${WANDB_PROJ:-DLB-CMP}"
USE_WANDB="${USE_WANDB:-false}"

SUBSET="${SUBSET:-HFF}"
CKPT="${CKPT:-GeneZip-70M-Transcript-balanced}"
HF_USER="${HF_USER:-andyjzhao}"
POOLING="${POOLING:-mean}"
EMB_POSITION="${EMB_POSITION:-before_lm_head}"

TRAIN_STEPS="${TRAIN_STEPS:-5000}"
EVAL_STEPS="${EVAL_STEPS:-2500}"
NUM_TRAIN_SAMPLES="${NUM_TRAIN_SAMPLES:-0}"
NUM_VALID_SAMPLES="${NUM_VALID_SAMPLES:-100}"
NUM_TEST_SAMPLES="${NUM_TEST_SAMPLES:-0}"
LR="${LR:-2e-4}"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-40}"

USE_CACHE="${USE_CACHE:-true}"
PRECOMPUTE_EMBEDDINGS="${PRECOMPUTE_EMBEDDINGS:-true}"
EMB_CACHE="${EMB_CACHE:-}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# -----------------------------
# Prefetch checkpoint (optional)
# -----------------------------
if [[ "${PRINT_ONLY:-false}" != "true" && ! -e "${CKPT}" ]]; then
  CKPT_REPO_ID="${CKPT}"
  if [[ "${CKPT_REPO_ID}" != */* ]]; then
    CKPT_REPO_ID="${HF_USER}/${CKPT_REPO_ID}"
  fi
  echo "Prefetching checkpoint: ${CKPT_REPO_ID}"
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${CKPT_REPO_ID}', resume_download=True)"
fi

# -----------------------------
# Runner
# -----------------------------
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  RUNNER="accelerate launch --num_processes ${NUM_PROCESSES} --main_process_port 0 --mixed_precision ${MIXED_PRECISION}"
else
  RUNNER="python"
fi

WANDB_ARGS="use_wandb=false"
if [[ "${USE_WANDB}" == "true" ]]; then
  WANDB_ARGS="use_wandb=true wandb.project=${WANDB_PROJ}"
fi

CMD_TO_RUN="${RUNNER} src/scripts/finetune_dlb_cmp.py \
  task=cmp \
  subset=${SUBSET} ckpt=${CKPT} hf_user=${HF_USER} \
  pooling=${POOLING} emb_position=${EMB_POSITION} \
  train_steps=${TRAIN_STEPS} eval_steps=${EVAL_STEPS} \
  num_train_samples=${NUM_TRAIN_SAMPLES} num_valid_samples=${NUM_VALID_SAMPLES} num_test_samples=${NUM_TEST_SAMPLES} \
  lr=${LR} \
  batch_size=${BATCH_SIZE} grad_acc_steps=${GRAD_ACC_STEPS} eval_batch_size=${EVAL_BATCH_SIZE} \
  use_cache=${USE_CACHE} precompute_embeddings=${PRECOMPUTE_EMBEDDINGS} encode_batch_size=${ENCODE_BATCH_SIZE} \
  ${WANDB_ARGS}"

if [[ -n "${EMB_CACHE}" ]]; then
  CMD_TO_RUN="${CMD_TO_RUN} emb_cache=${EMB_CACHE}"
fi

echo "Running command:"
echo "${CMD_TO_RUN}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi
eval "${CMD_TO_RUN}"
