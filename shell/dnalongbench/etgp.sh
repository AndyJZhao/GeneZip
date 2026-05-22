#!/bin/bash
# Canonical entrypoint: `bash shell/dnalongbench/etgp.sh`.
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
WANDB_PROJ="${WANDB_PROJ:-DNALLM_ETGP}"
USE_WANDB="${USE_WANDB:-false}"

CKPT="${CKPT:-GeneZip-70M-Intergenic-focused}"
HF_USER="${HF_USER:-andyjzhao}"
CELL_TYPE="${CELL_TYPE:-CRISPRi_EPI_K562_hg19}"
POOLING_METHOD="${POOLING_METHOD:-mean}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
USE_LORA="${USE_LORA:-true}"

DATA_ROOT="${DATA_ROOT:-./data/dnalongbench}"
SAVE_DIR="${SAVE_DIR:-./temp/ETGP_ckpt}"
DATA_MAX_LENGTH="${DATA_MAX_LENGTH:-12800}"
TEST_LOAD="${TEST_LOAD:-best}"

LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-15}"
GRAD_ACC="${GRAD_ACC:-8}"

# Optional caps for smoke runs (limits parsing cost too).
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-}"
MAX_VALID_RECORDS="${MAX_VALID_RECORDS:-}"
MAX_TEST_RECORDS="${MAX_TEST_RECORDS:-}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

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

CMD_TO_RUN="${RUNNER} src/scripts/finetune_dlb_binary_classification.py \
  task=etgp cell_type=${CELL_TYPE} \
  ckpt=${CKPT} hf_user=${HF_USER} tokenizer=fast pooling_method=${POOLING_METHOD} \
  freeze_encoder=${FREEZE_ENCODER} use_lora=${USE_LORA} \
  data_root=${DATA_ROOT} data_max_length=${DATA_MAX_LENGTH} save_dir=${SAVE_DIR} test_load=${TEST_LOAD} \
  training.learning_rate=${LR} training.num_train_epochs=${EPOCHS} training.gradient_accumulation_steps=${GRAD_ACC} \
  ${WANDB_ARGS}"

if [[ -n "${MAX_TRAIN_RECORDS}" ]]; then
  CMD_TO_RUN="${CMD_TO_RUN} max_train_records=${MAX_TRAIN_RECORDS}"
fi
if [[ -n "${MAX_VALID_RECORDS}" ]]; then
  CMD_TO_RUN="${CMD_TO_RUN} max_valid_records=${MAX_VALID_RECORDS}"
fi
if [[ -n "${MAX_TEST_RECORDS}" ]]; then
  CMD_TO_RUN="${CMD_TO_RUN} max_test_records=${MAX_TEST_RECORDS}"
fi

echo "Running command:"
echo "${CMD_TO_RUN}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi
eval "${CMD_TO_RUN}"
