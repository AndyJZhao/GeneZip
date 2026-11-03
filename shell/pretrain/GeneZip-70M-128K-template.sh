#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
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

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" \
  "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}"
cd "${PROJECT_ROOT}"

WANDB_PROJ="${WANDB_PROJ:-GeneZip}"
HF_USER="${HF_USER:-andyjzhao}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-false}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-GeneZip-70M}"

PREFETCH_CKPT="${PRETRAINED_CKPT}"
if [[ "${PREFETCH_CKPT}" != */* ]]; then
  PREFETCH_CKPT="${HF_USER}/${PREFETCH_CKPT}"
fi
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${PREFETCH_CKPT}', resume_download=True)"

CMD_TO_RUN="accelerate launch --num_processes ${NUM_PROCESSES} --main_process_port 0 --mixed_precision ${MIXED_PRECISION} \
  src/scripts/pretrain_genezip.py \
  task=pretrain pretraining_name=GeneZip-70M-128K data=gencode_human_128k model_name=genezip_70m \
  max_len=128000 batch_size=4 grad_acc_steps=1 max_train_steps=24414 eval_steps=1000 num_valid_samples=3000 \
  upload_to_hf=${UPLOAD_TO_HF} wandb.project=${WANDB_PROJ} hf_user=${HF_USER} pretrained_ckpt=${PRETRAINED_CKPT} \
  use_wandb=true"

echo "Running command:"
echo "${CMD_TO_RUN}"
eval "${CMD_TO_RUN}"
