#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=8
#SBATCH --mem=512G
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

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" \
  "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}"
cd "${PROJECT_ROOT}"

WANDB_PROJ="${WANDB_PROJ:-GeneZip}"
HF_USER="${HF_USER:-andyjzhao}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-false}"
BP_PER_TOKEN="${BP_PER_TOKEN:-32}"
MU_R_ALIAS="${MU_R_ALIAS:-Transcript-balanced}"
K_MIN_LIST="${K_MIN_LIST:-[8,8]}"
K_MAX_LIST="${K_MAX_LIST:-[30000,10000]}"

if [[ -z "${REGION_INFO:-}" ]]; then
  case "${MU_R_ALIAS}" in
    Transcript-balanced)
      REGION_INFO="promoter1_cds1_utr2_exon2_intron8_nig8_dig16"
      ;;
    Cis-regulatory-focused)
      REGION_INFO="promoter1_cds16_utr2_exon4_intron2_nig2_dig4"
      ;;
    Promoter-distal-regulatory)
      REGION_INFO="promoter1_cds16_utr8_exon8_intron2_nig2_dig4"
      ;;
    Intergenic-focused)
      REGION_INFO="promoter32_cds16_utr8_exon4_intron2_nig4_dig1"
      ;;
    *)
      echo "Unknown MU_R_ALIAS='${MU_R_ALIAS}'. Set REGION_INFO directly or use Transcript-balanced, Cis-regulatory-focused, Promoter-distal-regulatory, or Intergenic-focused." >&2
      exit 2
      ;;
  esac
fi
ALIAS="${ALIAS:-GeneZip-636M-${MU_R_ALIAS}-128K}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-GeneZip-636M-${MU_R_ALIAS}}"

PREFETCH_CKPT="${PRETRAINED_CKPT}"
if [[ "${PREFETCH_CKPT}" != */* ]]; then
  PREFETCH_CKPT="${HF_USER}/${PREFETCH_CKPT}"
fi
if [[ "${PRINT_ONLY:-false}" != "true" ]]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${PREFETCH_CKPT}', resume_download=True)"
fi

CMD_TO_RUN="accelerate launch --num_processes ${NUM_PROCESSES} --main_process_port 0 --mixed_precision ${MIXED_PRECISION} \
  src/scripts/pretrain_genezip.py \
  task=pretrain data=gencode_human_128k model_name=hnet_mamba_650m_2dc \
  max_len=128000 batch_size=1 grad_acc_steps=4 max_train_steps=24414 eval_steps=1000 num_valid_samples=3000 \
  use_routing_floor=true use_routing_ceiling=true k_min_list=${K_MIN_LIST} k_max_list=${K_MAX_LIST} \
  upload_to_hf=${UPLOAD_TO_HF} wandb.project=${WANDB_PROJ} hf_user=${HF_USER} pretrained_ckpt=${PRETRAINED_CKPT} \
  bp_per_token=${BP_PER_TOKEN} \
  region_info=${REGION_INFO} alias=${ALIAS} \
  use_wandb=true"

echo "Running command:"
echo "${CMD_TO_RUN}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi
eval "${CMD_TO_RUN}"
