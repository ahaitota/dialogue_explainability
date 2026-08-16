#!/bin/bash
#SBATCH -J b1_attnlrp
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=gpu_cc8.6
#SBATCH --gpus=nvidia_l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -o /home/haitotaa/dialogue_explainability/logs/%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alina.haitota451@student.cuni.cz

# Runs a pipeline stage (default: B1 AttnLRP) on the UFAL AIC cluster.
#
# Usage:
#   sbatch --export=ALL,CONFIG=configs/experiment.yaml scripts/run_aic.sh
#   # CONFIG defaults to configs/experiment.yaml if not set.
#   # Override the SBATCH header at submit time too, e.g.:
#   sbatch -t 1-00:00:00 --export=ALL,CONFIG=... scripts/run_aic.sh
#
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces memory fragmentation,
# which matters for B1: it keeps one large retained backward graph alive per
# example.

set -euo pipefail

nvidia-smi

REPO="/home/haitotaa/dialogue_explainability"  
LOG_DIR="$REPO/logs"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

CONFIG="${CONFIG:-configs/experiment.yaml}"
CONFIG_PATH="$CONFIG"
[[ "$CONFIG_PATH" != /* ]] && CONFIG_PATH="$REPO/$CONFIG_PATH"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config file not found: $CONFIG_PATH" >&2
    exit 1
fi

# --- Repo-local cache/temp (uses project storage quota, not $HOME) --------
CACHE_ROOT="$REPO/.cache"
export TMPDIR="$CACHE_ROOT/tmp"
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

# --- Environment: activate the venv  ---
cd "$REPO"
VENV="${VENV:-venvs/.venv}"
source "$VENV/bin/activate"

echo "Job:      ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Config:   $CONFIG_PATH"
echo "Task:     ${TASK:-<all>}"
echo "Setup:    ${SETUP:-<all>}"
echo "Started:  $(date)"

# --- Run B1 (writes results to results/attnlrp/ in the repo) --------------
B1_ARGS=()
[[ -n "${TASK:-}" ]] && B1_ARGS+=(--task "$TASK")
[[ -n "${SETUP:-}" ]] && B1_ARGS+=(--setup "$SETUP")
python scripts/run_b1.py "$CONFIG_PATH" "${B1_ARGS[@]}"

echo "Finished: $(date)"
