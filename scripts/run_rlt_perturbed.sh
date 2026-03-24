#!/bin/bash
# =============================================================================
# RLT Training Pipeline for Perturbed LIBERO
# =============================================================================
#
# Prerequisites (run once manually if not already done):
#   apt-get update && apt-get install -y libosmesa6-dev git-lfs
#   git lfs install
#
# Usage:
#   bash scripts/run_rlt_perturbed.sh [TASK_SUITE] [GPU_ID]
#
# Examples:
#   bash scripts/run_rlt_perturbed.sh libero_10 0
#   bash scripts/run_rlt_perturbed.sh libero_spatial 0
# =============================================================================

set -euo pipefail

TASK=${1:-"libero_10"}
GPU=${2:-0}

# ---- Environment setup ----
PROJ_DIR=~/workspace/fxz/openpi
CONDA_DIR=~/workspace/fxz/miniconda3
CONDA_ENV=pi

cd "${PROJ_DIR}"
source "${CONDA_DIR}/bin/activate" "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES=$GPU
# Headless MuJoCo rendering (no display needed)
unset DISPLAY
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# ---- Config ----
PERTURBATIONS="scene_swap obstacle occlusion object_swap_nontarget object_swap_target"
NUM_TRIALS=50
NUM_RL_EPISODES=500
WARMUP_EPISODES=20
PORT=8000
BASE_CKPT="gs://openpi-assets/checkpoints/pi05_libero"

# ---- Directories (all keyed by TASK) ----
DATA_DIR="data/libero/perturbed_trajs_${TASK}"
RL_TOKEN_DIR="checkpoints/rlt_${TASK}/rl_token"
ONLINE_RL_DIR="checkpoints/rlt_${TASK}/online_rl"
VIDEO_DIR="data/libero_rlt/${TASK}"
LOG_DIR="logs/rlt_${TASK}"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  RLT Pipeline  |  task=${TASK}  gpu=${GPU}"
echo "  proj=${PROJ_DIR}"
echo "  conda=${CONDA_ENV}"
echo "============================================================"

# ---- Helpers ----
# Server runs use `uv run` (handles openpi deps); client/training use plain python.
start_server() {
    local log="$1"; shift
    nohup uv run "$@" > "${log}" 2>&1 &
    local pid=$!
    echo "    Server PID=${pid}, log=${log}"
    echo "    Waiting for server to be ready..."
    # Wait until server prints "Creating server" or up to 120s
    for i in $(seq 1 120); do
        if grep -q -i "creating server\|Uvicorn running\|server started\|Serving" "${log}" 2>/dev/null; then
            echo "    Server ready (${i}s)"
            echo "${pid}"
            return 0
        fi
        sleep 1
    done
    echo "    WARNING: server may not be ready after 120s, continuing anyway"
    echo "${pid}"
}

kill_server() {
    kill "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
}

# =============================================================================
# Stage 1: Collect perturbed trajectories
# =============================================================================
echo ""
echo ">>> Stage 1: Collect perturbed trajectories"

SDE_PID=$(start_server "${LOG_DIR}/serve_sde.log" \
    scripts/serve_sde_policy.py \
        --env LIBERO --noise-level 1.5 --num-steps 3 --port ${PORT})

python third_party/libero/collect_perturbed_trajectories.py \
    --task-suite-name ${TASK} \
    --perturbations ${PERTURBATIONS} \
    --output-dir ${DATA_DIR} \
    --num-trials-per-task ${NUM_TRIALS} \
    --port ${PORT} \
    --resume \
    2>&1 | tee -a "${LOG_DIR}/collect.log"

kill_server ${SDE_PID}
echo "    Done -> ${DATA_DIR}"

# =============================================================================
# Stage 2: Train RL token encoder-decoder
# =============================================================================
echo ""
echo ">>> Stage 2: Train RL token"

python scripts/train_rlt.py \
    --phase rl_token \
    --task-suite-name ${TASK} \
    --base-checkpoint ${BASE_CKPT} \
    --demo-data-dir ${DATA_DIR}/perturbed_success \
    --output-dir ${RL_TOKEN_DIR} \
    --num-steps 5000 \
    --finetune-vla \
    2>&1 | tee "${LOG_DIR}/train_rl_token.log"

echo "    Done -> ${RL_TOKEN_DIR}"

# =============================================================================
# Stage 3: Online RL
# =============================================================================
echo ""
echo ">>> Stage 3: Online RL"

RLT_PID=$(start_server "${LOG_DIR}/serve_rlt.log" \
    scripts/serve_rlt_policy.py \
        --base-checkpoint ${BASE_CKPT} \
        --rlt-checkpoint ${RL_TOKEN_DIR}/best \
        --mode rlt --port ${PORT})

python scripts/train_rlt.py \
    --phase online_rl \
    --task-suite-name ${TASK} \
    --rlt-checkpoint ${RL_TOKEN_DIR}/best \
    --perturbations ${PERTURBATIONS} \
    --num-episodes ${NUM_RL_EPISODES} \
    --warmup-episodes ${WARMUP_EPISODES} \
    --output-dir ${ONLINE_RL_DIR} \
    --port ${PORT} \
    --video-out-path ${VIDEO_DIR}/online_rl \
    2>&1 | tee "${LOG_DIR}/train_online_rl.log"

kill_server ${RLT_PID}
echo "    Done -> ${ONLINE_RL_DIR}"

# =============================================================================
# Stage 4: Evaluate
# =============================================================================
echo ""
echo ">>> Stage 4: Evaluate"

EVAL_PID=$(start_server "${LOG_DIR}/serve_eval.log" \
    scripts/serve_rlt_policy.py \
        --base-checkpoint ${BASE_CKPT} \
        --rlt-checkpoint ${ONLINE_RL_DIR}/final \
        --mode rlt --port ${PORT})

python third_party/libero/main_rlt.py \
    --task-suite-name ${TASK} \
    --port ${PORT} \
    --replan-steps 10 \
    --num-trials-per-task 50 \
    --video-out-path ${VIDEO_DIR}/eval \
    2>&1 | tee "${LOG_DIR}/eval.log"

kill_server ${EVAL_PID}

echo ""
echo "============================================================"
echo "  Done!"
echo "  RL Token:   ${RL_TOKEN_DIR}"
echo "  Online RL:  ${ONLINE_RL_DIR}"
echo "  Videos:     ${VIDEO_DIR}/eval"
echo "  Logs:       ${LOG_DIR}/"
echo "============================================================"
