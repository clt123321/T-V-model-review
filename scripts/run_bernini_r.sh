#!/usr/bin/env bash
# ============================================================================
# Bernini-R-1.3B-Diffusers  single-GPU inference wrapper
# ----------------------------------------------------------------------------
# Env:
#   /root/bernini-venv                (uv-managed py3.11, torch 2.5.1+cu124)
#   /root/Bernini                     (bytedance/Bernini github clone)
#
# Weights (self-contained diffusers dir with all subcomponents):
#   /home/web_server/antispam/project/clt/video-generation/Bernini-R-1.3B-Diffusers
#
# Hardware profile: 8x A800-SXM4-80GB. Default GPU 1 (GPU 0 is often shared).
# ============================================================================
set -euo pipefail

ROOT="/home/web_server/antispam/project/clt/video-generation"
VENV="/root/bernini-venv"
BERNINI="/root/Bernini"
CONFIG="$ROOT/Bernini-R-1.3B-Diffusers"

# Activate venv
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

# Overridable knobs
TAG="${TAG:-first}"
CASE="${CASE:-}"                                     # path to case JSON (recommended)
PROMPT="${PROMPT:-}"                                 # or direct prompt
TASK_TYPE="${TASK_TYPE:-t2v}"                        # t2v | t2i | i2i | v2v | r2v | rv2v | mv2v
GUIDANCE_MODE="${GUIDANCE_MODE:-t2v_apg}"            # depends on TASK_TYPE
IMAGE="${IMAGE:-}"                                   # single source image (i2i)
IMAGES="${IMAGES:-}"                                 # comma-separated reference images (r2v/rv2v)
VIDEO="${VIDEO:-}"                                   # source video path (v2v/rv2v/mv2v)
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-848}"
FPS="${FPS:-16}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
SEED="${SEED:-}"

SAVE_FILE="${SAVE_FILE:-$ROOT/outputs/${TAG}_bernini_r_${TASK_TYPE}.mp4}"
LOG_FILE="$ROOT/logs/${TAG}_bernini_${TASK_TYPE}.log"

mkdir -p "$ROOT/outputs" "$ROOT/logs"

cd "$BERNINI"

echo "=== Bernini-R-1.3B ${TASK_TYPE} (tag=$TAG) ===" | tee    "$LOG_FILE"
echo "start: $(date '+%F %T')"                        | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"     | tee -a "$LOG_FILE"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,memory.free,driver_version --format=csv | tee -a "$LOG_FILE"

# Background GPU sampler
GPU_LOG="$ROOT/logs/${TAG}_bernini_gpu_mem.csv"
( nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
             --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
             --format=csv -lms 2000 > "$GPU_LOG" ) &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# Build args
ARGS=(--config "$CONFIG" --guidance_mode "$GUIDANCE_MODE"
      --num_frames "$NUM_FRAMES" --num_inference_steps "$NUM_INFERENCE_STEPS"
      --max_image_size "$MAX_IMAGE_SIZE" --fps "$FPS" --flow_shift "$FLOW_SHIFT")
[ -n "$CASE" ]    && ARGS+=(--case "$CASE")
[ -n "$PROMPT" ]  && ARGS+=(--prompt "$PROMPT" --task_type "$TASK_TYPE" --output "$SAVE_FILE")
[ -n "$IMAGE" ]   && ARGS+=(--image "$IMAGE")
[ -n "$VIDEO" ]   && ARGS+=(--video "$VIDEO")
[ -n "$SEED" ]    && ARGS+=(--seed "$SEED")
if [ -n "$IMAGES" ]; then
    IFS=',' read -ra IMG_ARR <<< "$IMAGES"
    ARGS+=(--images "${IMG_ARR[@]}")
fi

GEN_START=$(date +%s)
python infer_single_gpu.py "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
GEN_END=$(date +%s)

echo "gen elapsed: $((GEN_END-GEN_START))s" | tee -a "$LOG_FILE"
echo "end: $(date '+%F %T')" | tee -a "$LOG_FILE"

python - <<PY 2>&1 | tee -a "$LOG_FILE"
import csv
with open("$GPU_LOG") as f:
    rows=list(csv.reader(f))
mems=[int(r[1].strip().split()[0]) for r in rows[1:] if r and r[1].strip()]
if mems:
    print(f"peak GPU memory: {max(mems)} MiB / samples={len(mems)}")
PY
