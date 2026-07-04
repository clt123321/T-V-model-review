#!/usr/bin/env bash
# ============================================================================
# Wan2.2-TI2V-5B  T2V single-GPU run on this box (A800 80GB)
# ----------------------------------------------------------------------------
# Env prepared by:
#   /root/wan22-venv                (uv-managed py3.11, torch 2.4.1+cu121)
#   symlinked to project/.venv
#
# Hardware profile: 8x A800-SXM4-80GB. GPU 0 is often shared; default to GPU 1.
# Override with: CUDA_VISIBLE_DEVICES=<n> bash scripts/run_wan22_ti2v5b.sh
#
# Model checkpoint has been SHA256-verified against Hugging Face LFS metadata.
# ============================================================================
set -euo pipefail

ROOT="/home/web_server/antispam/project/clt/video-generation"
VENV="/root/wan22-venv"
cd "$ROOT/Wan2.2"

# Activate venv without sourcing shell rc
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

# CUDA / perf env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="$ROOT/Wan2.2-TI2V-5B"
TAG="${TAG:-first}"                                 # override to name output
SAVE_FILE="$ROOT/outputs/${TAG}_wan22_ti2v5b.mp4"
LOG_FILE="$ROOT/logs/${TAG}_run.log"

PROMPT="${PROMPT:-A cinematic tracking shot of a small silver robot walking through a rainy neon alley at night, wet pavement reflections, soft volumetric light, realistic camera movement, detailed background, smooth motion, high quality, no text, no watermark.}"

# Optional: input image path (activates image-to-video path in WanTI2V)
IMAGE="${IMAGE:-}"

# Overridable sampling knobs
SIZE="${SIZE:-1280*704}"                            # only 1280*704 or 704*1280 for ti2v-5B
FRAME_NUM="${FRAME_NUM:-121}"                       # must be 4n+1
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5.0}"
BASE_SEED="${BASE_SEED:-42}"

mkdir -p "$ROOT/outputs" "$ROOT/logs"

echo "=== Wan2.2-TI2V-5B T2V run (tag=$TAG) ==="       | tee    "$LOG_FILE"
echo "start: $(date '+%F %T')"                          | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"       | tee -a "$LOG_FILE"
echo "size=$SIZE frames=$FRAME_NUM steps=$SAMPLE_STEPS guide=$GUIDE_SCALE shift=$SAMPLE_SHIFT seed=$BASE_SEED" | tee -a "$LOG_FILE"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,memory.free,driver_version --format=csv | tee -a "$LOG_FILE"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import torch, sys
print("python:", sys.version.split()[0])
print("torch :", torch.__version__, "cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# Background GPU sampler
GPU_LOG="$ROOT/logs/${TAG}_gpu_mem.csv"
( nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
             --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
             --format=csv -lms 2000 > "$GPU_LOG" ) &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# --offload_model / --t5_cpu NOT set: A800 80GB has plenty of headroom.
# --convert_model_dtype still saves memory (fp32 -> bf16 DiT weights).
GEN_START=$(date +%s)
IMAGE_ARG=()
if [ -n "$IMAGE" ]; then
    IMAGE_ARG=(--image "$IMAGE")
    echo "IMAGE (i2v): $IMAGE" | tee -a "$LOG_FILE"
fi
python generate.py \
  --task ti2v-5B \
  --size "$SIZE" \
  --frame_num "$FRAME_NUM" \
  --sample_steps "$SAMPLE_STEPS" \
  --sample_guide_scale "$GUIDE_SCALE" \
  --sample_shift "$SAMPLE_SHIFT" \
  --base_seed "$BASE_SEED" \
  --ckpt_dir "$CKPT_DIR" \
  --convert_model_dtype \
  "${IMAGE_ARG[@]}" \
  --save_file "$SAVE_FILE" \
  --prompt "$PROMPT" 2>&1 | tee -a "$LOG_FILE"

GEN_END=$(date +%s)
echo "gen elapsed: $((GEN_END-GEN_START))s" | tee -a "$LOG_FILE"
echo "end: $(date '+%F %T')" | tee -a "$LOG_FILE"
echo "output video: $SAVE_FILE" | tee -a "$LOG_FILE"

python - <<PY 2>&1 | tee -a "$LOG_FILE"
import csv
with open("$GPU_LOG") as f:
    rows=list(csv.reader(f))
mems=[int(r[1].strip().split()[0]) for r in rows[1:] if r and r[1].strip()]
if mems:
    print(f"peak GPU memory: {max(mems)} MiB / samples={len(mems)}")
PY
