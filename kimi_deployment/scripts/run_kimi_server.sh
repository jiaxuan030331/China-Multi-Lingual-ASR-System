#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)/..

# 项目根目录（假设 scripts/ 在项目根下）
PROJECT_ROOT=$(cd .. && pwd)

# 默认值（可被 CLI 参数覆盖）
KIMI_MODEL_PATH="$PROJECT_ROOT/kimi_model/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
KIMI_DEVICE="cuda"
KIMI_DEVICE_INDEX=0
KIMI_TORCH_DTYPE="bfloat16"
KIMI_PORT=8000
KIMI_NUM_WORKERS=1

# 解析 CLI 参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model-path) KIMI_MODEL_PATH="$2"; shift ;;
    --device) KIMI_DEVICE="$2"; shift ;;
    --device-index) KIMI_DEVICE_INDEX="$2"; shift ;;
    --dtype) KIMI_TORCH_DTYPE="$2"; shift ;;
    --port) KIMI_PORT="$2"; shift ;;
    --workers) KIMI_NUM_WORKERS="$2"; shift ;; #当前显存只支持单进程
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# 导出为环境变量（FastAPI 内读取）
export KIMI_MODEL_PATH
export KIMI_DEVICE
export KIMI_DEVICE_INDEX
export KIMI_TORCH_DTYPE

# ✅ 使用共享 Conda 环境中的 Python 来运行 uvicorn 模块
python -m uvicorn app.fastapi_api:app \
  --host 0.0.0.0 \
  --port "$KIMI_PORT" \
  --workers "$KIMI_NUM_WORKERS"