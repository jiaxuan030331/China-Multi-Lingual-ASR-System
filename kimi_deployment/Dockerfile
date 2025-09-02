# Minimal CUDA-enabled image for Kimi ASR FastAPI service
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Basic env
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=UTF-8 \
    KIMI_DEVICE=cuda \
    KIMI_DEVICE_INDEX=0 \
    KIMI_TORCH_DTYPE=bfloat16 \
    KIMI_PORT=8000 \
    WS_PORT=9091 \
    KIMI_PYTHON_PATH=/data/kimi_env/bin/python

# System packages needed by audio libs and building some wheels
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    sox \
    libsoxr0 \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first to leverage Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir ninja && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge || true

# Copy project sources
COPY . .

# Ensure imports like `from app...` work
ENV PYTHONPATH=/workspace

# ✅ 使用环境变量，支持运行时覆盖模型路径
ENV KIMI_MODEL_PATH="/data/kimi_deployment/kimi_model/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
ENV KIMI_AUDIO_TOKENIZER_PATH="/data/kimi_deployment/kimi_model/models--THUDM--glm-4-voice-tokenizer/snapshots/a5f2404e63c84e92f5238908e1706316324ebafa"

# 创建数据目录，解决运行时路径问题
RUN mkdir -p /workspace/data /workspace/logs

# Expose HTTP API and WebSocket ports
EXPOSE 8000 9091

# Default: start FastAPI HTTP server. For WebSocket server, override CMD:
#   docker run ... python scripts/run_websocket_server.py -p 9091
CMD ["uvicorn", "app.fastapi_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 

### 运行示例
- 如需用外部 Python 环境跑 WebSocket，请把该环境挂载到容器的 `/data/kimi_env`:
```bash
docker build -t kimi-asr:latest /Users/jiaxuanhuang/Downloads/kimi_deployment

# 同时启动 HTTP:8000 + WS:9091
docker run -d --name kimi \
  --gpus all \
  -p 8000:8000 -p 9091:9091 \
  -v /data/kimi_env:/data/kimi_env \
  -e KIMI_PYTHON_PATH=/data/kimi_env/bin/python \
  kimi-asr:latest
```

### 进入容器手动控制
```bash
# 进入容器
docker exec -it kimi /bin/bash

# 关闭服务（示例）
pkill -f "uvicorn app.fastapi_api:app"      # 停 HTTP
pkill -f "scripts/run_websocket_server.py"  # 停 WS

# 重新拉起
./scripts/run_kimi_server.sh --port 8000 &
/data/kimi_env/bin/python scripts/run_websocket_server.py -p 9091 &
```

说明
- 默认用 `./scripts/run_kimi_server.sh` 拉起 HTTP 服务；WebSocket 用 `KIMI_PYTHON_PATH` 指定的 Python 启动。
- 若容器内不存在 `/data/kimi_env/bin/python`，会回退用系统 `python`。
- 容器入口脚本会在前台 `wait` 两个子进程；手动 kill 任意一个，另一个不受影响，容器会在两者都退出时结束。进入容器可任意手动启动/关闭。 