#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)/..

# 项目根目录
PROJECT_ROOT=$(cd .. && pwd)

# 默认值
WS_HOST="0.0.0.0"
WS_PORT=9092
WS_BACKEND="faster_whisper"
OMP_THREADS=1
CUSTOM_MODEL_PATH=""
TRT_MODEL_PATH=""
TRT_MULTILINGUAL=false
SINGLE_MODEL=true

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --host) WS_HOST="$2"; shift ;;
    --port|-p) WS_PORT="$2"; shift ;;
    --backend|-b) WS_BACKEND="$2"; shift ;;
    --omp-threads) OMP_THREADS="$2"; shift ;;
    --faster-whisper-model|-fw) CUSTOM_MODEL_PATH="$2"; shift ;;
    --trt-model|-trt) TRT_MODEL_PATH="$2"; shift ;;
    --trt-multilingual|-m) TRT_MULTILINGUAL=true ;;
    --no-single-model) SINGLE_MODEL=false ;;
    --help-config)
      echo "配置文件说明:"
      echo "请确保 /workspace/ASR/src/websocket/conf/config.ini 包含:"
      echo "[model]"
      echo "http_url=http://127.0.0.1:8001/transcribe_websocket"
      echo ""
      echo "该URL应指向IntegratedASR的流式转写接口"
      exit 0
      ;;
    --help)
      echo "IntegratedASR WebSocket Server 启动脚本"
      echo ""
      echo "用法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --host HOST              服务器主机地址 (默认: 0.0.0.0)"
      echo "  --port, -p PORT          WebSocket端口 (默认: 9092)"
      echo "  --backend, -b BACKEND    后端类型 (默认: faster_whisper)"
      echo "  --omp-threads NUM        OpenMP线程数 (默认: 1)"
      echo "  --faster-whisper-model PATH  自定义Whisper模型路径"
      echo "  --trt-model PATH         TensorRT模型路径"
      echo "  --trt-multilingual       启用TensorRT多语言支持"
      echo "  --no-single-model        禁用单模型模式"
      echo "  --help-config            显示配置文件帮助"
      echo "  --help                   显示此帮助信息"
      echo ""
      echo "示例:"
      echo "  $0                       # 使用默认配置启动"
      echo "  $0 --port 9093          # 指定端口启动"
      echo "  $0 --help-config        # 查看配置说明"
      exit 0
      ;;
    *) echo "未知参数: $1"; echo "使用 --help 查看帮助"; exit 1 ;;
  esac
  shift
done

# 验证配置文件
CONFIG_FILE="/workspace/ASR/src/websocket/conf/config.ini"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "❌ 配置文件不存在: $CONFIG_FILE"
  echo "请使用 --help-config 查看配置说明"
  exit 1
fi

# 验证TensorRT参数
if [[ "$WS_BACKEND" == "tensorrt" && -z "$TRT_MODEL_PATH" ]]; then
  echo "❌ TensorRT后端需要指定 --trt-model 参数"
  exit 1
fi

# 检查端口占用
if command -v lsof >/dev/null 2>&1; then
  if lsof -Pi :$WS_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  警告: 端口 $WS_PORT 已被占用"
    echo "   请使用 --port 指定其他端口或停止占用进程"
    exit 1
  fi
fi

# 设置环境变量
export OMP_NUM_THREADS=$OMP_THREADS

# 打印配置信息
echo "🚀 启动 IntegratedASR WebSocket 服务器"
echo "=================================================="
echo "主机:          $WS_HOST"
echo "端口:          $WS_PORT"
echo "后端:          $WS_BACKEND"
echo "配置文件:      $CONFIG_FILE"
echo "OpenMP线程:    $OMP_THREADS"
if [[ -n "$CUSTOM_MODEL_PATH" ]]; then
  echo "自定义模型:    $CUSTOM_MODEL_PATH"
fi
if [[ -n "$TRT_MODEL_PATH" ]]; then
  echo "TensorRT模型:  $TRT_MODEL_PATH"
  echo "多语言支持:    $TRT_MULTILINGUAL"
fi
echo "单模型模式:    $SINGLE_MODEL"
echo "=================================================="

# 切换到项目根目录
cd "$PROJECT_ROOT"



# 构建Python命令参数
PYTHON_ARGS="--host $WS_HOST --port $WS_PORT --backend $WS_BACKEND --omp_num_threads $OMP_THREADS"

if [[ -n "$CUSTOM_MODEL_PATH" ]]; then
  PYTHON_ARGS="$PYTHON_ARGS --faster_whisper_custom_model_path $CUSTOM_MODEL_PATH"
fi

if [[ -n "$TRT_MODEL_PATH" ]]; then
  PYTHON_ARGS="$PYTHON_ARGS --trt_model_path $TRT_MODEL_PATH"
fi

if [[ "$TRT_MULTILINGUAL" == "true" ]]; then
  PYTHON_ARGS="$PYTHON_ARGS --trt_multilingual"
fi

if [[ "$SINGLE_MODEL" == "false" ]]; then
  PYTHON_ARGS="$PYTHON_ARGS --no_single_model"
fi

# 启动服务器
echo "🔧 启动WebSocket服务中..."
python -c "
import os
import sys
sys.path.insert(0, '.')

from src.websocket.server import TranscriptionServer
import argparse

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', type=int, default=9092)
parser.add_argument('--backend', default='faster_whisper')
parser.add_argument('--omp_num_threads', type=int, default=1)
parser.add_argument('--faster_whisper_custom_model_path', default=None)
parser.add_argument('--trt_model_path', default=None)
parser.add_argument('--trt_multilingual', action='store_true')
parser.add_argument('--no_single_model', action='store_true')

args = parser.parse_args('$PYTHON_ARGS'.split())

# 启动服务器
server = TranscriptionServer()
server.run(
    args.host,
    port=args.port,
    backend=args.backend,
    faster_whisper_custom_model_path=args.faster_whisper_custom_model_path,
    whisper_tensorrt_path=args.trt_model_path,
    trt_multilingual=args.trt_multilingual,
    single_model=not args.no_single_model,
)
" 