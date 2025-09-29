#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)/..

# Project root
PROJECT_ROOT=$(cd .. && pwd)

# Defaults
WS_HOST="0.0.0.0"
WS_PORT=9092
WS_BACKEND="faster_whisper"
OMP_THREADS=1
CUSTOM_MODEL_PATH=""
TRT_MODEL_PATH=""
TRT_MULTILINGUAL=false
SINGLE_MODEL=true

# Parse CLI arguments
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
      echo "Config file notes:"
      echo "Ensure /workspace/ASR/src/websocket/conf/config.ini contains:"
      echo "[model]"
      echo "http_url=http://127.0.0.1:8001/transcribe_websocket"
      echo ""
      echo "This URL should point to the IntegratedASR streaming endpoint"
      exit 0
      ;;
    --help)
      echo "IntegratedASR WebSocket Server Startup Script"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --host HOST              Server host (default: 0.0.0.0)"
      echo "  --port, -p PORT          WebSocket port (default: 9092)"
      echo "  --backend, -b BACKEND    Backend type (default: faster_whisper)"
      echo "  --omp-threads NUM        OpenMP threads (default: 1)"
      echo "  --faster-whisper-model PATH  Custom Whisper model path"
      echo "  --trt-model PATH         TensorRT model path"
      echo "  --trt-multilingual       Enable TensorRT multilingual"
      echo "  --no-single-model        Disable single-model mode"
      echo "  --help-config            Show config file help"
      echo "  --help                   Show this help"
      echo ""
      echo "Examples:"
      echo "  $0                       # Start with defaults"
      echo "  $0 --port 9093          # Start on a specific port"
      echo "  $0 --help-config        # Show config instructions"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; echo "Use --help for usage"; exit 1 ;;
  esac
  shift
done

# ---- Config file handling (no hard-coded dev path) ----
DEFAULT_CONFIG_FILE="$PROJECT_ROOT/src/websocket/conf/config.ini"
CONFIG_FILE="${CONFIG_FILE:-$DEFAULT_CONFIG_FILE}"  # env override

# Parse CLI arguments (新增一个 --config)
# ...在 while [[ "$#" -gt 0 ]] 的 case 里加一条：
#   --config) CONFIG_FILE="$2"; shift ;;

# Validate Config
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "❌ Config file not found: $CONFIG_FILE"
  echo "   Tried default: $DEFAULT_CONFIG_FILE"
  echo "   Pass --config <path> or set env CONFIG_FILE to override"
  exit 1
fi

# Validate TensorRT args
if [[ "$WS_BACKEND" == "tensorrt" && -z "$TRT_MODEL_PATH" ]]; then
  echo "❌ TensorRT backend requires --trt-model"
  exit 1
fi

# Check port occupancy
if command -v lsof >/dev/null 2>&1; then
  if lsof -Pi :$WS_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Warning: Port $WS_PORT is already in use"
    echo "   Use --port to choose another port or stop the process"
    exit 1
  fi
fi

# Set environment variables
export OMP_NUM_THREADS=$OMP_THREADS

# Print configuration
echo "🚀 Starting IntegratedASR WebSocket server"
echo "=================================================="
echo "Host:           $WS_HOST"
echo "Port:           $WS_PORT"
echo "Backend:        $WS_BACKEND"
echo "Config file:    $CONFIG_FILE"
echo "OpenMP threads: $OMP_THREADS"
if [[ -n "$CUSTOM_MODEL_PATH" ]]; then
  echo "Custom model:   $CUSTOM_MODEL_PATH"
fi
if [[ -n "$TRT_MODEL_PATH" ]]; then
  echo "TensorRT model: $TRT_MODEL_PATH"
  echo "Multilingual:   $TRT_MULTILINGUAL"
fi
echo "Single model:   $SINGLE_MODEL"
echo "=================================================="

# Switch to project root
cd "$PROJECT_ROOT"



# Build Python command args
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

# Start server
echo "🔧 Starting WebSocket service..."
python -c "
import os
import sys
sys.path.insert(0, '.')

from src.websocket.server import TranscriptionServer
import argparse

# Parse args
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

# Launch server
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
