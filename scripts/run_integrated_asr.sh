#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)/..

# Project root (assuming scripts/ is under the project root)
PROJECT_ROOT=$(cd .. && pwd)

# Defaults (can be overridden by CLI args)
CT2_MODEL_PATH="large-v3"
KIMI_MODEL_PATH="moonshotai/Kimi-Audio-7B-Instruct"
LID_MODEL_PATH="$PROJECT_ROOT/models/lid/language_fnn_only2.pt"
CONFIDENCE_THRESHOLD=0.7
AUTO_WARMUP="true"
ASR_PORT=8001
ASR_NUM_WORKERS=1

# Parse CLI args
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ct2-model) CT2_MODEL_PATH="$2"; shift ;;
    --kimi-model) KIMI_MODEL_PATH="$2"; shift ;;
    --lid-model) LID_MODEL_PATH="$2"; shift ;;
    --confidence) CONFIDENCE_THRESHOLD="$2"; shift ;;
    --port) ASR_PORT="$2"; shift ;;
    --workers) ASR_NUM_WORKERS="$2"; shift ;;
    --no-warmup) AUTO_WARMUP="false" ;;
    --warmup) AUTO_WARMUP="true" ;;
    --help) 
      echo "IntegratedASR Server Startup Script"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --ct2-model PATH     CTranslate2 Whisper model path (default: large-v3)"
      echo "  --kimi-model PATH    Kimi model path (default: moonshotai/Kimi-Audio-7B-Instruct)"
      echo "  --lid-model PATH     Language ID model path (default: ./models/lid/language_fnn_only2.pt)"
      echo "  --confidence NUM     Language confidence threshold (default: 0.7)"
      echo "  --port NUM           Service port (default: 8001)"
      echo "  --workers NUM        Worker processes (default: 1)"
      echo "  --warmup             Enable auto warmup (default)"
      echo "  --no-warmup          Disable auto warmup"
      echo "  --help               Show this help"
      echo ""
      echo "Examples:"
      echo "  $0                                    # Start with defaults"
      echo "  $0 --port 8080 --no-warmup          # Specify port, disable warmup"
      echo "  $0 --confidence 0.8                 # Set confidence threshold"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; echo "Use --help for usage"; exit 1 ;;
  esac
  shift
done

# Validate required files
if [[ ! -f "$LID_MODEL_PATH" ]]; then
  echo "❌ Error: Language ID model file not found: $LID_MODEL_PATH"
  echo "   Please check the path or download the model file"
  exit 1
fi

# Export as environment variables (read by FastAPI)
export CT2_MODEL_PATH
export KIMI_MODEL_PATH
export LID_MODEL_PATH
export CONFIDENCE_THRESHOLD
export AUTO_WARMUP

# Print configuration
echo "🚀 Starting IntegratedASR server"
echo "=================================="
echo "CT2 model:      $CT2_MODEL_PATH"
echo "Kimi model:     $KIMI_MODEL_PATH"
echo "LID model:      $LID_MODEL_PATH"
echo "Confidence:     $CONFIDENCE_THRESHOLD"
echo "Auto warmup:    $AUTO_WARMUP"
echo "Port:           $ASR_PORT"
echo "Workers:        $ASR_NUM_WORKERS"
echo "=================================="

# Check if port is in use
if command -v lsof >/dev/null 2>&1; then
  if lsof -Pi :$ASR_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Warning: Port $ASR_PORT is already in use"
    echo "   Use --port to choose another port or stop the process"
    exit 1
  fi
fi

# Switch to project root
cd "$PROJECT_ROOT"

# ✅ Use current Python environment to run uvicorn
echo "🔧 Starting service..."
python -m uvicorn src.app.fastapi_api:app \
  --host 0.0.0.0 \
  --port "$ASR_PORT" \
  --workers "$ASR_NUM_WORKERS" 