#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)/..

# 项目根目录（假设 scripts/ 在项目根下）
PROJECT_ROOT=$(cd .. && pwd)

# 默认值（可被 CLI 参数覆盖）
CT2_MODEL_PATH="large-v3"
KIMI_MODEL_PATH="moonshotai/Kimi-Audio-7B-Instruct"
LID_MODEL_PATH="$PROJECT_ROOT/WhisperLive/language_fnn_only2.pt"
CONFIDENCE_THRESHOLD=0.7
AUTO_WARMUP="true"
ASR_PORT=8001
ASR_NUM_WORKERS=1

# 解析 CLI 参数
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
      echo "IntegratedASR Server 启动脚本"
      echo ""
      echo "用法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --ct2-model PATH     CTranslate2 Whisper模型路径 (默认: large-v3)"
      echo "  --kimi-model PATH    Kimi模型路径 (默认: moonshotai/Kimi-Audio-7B-Instruct)"
      echo "  --lid-model PATH     语言检测模型路径 (默认: ./WhisperLive/language_fnn_only2.pt)"
      echo "  --confidence NUM     语言置信度阈值 (默认: 0.7)"
      echo "  --port NUM           服务端口 (默认: 8001)"
      echo "  --workers NUM        工作进程数 (默认: 1)"
      echo "  --warmup             启用自动预热 (默认)"
      echo "  --no-warmup          禁用自动预热"
      echo "  --help               显示此帮助信息"
      echo ""
      echo "示例:"
      echo "  $0                                    # 使用默认配置启动"
      echo "  $0 --port 8080 --no-warmup          # 指定端口，禁用预热"
      echo "  $0 --confidence 0.8                 # 设置置信度阈值"
      exit 0
      ;;
    *) echo "未知参数: $1"; echo "使用 --help 查看帮助"; exit 1 ;;
  esac
  shift
done

# 验证关键文件是否存在
if [[ ! -f "$LID_MODEL_PATH" ]]; then
  echo "❌ 错误: 语言检测模型文件不存在: $LID_MODEL_PATH"
  echo "   请检查路径或下载模型文件"
  exit 1
fi

# 导出为环境变量（FastAPI 内读取）
export CT2_MODEL_PATH
export KIMI_MODEL_PATH
export LID_MODEL_PATH
export CONFIDENCE_THRESHOLD
export AUTO_WARMUP

# 打印配置信息
echo "🚀 启动 IntegratedASR 服务器"
echo "=================================="
echo "CT2 模型:      $CT2_MODEL_PATH"
echo "Kimi 模型:     $KIMI_MODEL_PATH"
echo "语言检测模型:  $LID_MODEL_PATH"
echo "置信度阈值:    $CONFIDENCE_THRESHOLD"
echo "自动预热:      $AUTO_WARMUP"
echo "端口:          $ASR_PORT"
echo "工作进程:      $ASR_NUM_WORKERS"
echo "=================================="

# 检查端口是否被占用
if command -v lsof >/dev/null 2>&1; then
  if lsof -Pi :$ASR_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  警告: 端口 $ASR_PORT 已被占用"
    echo "   请使用 --port 指定其他端口或停止占用进程"
    exit 1
  fi
fi

# 切换到项目根目录
cd "$PROJECT_ROOT"

# ✅ 使用当前环境中的 Python 来运行 uvicorn 模块
echo "🔧 启动服务中..."
python -m uvicorn src.app.fastapi_api:app \
  --host 0.0.0.0 \
  --port "$ASR_PORT" \
  --workers "$ASR_NUM_WORKERS" 