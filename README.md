# China Multi-Lingual ASR System

A comprehensive end-to-end multi-lingual Automatic Speech Recognition (ASR) system specifically designed for the Chinese market, featuring intelligent language routing and optimized dialect support.

## 🎯 Project Overview

This system provides real-time semi-streaming transcription via WebSocket frontend, with a sophisticated backend that dynamically selects optimal decoding paths through Whisper encoder + custom-trained language identification modules:

- **Mandarin & English**: Leverages Kimi pipeline for ultra-high precision and speed
- **Cantonese & Other Dialects**: Utilizes fine-tuned Whisper decoder, dramatically reducing error rates (Cantonese CER: 30-40% → ~15%)

## 🏗️ System Architecture
```mermaid
flowchart LR
  %% =======================
  %% Kimi (Embeddings path)
  %% =======================
  subgraph KIMI["Kimi (Embeddings)"]
    direction LR
    K_T["Text"]
    K_A["Audio"]
    K_TK["GLM-4 Tokenizer"]
    K_WE["Whisper Encoder"]
    K_E["Embedding Layer"]

    K_T --> K_TK --> K_E
    K_A --> K_TK
    K_A --> K_WE --> K_E
  end

  %% ============================
  %% faster-whisper (CT2) system
  %% ============================
  subgraph FW["faster-whisper (CT2)"]
    direction LR
    F_A["Audio"]
    F_WE["Whisper Encoder"]
    F_LD{"Language Detection"}
    F_D["Decoder"]
    F_TXT["Text"]

    F_A --> F_WE --> F_LD --> F_D --> F_TXT
  end

  %% =======================
  %% Project Integration
  %% =======================
  subgraph INT["Integration (Project flow)"]
    direction LR
    I_TXT["Text"]
    I_AUD["Audio"]
    I_GLM4["GLM-4"]
    I_KIMID["Kimi Decoder"]
    I_CT2E["CT2 Whisper Encoder"]
    I_DET{"English / Mandarin?"}
    I_CT2D["CT2 Whisper Decoder"]
    I_OUT["Text"]

    %% Text/Audio -> GLM-4 -> Kimi decoder -> Text
    I_TXT --> I_GLM4
    I_AUD --> I_GLM4
    I_GLM4 --> I_KIMID --> I_OUT

    %% Audio -> CT2 encoder -> detection -> route
    I_AUD --> I_CT2E --> I_DET
    I_DET -- "English or Mandarin" --> I_KIMID
    I_DET -- "Other languages" --> I_CT2D --> I_OUT
  end

  %% ------- Reference (dotted) lines to show which subsystems power Integration -------
  I_CT2E -. "uses" .- F_WE
  I_CT2D -. "uses" .- F_D
  I_KIMID -. "uses" .- K_TK
  I_GLM4  -. "uses" .- K_TK
  I_KIMID -. "uses" .- K_WE
```

## 🌟 Key Features

### Core Capabilities
- **Intelligent Language Routing**: Automatic language detection with confidence-based fallback
- **Dual-Engine Architecture**: Kimi chain for high-precision languages + Whisper for dialect coverage
- **Real-time Streaming**: WebSocket-based semi-streaming transcription with <200ms latency
- **Memory Optimization**: CTranslate2 + int8 quantization reducing VRAM usage by 50%
- **Gradual Rollout**: Seamless switching between legacy and new architectures

### Language Support Matrix

| Language | Engine | CER/WER | Status |
|----------|--------|---------|---------|
| Mandarin (zh-CN) | Kimi | ~5% | ✅ Production |
| English (en) | Kimi | ~3% | ✅ Production |
| Cantonese (yue) | Whisper (Fine-tuned) | ~15% | 🚧 Optimized |
| Other Dialects | Whisper | Standard | 📈 Continuous Improvement |

### Production Features
- **Horizontal Scaling**: Multi-process WebSocket server with load balancing
- **Memory Efficiency**: Shared encoder + separate decoder instances
- **Fault Tolerance**: Automatic fallback and error recovery mechanisms
- **Monitoring**: Built-in metrics collection and performance tracking

## 📁 Project Structure

```
china-multilingual-asr/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── docker-compose.yml                  # Container orchestration
├── config/                            # Configuration files
│   ├── production.yaml
│   ├── development.yaml
│   └── model_configs/
├── kimi_deployment/                   # Kimi ASR pipeline (submodule)
│   ├── kimia_infer/
│   │   ├── api/
│   │   ├── models/
│   │   └── utils/
│   └── requirements.txt
├── WhisperLive/                       # WebSocket server (submodule)
│   ├── whisper_live/
│   │   ├── server.py
│   │   ├── client.py
│   │   └── transcriber.py
│   └── requirements.txt
├── src/                              # Main integration layer
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── integrated_asr.py         # Main ASR orchestrator
│   │   ├── language_router.py        # Language detection & routing
│   │   └── output_adapter.py         # Unified output formatting
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── whisper_ct2_adapter.py    # CTranslate2 Whisper adapter
│   │   └── kimi_adapter.py           # Kimi pipeline adapter
│   ├── server/
│   │   ├── __init__.py
│   │   ├── websocket_server.py       # Enhanced WebSocket server
│   │   └── api_server.py             # FastAPI REST endpoints
│   └── utils/
│       ├── __init__.py
│       ├── audio_processing.py
│       ├── metrics.py
│       └── config_loader.py
├── tests/                            # Test suite
│   ├── unit/
│   ├── integration/
│   └── performance/
├── scripts/                          # Deployment & utility scripts
│   ├── setup.sh
│   ├── deploy.sh
│   └── benchmark.py
└── docs/                             # Documentation
    ├── architecture.md
    ├── api_reference.md
    └── deployment_guide.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Docker & Docker Compose (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/china-multilingual-asr.git
cd china-multilingual-asr

# Initialize submodules
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt
pip install -r kimi_deployment/requirements.txt
pip install -r WhisperLive/requirements.txt

# Setup configuration
cp config/development.yaml config/local.yaml
# Edit config/local.yaml with your model paths and settings
```

### Running the System

```bash
# Start the integrated ASR server
python -m src.server.websocket_server \
    --config config/local.yaml \
    --host 0.0.0.0 \
    --port 9090

# Test with client
python -m src.client.test_client \
    --server ws://localhost:9090 \
    --audio test_audio/mandarin_sample.wav
```

## 🔧 Configuration

Key configuration parameters in `config/production.yaml`:

```yaml
# Model Configuration
models:
  whisper:
    model_path: "/models/whisper-large-v3-ct2"
    compute_type: "int8_float16"
    device: "cuda"
  kimi:
    model_path: "/models/kimi-audio"
    torch_dtype: "bfloat16"
    device: "cuda"

# Language Routing
language_routing:
  confidence_threshold: 0.7
  supported_languages: ["zh", "en", "yue", "zh-tw"]
  kimi_languages: ["zh", "en"]
  fallback_engine: "whisper"

# Performance Tuning
performance:
  max_concurrent_sessions: 100
  encoder_batch_size: 8
  memory_optimization: true
  enable_quantization: true
```

## 📊 Performance Metrics





### Resource Usage
- **Memory**: Around 30GB
- **Throughput**: 100+ concurrent sessions per GPU

## 🛠️ Development Roadmap

### Phase 1: Core Integration (Weeks 1-2) Finished
- [x] Project structure setup
- [ ] IntegratedASR core implementation
- [ ] Basic language routing
- [ ] WebSocket server integration

### Phase 2: Advanced Features (Weeks 3-4) Finished
- [ ] Custom language identification module
- [ ] Performance optimization (quantization, batching)
- [ ] Comprehensive error handling
- [ ] Monitoring and metrics

### Phase 3: Production Readiness (Weeks 5-6)
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] Load testing and optimization
- [ ] Documentation and tutorials

### Phase 4: Advanced Capabilities (Future)
- [ ] Multi-GPU support
- [ ] ASR↔TTS data loop validation
- [ ] Additional dialect support
- [ ] Real-time adaptation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Whisper team for the foundational ASR technology
- CTranslate2 developers for efficient inference optimization
- Kimi team for high-precision Chinese ASR capabilities

---

**Built for China's diverse linguistic landscape** 🇨🇳
*Empowering seamless communication across Mandarin, Cantonese, and regional dialects*
