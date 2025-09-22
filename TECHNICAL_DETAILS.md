# Technical Details

This document summarizes the design and implementation details of the Integrated ASR service.

## Repository Structure
```
ASR/
  models/
    lid/                          # Language ID models (e.g., language_fnn_only2.pt)
  scripts/
    run_integrated_asr.sh         # Start FastAPI API
    run_websocket_server.sh       # Start streaming proxy server
  src/
    app/
      fastapi_api.py              # FastAPI app (endpoints: /transcribe, /transcribe_websocket, /warmup)
      load_model.py               # Model lifecycle (singleton + locking)
      transcribe.py               # App-level transcription wrappers
    core/
      integrated_asr.py           # Orchestration, routing, timing
    backends/
      faster_whisper_transcriber.py
      kimia_infer/                # Kimi tokenizer & helpers (adapted)
    websocket/
      server.py                   # Frame-based streaming server
      conf/config.ini             # Target FastAPI URL
  tests/
    backend_demo.py               # Basic backend test
    fastapi_demo.py               # API demo
    websocket_demo.py             # Streaming demo
  README.md
  TECHNICAL_DETAILS.md
  requirements.txt
```

## Architecture
- Dual-encoder/dual-backend design:
  - Faster-Whisper (CTranslate2) for broad multilingual coverage and efficient decoding
  - Kimi (GLM4) for high-accuracy English/Mandarin (and zh/en/yue) paths
- Language ID (FNN) used to route: zh/en/yue → Kimi; others → CT2
- Orchestration (`src/core/integrated_asr.py`):
  - Parallel prepare phase:
    - CT2: feature extraction + encoder + LID
    - Kimi: audio tokenization + prompt/text tokens
  - Route based on LID confidence; decode only the selected chain

## Data Flow & Interop
- Audio inputs normalized to mono, 16 kHz, float32 (numpy)
- CT2 feature/encoder outputs bridged to Kimi format:
  - CT2 `StorageView` → CPU float32 → torch.bfloat16 → (B, T/4, H*4) padding to mult. of 4
- Tokenizers:
  - Faster-Whisper: `faster_whisper.tokenizer.Tokenizer`
  - Kimi: GLM4 tokenizer (audio + text) from `kimia_infer`

## Key Modules
- `src/core/integrated_asr.py`: routing, timing, options, high-level API
- `src/backends/faster_whisper_transcriber.py`: adapted CT2 transcription/segmentation
- `src/backends/kimia_infer/...`: Kimi tokenizer + APIs (adapted)
- `src/app/fastapi_api.py`: FastAPI app; `/transcribe`, `/transcribe_websocket`, `/warmup`, `/health`
- `src/websocket/server.py`: frame-based streaming proxy; forwards to `/transcribe_websocket`

## Endpoints (FastAPI)
- `POST /transcribe` (file upload)
  - Returns `{ status, text, language, confidence, engine, total_time }`
- `POST /transcribe_websocket` (raw PCM float32 body)
  - Returns `{ status, result: [segment], info: { language, language_probability, engine } }`
- `POST /warmup`, `GET /health`

## Performance & Resources
- Memory footprint (single GPU): ~24 GB total
  - Faster-Whisper: ~6 GB
  - Kimi: ~23 GB
- Throughput
  - Faster-Whisper: matches baseline speed
  - Kimi: encoder ~0.1 s faster
- Warmup recommended for lower cold-start latency

## Configuration
- Defaults via environment variables or script flags:
  - CT2_MODEL_PATH, KIMI_MODEL_PATH, LID_MODEL_PATH
  - CONFIDENCE_THRESHOLD (default 0.7)
  - AUTO_WARMUP (`true`/`false`)

## Notes
- Use `--workers 1` unless you plan for multi-GPU or ample VRAM
- Hugging Face caches redirected to `/workspace/.cache/huggingface` 