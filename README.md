# Integrated ASR (Whisper CT2 + Kimi GLM4)

## Introduction
An ASR service that combines Faster-Whisper (CTranslate2) and Kimi (GLM4) with language-based routing. It leverages Kimi’s high accuracy for English and Mandarin, while keeping Whisper’s broad-language coverage (≈100 languages) and strong finetuning ecosystem for less-popular languages. Ships with a FastAPI HTTP API and a lightweight streaming server.

## Highlights
- Dual backends with automatic routing: zh/en/yue → Kimi, others → Faster-Whisper
- High-accuracy path for English/Mandarin via Kimi; broad multilingual coverage via Whisper
- Easy to finetune Whisper for long-tail languages and still benefit from its multilingual knowledge
- Customizable Language ID routing to mitigate Whisper’s lower precision on certain long-tail languages
- Simple startup scripts; optional warmup for lower cold start
- Clean adapters: CT2 ↔ Kimi feature bridge, dtype-safe pipelines

[Read the Technical Details →](./TECHNICAL_DETAILS.md)

## Structure Overview
```
ASR/
  models/lid/                    # Language-ID model(s)
  scripts/
    run_integrated_asr.sh        # FastAPI server
    run_websocket_server.sh      # Streaming proxy server
  src/
    app/     (FastAPI, app helpers)
    core/    (integrated_asr.py: routing/orchestration)
    backends/(faster_whisper_transcriber.py, kimia_infer/...)
    websocket/(server.py, conf/config.ini)
  tests/                         # small demo scripts
```

### Architecture (Mermaid)
```mermaid
flowchart TD
  subgraph KIMI["Kimi (Embeddings)"]
    direction TB
    K_T[Text]
    K_A[Audio]
    K_TK[GLM-4 Tokenizer]
    K_WE[Whisper Encoder]
    K_E[Embedding Layer]

    K_T --> K_TK --> K_E
    K_A --> K_TK
    K_A --> K_WE --> K_E
  end

  subgraph FW["faster-whisper (CT2)"]
    direction TB
    F_A[Audio]
    F_WE[Whisper Encoder]
    F_LD{Language Detection}
    F_D[Decoder]
    F_TXT[Text]

    F_A --> F_WE --> F_LD --> F_D --> F_TXT
  end
'''
 

## Minimal Demo
- Start API (default port 8001):
```
./scripts/run_integrated_asr.sh --no-warmup
```
- Transcribe a file:
```
curl -X POST -F "file=@audio_examples/mandarin.mp3" \
  http://127.0.0.1:8001/transcribe
```
- Start streaming proxy (default port 9092):
```
./scripts/run_websocket_server.sh
```
Ensure `src/websocket/conf/config.ini` targets:
```
[model]
http_url=http://127.0.0.1:8001/transcribe_websocket
```
## Demo
- Screen recording: [demo_screen_recording.mp4](./demo_screen_recording.mov)


## Raw Performance (single GPU)
- Memory footprint: ~24 GB total
  - Faster-Whisper: ~6 GB
  - Kimi: ~23 GB
- Throughput
  - Faster-Whisper: same speed as baseline (no regression)
  - Kimi: encoder ~0.1 s faster than baseline

## Credits
- Kimi-Audio (GLM4) by MoonshotAI: [Kimi-Audio GitHub](https://github.com/MoonshotAI/Kimi-Audio)
- Faster-Whisper by SYSTRAN: [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- Whisper by OpenAI: [whisper GitHub](https://github.com/openai/whisper)
