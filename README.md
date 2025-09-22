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
flowchart LR
  %% =======================
  %% Integrated System
  %% =======================
  subgraph INT["Integrated System (Project Flow)"]
    direction TB
    I_TXT["Text"]:::int
    I_AUD["Audio"]:::int

    %% Text path
    I_GLM4["GLM-4 (LLM)"]:::int
    I_KIMID["Kimi Decoder (token→text)"]:::int
    I_OUT1["Text"]:::int

    %% Audio path
    I_CT2E["CT2 Whisper Encoder (audio→embeddings)"]:::int
    I_DET{"Router: English/Mandarin?"}:::decision
    I_CT2D["CT2 Whisper Decoder (ASR)"]:::int
    I_OUT2["Text"]:::int

    %% Flows
    I_TXT --> I_GLM4 --> I_KIMID --> I_OUT1
    I_AUD --> I_GLM4
    I_AUD --> I_CT2E --> I_DET
    I_DET -- "EN/ZH" --> I_KIMID
    I_DET -- "Other" --> I_CT2D --> I_OUT2
  end

  %% Concise explanations (notes)
  Note1["GLM-4: general reasoning; shares tokenizer with Kimi"]:::note
  Note2["Kimi Decoder: converts GLM tokens to text output"]:::note
  Note3["CT2: high-speed Whisper inference"]:::note

  Note1 -.-> I_GLM4
  Note2 -.-> I_KIMID
  Note3 -.-> I_CT2E

  %% --------- Styles ---------
  classDef int fill:#E6FFEE,stroke:#2E8B57,color:#0F5132,stroke-width:1.5px;
  classDef decision fill:#FFF0F6,stroke:#C71585,color:#5B1A42,stroke-width:1.5px;
  classDef note fill:#F8F9FA,stroke:#A0A0A0,color:#333,stroke-dasharray:3 3,stroke-width:1px;

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
