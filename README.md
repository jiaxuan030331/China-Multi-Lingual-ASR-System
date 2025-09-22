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
  %% Kimi (Embeddings path)
  %% =======================
  subgraph KIMI["Kimi (Embeddings)"]
    direction LR
    K_T["Text"]:::kimi
    K_A["Audio"]:::kimi
    K_TK["GLM-4 Tokenizer"]:::kimi
    K_WE["Whisper Encoder"]:::kimi
    K_E["Embedding Layer"]:::kimi

    K_T --> K_TK --> K_E
    K_A --> K_TK
    K_A --> K_WE --> K_E
  end

  %% ============================
  %% faster-whisper (CT2) system
  %% ============================
  subgraph FW["faster-whisper (CT2)"]
    direction LR
    F_A["Audio"]:::fw
    F_WE["Whisper Encoder"]:::fw
    F_LD{"Language Detection"}:::decision
    F_D["Decoder"]:::fw
    F_TXT["Text"]:::fw

    F_A --> F_WE --> F_LD --> F_D --> F_TXT
  end

  %% --------- Styles (Contrasted Colors) ---------
  classDef kimi fill:#E6F2FF,stroke:#1E90FF,color:#0B3D91,stroke-width:1.5px;
  classDef fw fill:#FFF5E6,stroke:#FF8C00,color:#7F4F00,stroke-width:1.5px;
  classDef decision fill:#FFF0F6,stroke:#C71585,color:#5B1A42,stroke-width:1.5px;
```
```mermaid
flowchart LR
  %% =======================
  %% Project Integration
  %% =======================
  subgraph INT["Integration (Project flow)"]
    direction LR
    I_TXT["Text"]:::int
    I_AUD["Audio"]:::int
    I_GLM4["GLM-4"]:::int
    I_KIMID["Kimi Decoder"]:::int
    I_CT2E["CT2 Whisper Encoder"]:::int
    I_DET{"English / Mandarin?"}:::decision
    I_CT2D["CT2 Whisper Decoder"]:::int
    I_OUT["Text"]:::int

    %% Text/Audio -> GLM-4 -> Kimi decoder -> Text
    I_TXT --> I_GLM4
    I_AUD --> I_GLM4
    I_GLM4 --> I_KIMID --> I_OUT

    %% Audio -> CT2 encoder -> detection -> route
    I_AUD --> I_CT2E --> I_DET
    I_DET -- "English or Mandarin" --> I_KIMID
    I_DET -- "Other languages" --> I_CT2D --> I_OUT
  end

  %% --------- Styles (Contrasted Colors) ---------
  classDef int fill:#E6FFEE,stroke:#2E8B57,color:#0F5132,stroke-width:1.5px;
  classDef decision fill:#FFF0F6,stroke:#C71585,color:#5B1A42,stroke-width:1.5px;
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
