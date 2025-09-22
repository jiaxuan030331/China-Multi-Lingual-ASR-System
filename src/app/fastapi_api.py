# src/app/fastapi_api.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from src.app.load_model import load_integrated_asr, model_lock
from src.app.transcribe import transcribe_auto, transcribe_from_waveform
from fastapi.concurrency import run_in_threadpool
import os
from typing import Optional
import io
import json
import numpy as np
from fastapi import Form

app = FastAPI(title="IntegratedASR API")

# Pre-inject startup parameters
app.state.ct2_model_path = os.environ.get("CT2_MODEL_PATH", "large-v3")
app.state.kimi_model_path = os.environ.get("KIMI_MODEL_PATH", "moonshotai/Kimi-Audio-7B-Instruct")
app.state.lid_model_path = os.environ.get("LID_MODEL_PATH", "/workspace/ASR/models/lid/language_fnn_only2.pt")
app.state.confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))
# AUTO_WARMUP=true auto warmup on startup; false requires manual /warmup

async def warmup_model():
    """
    Model warmup: run a few short audio inferences to preload GPU kernels & caches
    """
   
    
    print("Starting model warmup...")
    
    try:
        # Create short audios for warmup (1s, 2s, 3s)
        warmup_audios = [
            np.random.randn(16000).astype(np.float32) * 0.01,      # 1 second
            np.random.randn(16000 * 2).astype(np.float32) * 0.01,  # 2 seconds
            np.random.randn(16000 * 3).astype(np.float32) * 0.01,  # 3 seconds
        ]
        
        for i, audio in enumerate(warmup_audios, 1):
            
            
            # Run a full transcription pipeline once
            result = await run_in_threadpool(
                transcribe_from_waveform,
                model=model,
                waveform=audio,
                prep_timeout=30,
                ct2_only=True
            )
            result = await run_in_threadpool(
                transcribe_from_waveform,
                model=model,
                waveform=audio,
                prep_timeout=30,
                kimi_only=True
            )
            
            if result["status"] == 0:
                print(f"   Warmup completed, engine: {result.get('engine', 'unknown')}")
            else:
                print(f"   ⚠️ Warmup warning: {result.get('error', 'unknown')}")
        
        print("Model warmup done!")
        
    except Exception as e:
        print(f"⚠️ Warmup error: {e}")

@app.on_event("startup")
async def startup_event():
    """
    Load IntegratedASR model at FastAPI startup using injected app.state configuration.
    """
    global model

    ct2_model_path = app.state.ct2_model_path
    kimi_model_path = app.state.kimi_model_path
    lid_model_path = app.state.lid_model_path
    confidence_threshold = app.state.confidence_threshold

    # Load model
    print("📥 Loading IntegratedASR model...")
    model = load_integrated_asr(
        ct2_model_path=ct2_model_path,
        kimi_model_path_or_name=kimi_model_path,
        lid_model_path=lid_model_path,
        confidence_threshold=confidence_threshold
    )
    print("✅ Model loaded")
    
    # Auto warmup (controllable via env)
    auto_warmup = os.environ.get("AUTO_WARMUP", "true").lower() == "true"
    if auto_warmup:
        await warmup_model()
    else:
        print("⚠️ Auto warmup disabled, call /warmup to warm manually")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    prep_timeout: Optional[int] = Form(60),           # Preparation timeout
):
    """
    General audio transcription endpoint
    Supports multiple formats (MP3/WAV, etc.) and routes to the best engine automatically
    
    Args:
    - file: uploaded audio file
    - prep_timeout: preparation timeout (seconds)
    
    Returns:
    - JSON transcription result
    """
    try:
        contents = await file.read()
        
        # Run inference in threadpool to avoid blocking event loop
        result = await run_in_threadpool(
            transcribe_auto,
            model=model,
            audio=contents,
            prep_timeout=prep_timeout,
            kimi_only=False,
            ct2_only=False
        )
        print(result)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/transcribe_websocket")
async def transcribe_pcm(request: Request):
    """
    WebSocket/streaming audio transcription endpoint
    Accepts raw PCM frames for real-time transcription
    Response format fully matches kimi_deployment
    
    Headers:
    - prep_timeout: preparation timeout (optional)
    - language: language hint (optional)
    - engine: force engine (optional)
    """
    try:
        audio_bytes = await request.body()

        # 从请求头获取参数
        prep_timeout = int(request.headers.get("prep_timeout", "60"))
        

        # 直接解析PCM数据 (假设为float32格式)
        waveform = np.frombuffer(audio_bytes, dtype=np.float32)

        # 使用线程池执行转写
        result = await run_in_threadpool(
            transcribe_from_waveform,
            model=model,
            waveform=waveform,
            prep_timeout=prep_timeout
        )
        
        # 构建segment对象，只包含text和必要信息
        if result["status"] == 0:
            segment = {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "engine": result.get("engine", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "total_time": result.get("total_time", 0.0),
                "start": 0,
                "end": len(waveform) / 16000  # 假设采样率为 16000 Hz
            }
            
            # 构建兼容WebSocket的返回格式 - 完全匹配kimi_deployment
            return JSONResponse(content={
                "result": [segment],  # 只包含segment信息，text在segment中
                "info": {
                    "language": result.get("language", "unknown"),
                    "language_probability": result.get("confidence", 0.0),  # 关键：添加language_probability
                    "engine": result.get("engine", "unknown")
                },
                "status": 0
            })
        else:
            # 错误情况
            return JSONResponse(content={
                "result": [],
                "info": None,
                "status": -1,
                "error": result.get("error", "Unknown error")
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "result": [],
            "info": None,
            "status": -1,
            "error": str(e)
        })
@app.post("/warmup")
async def manual_warmup():
    """
    手动预热接口
    在生产环境中可以在流量到来前主动调用此接口进行预热
    """
    try:
        await warmup_model()
        return JSONResponse(content={
            "status": "success",
            "message": "模型预热完成",
            "service": "IntegratedASR"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"预热失败: {str(e)}",
                "service": "IntegratedASR"
            }
        )

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    try:
        # 简单检查模型是否加载
        if 'model' in globals() and model is not None:
            return JSONResponse(content={
                "status": "healthy",
                "model_loaded": True,
                "service": "IntegratedASR"
            })
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy", 
                    "model_loaded": False,
                    "service": "IntegratedASR"
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "service": "IntegratedASR"
            }
        ) 