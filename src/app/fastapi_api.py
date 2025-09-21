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

# 提前注入启动参数
app.state.ct2_model_path = os.environ.get("CT2_MODEL_PATH", "large-v3")
app.state.kimi_model_path = os.environ.get("KIMI_MODEL_PATH", "moonshotai/Kimi-Audio-7B-Instruct")
app.state.lid_model_path = os.environ.get("LID_MODEL_PATH", "/workspace/ASR/WhisperLive/language_fnn_only2.pt")
app.state.confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))
# AUTO_WARMUP=true 启动时自动预热，false 则需手动调用 /warmup

async def warmup_model():
    """
    模型预热：使用短音频进行几次推理，预加载GPU内核和缓存
    """
   
    
    print("🔥 开始模型预热...")
    
    try:
        # 创建短音频进行预热 (1秒, 2秒, 3秒)
        warmup_audios = [
            np.random.randn(16000).astype(np.float32) * 0.01,      # 1秒
            np.random.randn(16000 * 2).astype(np.float32) * 0.01,  # 2秒
            np.random.randn(16000 * 3).astype(np.float32) * 0.01,  # 3秒
        ]
        
        for i, audio in enumerate(warmup_audios, 1):
            
            
            # 执行一次完整的转写流程
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
                print(f"   预热完成，引擎: {result.get('engine', 'unknown')}")
            else:
                print(f"   ⚠️ 预热警告: {result.get('error', 'unknown')}")
        
        print("🔥 模型预热完成！")
        
    except Exception as e:
        print(f"⚠️ 预热过程出错: {e}")

@app.on_event("startup")
async def startup_event():
    """
    FastAPI 启动时加载 IntegratedASR 模型，使用 app.state 中注入的启动配置。
    """
    global model

    ct2_model_path = app.state.ct2_model_path
    kimi_model_path = app.state.kimi_model_path
    lid_model_path = app.state.lid_model_path
    confidence_threshold = app.state.confidence_threshold

    # 加载模型
    print("📥 加载 IntegratedASR 模型...")
    model = load_integrated_asr(
        ct2_model_path=ct2_model_path,
        kimi_model_path_or_name=kimi_model_path,
        lid_model_path=lid_model_path,
        confidence_threshold=confidence_threshold
    )
    print("✅ 模型加载完成")
    
    # 自动预热（可通过环境变量控制）
    auto_warmup = os.environ.get("AUTO_WARMUP", "true").lower() == "true"
    if auto_warmup:
        await warmup_model()
    else:
        print("⚠️ 自动预热已禁用，可调用 /warmup 手动预热")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    prep_timeout: Optional[int] = Form(60),           # 准备阶段超时
):
    """
    通用音频转写接口
    支持多种音频格式(MP3/WAV等)，自动路由到最适合的引擎
    
    参数:
    - file: 上传的音频文件
    - prep_timeout: 准备阶段超时时间(秒)
    
    返回:
    - JSON格式的转写结果
    """
    try:
        contents = await file.read()
        
        # 将模型推理部分放入线程池中执行，防止阻塞事件循环
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
    WebSocket/流式音频转写接口
    接收原始PCM音频数据进行实时转写
    格式完全匹配 kimi_deployment 的返回结构
    
    Headers:
    - prep_timeout: 准备阶段超时时间(可选)
    - language: 语言提示(可选)
    - prompt: 转写提示(可选)
    
    Body:
    - 原始PCM音频数据 (float32格式)
    
    返回:
    - 兼容WebSocket格式的JSON结果，与kimi_deployment完全一致
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
                "result": [segment['text']],  # 只包含segment信息，text在segment中
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