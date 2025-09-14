# app/fastapi_api.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from kimi_deployment.app.load_model import load_kimi_model, model_lock
from kimi_deployment.app.transcribe import transcribe_auto
from fastapi.concurrency import run_in_threadpool
import os
from typing import Optional
import io

from fastapi import Form
app = FastAPI(title="Kimi Audio ASR API")
import json
from fastapi import Request

# 提前注入启动参数
app.state.model_path = os.environ.get("KIMI_MODEL_PATH", None)
app.state.device = os.environ.get("KIMI_DEVICE", "cuda")
app.state.device_index = int(os.environ.get("KIMI_DEVICE_INDEX", 0))
app.state.torch_dtype = os.environ.get("KIMI_TORCH_DTYPE", "bfloat16")

@app.on_event("startup")
async def startup_event():
    """
    FastAPI 启动时加载 Kimi 模型，使用 app.state 中注入的启动配置。
    """
    global model

    model_path = app.state.model_path
    device = app.state.device
    device_index = app.state.device_index
    torch_dtype = app.state.torch_dtype

    model = load_kimi_model(
        model_path=model_path,
        device=device,
        device_index=device_index,
        torch_dtype=torch_dtype,
    )

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),           # ✅ 默认 None
    language: Optional[str] = Form(None),         # ✅ 默认 None
    sampling_params: Optional[str] = Form(None),  # ✅ 默认 None
):
    try:
        contents = await file.read()
        # 尝试解析 sampling_params 为字典
        params = None
        if sampling_params:
            params = json.loads(sampling_params)
        # 将模型推理部分放入线程池中执行，防止阻塞事件循环
        result = await run_in_threadpool(
            transcribe_auto,
            model=model,
            audio = contents,
            prompt=prompt,
            language=language,
            transcribe_params = params

        )

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




from app.transcribe import transcribe_from_waveform
import numpy as np

@app.post("/transcribe_websocket")
async def transcribe_pcm(request: Request):
    try:
        audio_bytes = await request.body()

        language = request.headers.get("language", None)
        prompt = request.headers.get("prompt", "请转写以下音频")

        # 直接送入 waveform 处理
        waveform = np.frombuffer(audio_bytes, dtype=np.float32)

        segments = await run_in_threadpool(
            transcribe_from_waveform,
            model=model,
            waveform=waveform,
            prompt=prompt,
            language=language,
            transcribe_params=None
        )
        segments['start'] = 0
        segments['end'] = len(waveform) / 16000  # 假设采样率为 16000 Hz
        print(segments)
            

       # 构建最终 JSON 返回值
        
        return JSONResponse(content={
            "result":[segments],
            "info": None,
            "status": 0
        })

        
        
    

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
