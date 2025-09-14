import logging
import tempfile
import soundfile as sf
import numpy as np
import io
from pydub import AudioSegment
from kimi_deployment.app.load_model import model_lock  # 线程锁，保证模型线程安全
from typing import Union
import os
import librosa
import torch 
logger = logging.getLogger(__name__)
from pathlib import Path






import librosa
import os
from loguru import logger

  # 若未定义，确保线程安全

import numpy as np
import torch
from typing import Union

def transcribe_from_waveform(model, waveform, prompt=None, language=None, transcribe_params=None) -> dict:
    """
    使用 KimiAudio 模型进行音频转写。
    这里统一将 waveform 转为 float32 再送入模型（支持 numpy 或 torch）。
    """
    import numpy as np
    import torch

    try:
        # 统一 float32（支持 numpy 或 torch）
        if isinstance(waveform, np.ndarray):
            # 清理 NaN/Inf，转 float32，避免额外拷贝
            wf = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        elif isinstance(waveform, torch.Tensor):
            # 清理 NaN/Inf，转 float32（保留设备/不改变形状）
            wf = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
        else:
            # 其他类型先转 numpy 再处理
            wf = np.asarray(waveform)
            wf = np.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # 默认采样参数（保持不变）
        sampling_params = transcribe_params or {
            "audio_temperature": 0,
            "audio_top_k": 1,
            "text_temperature": 0,
            "text_top_k": 1,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

        # 构造 prompt（保持不变）
        prompt_text = prompt.strip() if prompt else "Please transcribe the audio"
        if language:
            prompt_text += f" in {language}"

        #logger.info(f"[DEBUG] Using prompt: {prompt_text}")

        # 推理（保持不变）
        with model_lock:
            text_output = model.generate_from_waveform(
                waveform=wf,
                prompt=prompt_text,
                **sampling_params
            )

        return {
            "status": 0,
            "text": text_output,
            "language": language or "Auto detection",
            "prompt": prompt_text
        }

    except Exception as e:
        logger.exception("ASR 推理失败")
        return {
            "status": -1,
            "error": str(e)
        }


    

def transcribe_from_path(model, audio_path: str, prompt=None, language=None, transcribe_params=None) -> dict:
    """
    使用 KimiAudio 模型进行音频转写。

    参数:
    - model: 已加载的 KimiAudio 实例
    - audio_path: 本地音频文件路径
    - prompt: 引导提示词（可选，仅用于语言 hint）
    - language: 控制语言偏向（拼接入 prompt）
    - transcribe_params: 可选采样参数（温度、top_k 等）

    返回:
    - dict，包括识别文本、语言、文件名、时长等信息
    """
    try:
        # 默认采样参数
        sampling_params = transcribe_params or {
            "audio_temperature": 0,
            "audio_top_k": 1,
            "text_temperature": 0,
            "text_top_k": 1,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

        # 构造 prompt 文字（语言作为提示）
        prompt_text = prompt.strip() if prompt else "Please transcribe the audio"
        if language:
            prompt_text += f" in {language}"

        logger.info(f"[DEBUG] Using prompt: {prompt_text}")

        # 读取音频 waveform
        wav, _ = librosa.load(audio_path, sr=16000)
        

        # 推理调用（线程锁防并发）
        with model_lock:
            text_output = model.generate_from_waveform(
                waveform = wav,
                prompt=prompt_text,
                **sampling_params
            )

        logger.info(f">>> Transcription Success: {audio_path}\n{text_output}")

        duration = librosa.get_duration(path=audio_path)

        return {
            "status": 0,
            "text": text_output,
            "language": language or "auto detection",
            "prompt": prompt_text,
            "file": os.path.basename(audio_path),
            "duration": round(duration, 2)
        }

    except Exception as e:
        logger.exception("ASR 推理失败")
        return {
            "status": -1,
            "error": str(e)
        }


def load_waveform_from_bytes(audio_bytes: bytes, preferred_sr=16000) -> np.ndarray:
    """将上传的音频字节流转换为 float32 waveform，支持多种格式。"""
    try:
        with io.BytesIO(audio_bytes) as f:
            waveform, sr = sf.read(f)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # 转单声道
        if sr != preferred_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=preferred_sr)
        logger.info("[Audio] soundfile 读取成功")
        return waveform.astype(np.float32)
    except Exception:
        logger.warning("[Audio] soundfile 失败，尝试 pydub 解码")

    # fallback to pydub
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(preferred_sr).set_channels(1).set_sample_width(2)

        pcm = io.BytesIO()
        audio.export(pcm, format="wav")
        pcm.seek(0)

        waveform, sr = librosa.load(pcm, sr=None)
        if sr != preferred_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=preferred_sr)
        logger.info("[Audio] pydub 读取成功")
        return waveform.astype(np.float32)
    except Exception as e:
        logger.error(f"[Audio] 解码失败: {e}")
        raise RuntimeError(f"音频解码失败: {e}")

def transcribe_from_bytes(
    model,
    audio_bytes: bytes,
    prompt: str = None,
    language: str = None,
    transcribe_params: dict = None
) -> dict:
    """
    从音频字节流进行 Kimi ASR 推理，带线程锁和日志。
    返回 JSON 格式：{status, text, error, ...}
    """
    try:
        # 解码 waveform
        waveform = load_waveform_from_bytes(audio_bytes)

        # 设置提示词
        prompt_text = prompt.strip() if prompt else "Please transcribe the audio"
        if language:
            prompt_text += f" in {language}"

        logger.info(f"[ASR] Prompt: {prompt_text}")

        # 推理参数
        sampling_params = transcribe_params or {
            "audio_temperature": 0,
            "audio_top_k": 1,
            "text_temperature": 0,
            "text_top_k": 1,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

        with model_lock:
            text = model.generate_from_waveform(
                waveform = waveform,
                prompt=prompt_text,
                **sampling_params
            )

        logger.info(f"[ASR] Success: {text}")
        duration = librosa.get_duration(y=waveform, sr=16000)

        return {
            "status": 0,
            "text": text,
            "language": language or "auto",
            "prompt": prompt_text,
            "duration": round(duration, 2)
        }

    except Exception as e:
        logger.exception("[ASR] 转写失败")
        return {
            "status": -1,
            "error": str(e)
        }




def transcribe_auto(
    model,
    audio: Union[str, Path, bytes, np.ndarray, torch.Tensor],
    prompt: str = "请转写以下音频",
    language: str = None,
    transcribe_params: dict = None,
) -> str:
    """
    自动识别音频格式并调用对应的转写流程。
    支持输入类型: 路径、bytes、numpy、Tensor

    返回：识别出的文本
    """
    
    # Case 1: 文件路径
    if isinstance(audio, (str, Path)):
        return transcribe_from_path(
            model=model,
            audio_path=str(audio),
            prompt=prompt,
            language=language,
            transcribe_params=transcribe_params
        )

    # Case 2: 字节流（如 HTTP API 中接收的）
    elif isinstance(audio, bytes):
        return transcribe_from_bytes(
            model=model,
            audio_bytes=audio,
            prompt=prompt,
            language=language,
            transcribe_params=transcribe_params
        )

    # Case 3: numpy 或 torch waveform
    elif isinstance(audio, (np.ndarray, torch.Tensor)):
        return transcribe_from_waveform(
            model=model,
            waveform=audio,
            prompt=prompt,
            language=language,
            transcribe_params=transcribe_params
        )

    else:
        raise TypeError(f"[transcribe_auto] Unsupported audio input type: {type(audio)}")

            
    