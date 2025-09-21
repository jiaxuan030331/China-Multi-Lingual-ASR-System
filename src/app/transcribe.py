import logging
import tempfile
import soundfile as sf
import numpy as np
import io
from pydub import AudioSegment
from src.app.load_model import model_lock  # 线程锁，保证模型线程安全
from typing import Union
import os
import librosa
import torch 
from pathlib import Path

logger = logging.getLogger(__name__)

def load_audio_safe(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    安全加载音频文件为16kHz单声道float32格式，支持MP3/WAV等多种格式
    
    参数:
    - audio_path: 音频文件路径
    - sr: 目标采样率，默认16000Hz
    
    返回:
    - np.ndarray: 单声道float32音频数组
    """
    try:
        # 优先使用librosa，支持MP3等格式
        audio, sr = librosa.load(audio_path, sr=sr, mono=True)
        return audio, sr
    except Exception as e:
        logger.warning(f"[Audio] librosa加载失败，尝试备用方案: {e}")
        
        # 备用方案：soundfile + pydub
        try:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            return load_waveform_from_bytes(audio_bytes, preferred_sr=sr)
        except Exception as e2:
            logger.error(f"[Audio] 所有加载方案失败: {e2}")
            raise RuntimeError(f"音频文件加载失败: {audio_path}")

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

def normalize_audio_input(audio_input) -> np.ndarray:
    """
    标准化音频输入为np.ndarray格式
    
    参数:
    - audio_input: torch.Tensor, np.ndarray, list等
    
    返回:
    - np.ndarray: 标准化后的音频数组
    """
    if isinstance(audio_input, torch.Tensor):
        # 处理torch.Tensor
        if audio_input.ndim == 2 and audio_input.shape[0] > 1:
            # 立体声转单声道
            audio_input = audio_input.mean(dim=0)
        elif audio_input.ndim == 2:
            # 单声道但有额外维度
            audio_input = audio_input.squeeze(0)
        audio_np = audio_input.detach().cpu().numpy()
    elif isinstance(audio_input, np.ndarray):
        audio_np = audio_input.copy()
        if audio_np.ndim == 2:
            # 立体声转单声道
            audio_np = audio_np.mean(axis=0) if audio_np.shape[0] <= audio_np.shape[1] else audio_np.mean(axis=1)
    else:
        # 其他类型转numpy
        audio_np = np.asarray(audio_input)
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
    
    # 清理异常值并转float32
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return audio_np

def transcribe_from_waveform(
    model, 
    waveform, 
    prep_timeout: int = 60,
    kimi_only: bool = False,
    ct2_only: bool = False
) -> dict:
    """
    使用 IntegratedASR 模型进行音频转写。
    
    参数:
    - model: IntegratedASR实例
    - waveform: 音频波形数据 (numpy/torch)
    - prep_timeout: 准备阶段超时时间
    
    返回:
    - dict: 包含转写结果的字典
    """
    try:
        # 标准化音频输入
        audio_np = normalize_audio_input(waveform)
        
        # 使用线程锁确保线程安全
        with model_lock:
            result = model.transcribe(
                audio=audio_np, 
                prep_timeout=prep_timeout,
                kimi_only=kimi_only,
                ct2_only=ct2_only
            )
        
        return {
            "status": 0,
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "engine": result.engine,
            "total_time": result.total_time
        }
        
    except Exception as e:
        logger.exception("IntegratedASR 推理失败")
        return {
            "status": -1,
            "error": str(e)
        }

def transcribe_from_path(
    model, 
    audio_path: str, 
    prep_timeout: int = 60,
    kimi_only: bool = False,
    ct2_only: bool = False
) -> dict:
    """
    从文件路径进行音频转写
    
    参数:
    - model: IntegratedASR实例
    - audio_path: 音频文件路径
    - prep_timeout: 准备阶段超时时间
   
    
    返回:
    - dict: 包含转写结果的字典
    """
    try:
        # 安全加载音频
        audio = load_audio_safe(audio_path)
        
        # 获取文件信息
        duration = librosa.get_duration(path=audio_path)
        file_name = os.path.basename(audio_path)
        
        # 转写
        with model_lock:
            result = model.transcribe(
                audio=audio,
                prep_timeout=prep_timeout,
                kimi_only=kimi_only,
                ct2_only=ct2_only
            )
        
        logger.info(f">>> Transcription Success: {audio_path}\n{result.text}")
        
        return {
            "status": 0,
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "engine": result.engine,
            "total_time": result.total_time,
            "file": file_name,
            "duration": round(duration, 2)
        }
        
    except Exception as e:
        logger.exception("IntegratedASR 路径转写失败")
        return {
            "status": -1,
            "error": str(e),
            "file": os.path.basename(audio_path) if audio_path else "unknown"
        }

def transcribe_from_bytes(
    model,
    audio_bytes: bytes,
    prep_timeout: int = 60,
    kimi_only: bool = False,
    ct2_only: bool = False
) -> dict:
    """
    从音频字节流进行转写
    
    参数:
    - model: IntegratedASR实例  
    - audio_bytes: 音频字节数据
    - prep_timeout: 准备阶段超时时间
    
    
    返回:
    - dict: 包含转写结果的字典
    """
    try:
        # 解码音频字节流
        waveform = load_waveform_from_bytes(audio_bytes)
        
        # 计算时长
        duration = librosa.get_duration(y=waveform, sr=16000)
        
        # 转写
        with model_lock:
            result = model.transcribe(
                audio=waveform,
                prep_timeout=prep_timeout,
                kimi_only=kimi_only,
                ct2_only=ct2_only
            )
        
        logger.info(f"[ASR] Success: {result.text}")
        
        return {
            "status": 0,
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "engine": result.engine,
            "total_time": result.total_time,
            "duration": round(duration, 2)
        }
        
    except Exception as e:
        logger.exception("[ASR] 字节流转写失败")
        return {
            "status": -1,
            "error": str(e)
        }

def transcribe_auto(
    model,
    audio: Union[str, Path, bytes, np.ndarray, torch.Tensor],
    prep_timeout: int = 60,
    kimi_only: bool = False,
    ct2_only: bool = False
) -> dict:
    """
    自动识别音频格式并调用对应的转写流程。
    支持输入类型: 路径、bytes、numpy、Tensor
    
    参数:
    - model: IntegratedASR实例
    - audio: 音频输入（多种格式）
    - prep_timeout: 准备阶段超时时间
   
    
    返回:
    - dict: 转写结果字典
    """
    
    # Case 1: 文件路径
    if isinstance(audio, (str, Path)):
        return transcribe_from_path(
            model=model,
            audio_path=str(audio),
            prep_timeout=prep_timeout,
            kimi_only=kimi_only,
            ct2_only=ct2_only
        )

    # Case 2: 字节流（如 HTTP API 中接收的）
    elif isinstance(audio, bytes):
        return transcribe_from_bytes(
            model=model,
            audio_bytes=audio,
            prep_timeout=prep_timeout,
            kimi_only=kimi_only,
            ct2_only=ct2_only
        )

    # Case 3: numpy 或 torch waveform
    elif isinstance(audio, (np.ndarray, torch.Tensor)):
        return transcribe_from_waveform(
            model=model,
            waveform=audio,
            prep_timeout=prep_timeout,
            kimi_only=kimi_only,
            ct2_only=ct2_only
        )

    else:
        raise TypeError(f"[transcribe_auto] Unsupported audio input type: {type(audio)}")

# 便捷函数：直接文本输出
def transcribe_text(
    model,
    audio: Union[str, Path, bytes, np.ndarray, torch.Tensor],
    prep_timeout: int = 60,
    kimi_only: bool = False,
    ct2_only: bool = False
) -> str:
    """
    转写音频并直接返回文本（简化版）
    
    返回:
    - str: 转写文本，失败时返回错误信息
    """
    result = transcribe_auto(model, audio, prep_timeout, kimi_only, ct2_only)
    if result["status"] == 0:
        return result["text"]
    else:
        return f"[转写失败] {result.get('error', 'Unknown error')}" 