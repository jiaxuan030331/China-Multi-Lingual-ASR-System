import logging
import tempfile
import soundfile as sf
import numpy as np
import io
from pydub import AudioSegment
from src.app.load_model import model_lock  # Thread lock to ensure model thread-safety
from typing import Union
import os
import librosa
import torch 
from pathlib import Path

logger = logging.getLogger(__name__)

def load_audio_safe(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Safely load an audio file as 16kHz mono float32, supporting MP3/WAV and more
    
    Args:
    - audio_path: path to the audio file
    - sr: target sample rate, default 16000Hz
    
    Returns:
    - np.ndarray: mono float32 audio array
    """
    try:
        # Prefer librosa, which supports MP3 and more formats
        audio, sr = librosa.load(audio_path, sr=sr, mono=True)
        return audio, sr
    except Exception as e:
        logger.warning(f"[Audio] librosa failed, trying fallback: {e}")
        
        # Fallback: soundfile + pydub
        try:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            return load_waveform_from_bytes(audio_bytes, preferred_sr=sr)
        except Exception as e2:
            logger.error(f"[Audio] All loaders failed: {e2}")
            raise RuntimeError(f"Failed to load audio file: {audio_path}")

def load_waveform_from_bytes(audio_bytes: bytes, preferred_sr=16000) -> np.ndarray:
    """Convert uploaded audio bytes into a float32 waveform; supports multiple formats."""
    try:
        with io.BytesIO(audio_bytes) as f:
            waveform, sr = sf.read(f)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # to mono
        if sr != preferred_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=preferred_sr)
        logger.info("[Audio] soundfile read succeeded")
        return waveform.astype(np.float32)
    except Exception:
        logger.warning("[Audio] soundfile failed, trying pydub decode")

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
        logger.info("[Audio] pydub read succeeded")
        return waveform.astype(np.float32)
    except Exception as e:
        logger.error(f"[Audio] Decode failed: {e}")
        raise RuntimeError(f"Audio decode failed: {e}")

def normalize_audio_input(audio_input) -> np.ndarray:
    """
    Normalize audio input to np.ndarray
    
    Args:
    - audio_input: torch.Tensor, np.ndarray, list, etc.
    
    Returns:
    - np.ndarray: normalized audio array
    """
    if isinstance(audio_input, torch.Tensor):
        # Handle torch.Tensor
        if audio_input.ndim == 2 and audio_input.shape[0] > 1:
            # stereo to mono
            audio_input = audio_input.mean(dim=0)
        elif audio_input.ndim == 2:
            # mono with extra dimension
            audio_input = audio_input.squeeze(0)
        audio_np = audio_input.detach().cpu().numpy()
    elif isinstance(audio_input, np.ndarray):
        audio_np = audio_input.copy()
        if audio_np.ndim == 2:
            # stereo to mono
            audio_np = audio_np.mean(axis=0) if audio_np.shape[0] <= audio_np.shape[1] else audio_np.mean(axis=1)
    else:
        # other types to numpy
        audio_np = np.asarray(audio_input)
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
    
    # clean invalid values and cast to float32
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
    Transcribe audio with the IntegratedASR model.
    
    Args:
    - model: IntegratedASR instance
    - waveform: audio waveform (numpy/torch)
    - prep_timeout: preparation timeout
    
    Returns:
    - dict: transcription result dict
    """
    try:
        # Normalize audio input
        audio_np = normalize_audio_input(waveform)
        
        # Ensure thread safety with lock
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
        logger.exception("IntegratedASR inference failed")
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
    Transcribe audio from a file path
    
    Args:
    - model: IntegratedASR instance
    - audio_path: path to audio file
    - prep_timeout: preparation timeout
    
    Returns:
    - dict: transcription result dict
    """
    try:
        # Safely load audio
        audio = load_audio_safe(audio_path)
        
        # File info
        duration = librosa.get_duration(path=audio_path)
        file_name = os.path.basename(audio_path)
        
        # Transcribe
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
        logger.exception("IntegratedASR path transcription failed")
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
    Transcribe from audio bytes
    
    Args:
    - model: IntegratedASR instance  
    - audio_bytes: audio bytes
    - prep_timeout: preparation timeout
    
    Returns:
    - dict: transcription result dict
    """
    try:
        # Decode audio bytes
        waveform = load_waveform_from_bytes(audio_bytes)
        
        # Duration
        duration = librosa.get_duration(y=waveform, sr=16000)
        
        # Transcribe
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
        logger.exception("[ASR] Byte-stream transcription failed")
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
    Automatically detect audio input type and call the appropriate path.
    Supported types: path, bytes, numpy, torch.Tensor
    
    Args:
    - model: IntegratedASR instance
    - audio: audio input (various types)
    - prep_timeout: preparation timeout
    
    Returns:
    - dict: transcription result
    """
    
    # Case 1: file path
    if isinstance(audio, (str, Path)):
        return transcribe_from_path(
            model=model,
            audio_path=str(audio),
            prep_timeout=prep_timeout,
            kimi_only=kimi_only,
            ct2_only=ct2_only
        )

    # Case 2: bytes (e.g., received in HTTP API)
    elif isinstance(audio, bytes):
        return transcribe_from_bytes(
            model=model,
            audio_bytes=audio,
            prep_timeout=prep_timeout,
            kimi_only=kimi_only,
            ct2_only=ct2_only
        )

    # Case 3: numpy or torch waveform
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

# Convenience: return text directly

def transcribe_text(
    model,
    audio: Union[str, Path, bytes, np.ndarray, torch.Tensor],
    prep_timeout: int = 60,
    kimi_only: bool = False,
    ct2_only: bool = False
) -> str:
    """
    Transcribe audio and return text directly (simplified)
    
    Returns:
    - str: transcribed text, or error message on failure
    """
    result = transcribe_auto(model, audio, prep_timeout, kimi_only, ct2_only)
    if result["status"] == 0:
        return result["text"]
    else:
        return f"[Transcription failed] {result.get('error', 'Unknown error')}" 