# src/app/__init__.py

"""
IntegratedASR 应用层封装

提供简洁的API接口，方便FastAPI等部署使用，参考kimi_deployment的设计模式
"""

from .load_model import load_integrated_asr, get_model_instance, close_model
from .transcribe import (
    transcribe_auto, 
    transcribe_from_path, 
    transcribe_from_bytes, 
    transcribe_from_waveform,
    transcribe_text,
    load_audio_safe
)

__all__ = [
    'load_integrated_asr',
    'get_model_instance', 
    'close_model',
    'transcribe_auto',
    'transcribe_from_path',
    'transcribe_from_bytes', 
    'transcribe_from_waveform',
    'transcribe_text',
    'load_audio_safe'
] 