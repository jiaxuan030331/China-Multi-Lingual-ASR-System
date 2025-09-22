# src/app/__init__.py

"""
IntegratedASR application layer wrappers

Provide a simple API surface for deployment (e.g., FastAPI), inspired by kimi_deployment's structure
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