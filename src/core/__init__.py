"""
Core components for the China Multi-Lingual ASR System.

This package contains the main orchestration and routing logic.
"""

from .integrated_asr import IntegratedASR
from .language_router import LanguageRouter
from .output_adapter import OutputAdapter

__all__ = [
    "IntegratedASR",
    "LanguageRouter",
    "OutputAdapter",
] 