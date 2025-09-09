"""
China Multi-Lingual ASR System

A comprehensive end-to-end multi-lingual ASR system designed for the Chinese market.
"""

__version__ = "0.1.0"
__author__ = "Jiaxuan Huang"
__description__ = "Multi-lingual ASR system with intelligent language routing"

from .core.integrated_asr import IntegratedASR
from .core.language_router import LanguageRouter
from .core.output_adapter import OutputAdapter

__all__ = [
    "IntegratedASR",
    "LanguageRouter", 
    "OutputAdapter",
] 