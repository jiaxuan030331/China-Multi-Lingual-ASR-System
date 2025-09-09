"""
LanguageRouter: Intelligent language detection and routing system.

This module handles language identification and routing decisions
for the multi-lingual ASR system.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class DetectionMethod(Enum):
    """Language detection methods."""
    WHISPER_ENCODER = "whisper_encoder"
    CUSTOM_CLASSIFIER = "custom_classifier" 
    HYBRID = "hybrid"


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str
    confidence: float
    method_used: str
    all_probabilities: Optional[Dict[str, float]] = None
    processing_time: float = 0.0


@dataclass
class RouterConfig:
    """Configuration for language router."""
    confidence_threshold: float = 0.7
    supported_languages: List[str] = None
    kimi_languages: List[str] = None
    fallback_engine: str = "whisper"
    detection_method: DetectionMethod = DetectionMethod.WHISPER_ENCODER
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["zh", "en", "yue", "zh-tw"]
        if self.kimi_languages is None:
            self.kimi_languages = ["zh", "en"]


class LanguageDetector:
    """Base language detection interface."""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect(self, audio_features) -> LanguageDetectionResult:
        """Detect language from audio features."""
        raise NotImplementedError


class WhisperLanguageDetector(LanguageDetector):
    """Language detection using Whisper encoder."""
    
    def __init__(self, config: RouterConfig, whisper_model=None):
        super().__init__(config)
        self.whisper_model = whisper_model
        self.is_initialized = False
    
    def initialize(self, whisper_model) -> bool:
        """Initialize with Whisper model."""
        try:
            self.whisper_model = whisper_model
            self.is_initialized = True
            self.logger.info("Whisper language detector initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper detector: {e}")
            return False
    
    def detect(self, audio_features) -> LanguageDetectionResult:
        """Detect language using Whisper encoder."""
        if not self.is_initialized:
            raise RuntimeError("Whisper detector not initialized")
        
        import time
        start_time = time.time()
        
        try:
            # TODO: Implement actual Whisper language detection
            # encoder_output = self.whisper_model.encode(audio_features)
            # results = self.whisper_model.detect_language(encoder_output)
            
            # Placeholder implementation
            detected_lang = "zh"
            confidence = 0.9
            all_probs = {"zh": 0.9, "en": 0.08, "yue": 0.02}
            
            processing_time = time.time() - start_time
            
            return LanguageDetectionResult(
                language=detected_lang,
                confidence=confidence,
                method_used="whisper_encoder",
                all_probabilities=all_probs,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            # Return fallback result
            return LanguageDetectionResult(
                language="auto",
                confidence=0.0,
                method_used="fallback",
                processing_time=time.time() - start_time
            )


class CustomLanguageDetector(LanguageDetector):
    """Custom trained language detector."""
    
    def __init__(self, config: RouterConfig, model_path: Optional[str] = None):
        super().__init__(config)
        self.model_path = model_path
        self.model = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize custom language detection model."""
        try:
            if self.model_path:
                # TODO: Load custom language detection model
                # self.model = load_custom_lid_model(self.model_path)
                pass
            
            self.is_initialized = True
            self.logger.info("Custom language detector initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize custom detector: {e}")
            return False
    
    def detect(self, audio_features) -> LanguageDetectionResult:
        """Detect language using custom model."""
        if not self.is_initialized:
            raise RuntimeError("Custom detector not initialized")
        
        import time
        start_time = time.time()
        
        try:
            # TODO: Implement custom language detection
            # results = self.model.predict(audio_features)
            
            # Placeholder implementation
            detected_lang = "yue"  # Cantonese detection example
            confidence = 0.85
            all_probs = {"yue": 0.85, "zh": 0.12, "en": 0.03}
            
            processing_time = time.time() - start_time
            
            return LanguageDetectionResult(
                language=detected_lang,
                confidence=confidence,
                method_used="custom_classifier",
                all_probabilities=all_probs,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Custom language detection failed: {e}")
            return LanguageDetectionResult(
                language="auto",
                confidence=0.0,
                method_used="fallback",
                processing_time=time.time() - start_time
            )


class LanguageRouter:
    """
    Main language routing system that combines detection and routing logic.
    """
    
    def __init__(self, config: RouterConfig):
        """Initialize the language router."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors based on configuration
        self.detectors = {}
        self.primary_detector = None
        
        # Routing statistics
        self.stats = {
            "total_detections": 0,
            "kimi_routes": 0,
            "whisper_routes": 0,
            "detection_times": [],
            "confidence_distribution": {}
        }
        
        self.is_initialized = False
    
    def initialize(self, whisper_model=None, custom_model_path: Optional[str] = None) -> bool:
        """Initialize all language detectors."""
        try:
            # Initialize Whisper detector
            if self.config.detection_method in [DetectionMethod.WHISPER_ENCODER, DetectionMethod.HYBRID]:
                whisper_detector = WhisperLanguageDetector(self.config, whisper_model)
                if whisper_detector.initialize(whisper_model):
                    self.detectors["whisper"] = whisper_detector
                    self.primary_detector = whisper_detector
            
            # Initialize custom detector
            if self.config.detection_method in [DetectionMethod.CUSTOM_CLASSIFIER, DetectionMethod.HYBRID]:
                custom_detector = CustomLanguageDetector(self.config, custom_model_path)
                if custom_detector.initialize():
                    self.detectors["custom"] = custom_detector
                    if self.primary_detector is None:
                        self.primary_detector = custom_detector
            
            if not self.detectors:
                self.logger.error("No language detectors initialized")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Language router initialized with {len(self.detectors)} detectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize language router: {e}")
            return False
    
    def detect_language(self, audio_features, use_hybrid: bool = False) -> LanguageDetectionResult:
        """
        Detect language from audio features.
        
        Args:
            audio_features: Audio features for detection
            use_hybrid: Whether to use hybrid detection (combine multiple methods)
            
        Returns:
            LanguageDetectionResult with detection details
        """
        if not self.is_initialized:
            raise RuntimeError("Language router not initialized")
        
        try:
            if use_hybrid and len(self.detectors) > 1:
                return self._hybrid_detection(audio_features)
            else:
                return self.primary_detector.detect(audio_features)
                
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            # Return fallback result
            return LanguageDetectionResult(
                language="auto",
                confidence=0.0,
                method_used="fallback"
            )
    
    def _hybrid_detection(self, audio_features) -> LanguageDetectionResult:
        """Combine results from multiple detectors."""
        results = []
        
        # Get results from all available detectors
        for name, detector in self.detectors.items():
            try:
                result = detector.detect(audio_features)
                results.append((name, result))
            except Exception as e:
                self.logger.warning(f"Detector {name} failed: {e}")
        
        if not results:
            return LanguageDetectionResult(
                language="auto",
                confidence=0.0,
                method_used="fallback"
            )
        
        # Simple voting/averaging strategy
        # TODO: Implement more sophisticated fusion
        best_result = max(results, key=lambda x: x[1].confidence)
        
        return LanguageDetectionResult(
            language=best_result[1].language,
            confidence=best_result[1].confidence,
            method_used="hybrid",
            all_probabilities=best_result[1].all_probabilities,
            processing_time=sum(r[1].processing_time for r in results) / len(results)
        )
    
    def route_to_engine(self, detection_result: LanguageDetectionResult) -> str:
        """
        Determine which ASR engine to use based on detection result.
        
        Args:
            detection_result: Language detection result
            
        Returns:
            Engine name ("kimi" or "whisper")
        """
        language = detection_result.language
        confidence = detection_result.confidence
        
        # Update statistics
        self.stats["total_detections"] += 1
        self.stats["detection_times"].append(detection_result.processing_time)
        
        # Route to Kimi for high-confidence supported languages
        if (language in self.config.kimi_languages and 
            confidence >= self.config.confidence_threshold):
            self.stats["kimi_routes"] += 1
            self.logger.debug(f"Routing to Kimi: {language} (conf: {confidence:.3f})")
            return "kimi"
        
        # Route to Whisper for other cases
        self.stats["whisper_routes"] += 1
        self.logger.debug(f"Routing to Whisper: {language} (conf: {confidence:.3f})")
        return "whisper"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats["detection_times"]:
            stats["average_detection_time"] = sum(stats["detection_times"]) / len(stats["detection_times"])
            stats["max_detection_time"] = max(stats["detection_times"])
            stats["min_detection_time"] = min(stats["detection_times"])
        
        if stats["total_detections"] > 0:
            stats["kimi_route_percentage"] = (stats["kimi_routes"] / stats["total_detections"]) * 100
            stats["whisper_route_percentage"] = (stats["whisper_routes"] / stats["total_detections"]) * 100
        
        return stats
    
    def reset_statistics(self):
        """Reset routing statistics."""
        self.stats = {
            "total_detections": 0,
            "kimi_routes": 0,
            "whisper_routes": 0,
            "detection_times": [],
            "confidence_distribution": {}
        } 