"""
OutputAdapter: Unified output formatting for multi-lingual ASR system.

This module standardizes output formats from different ASR engines
to maintain compatibility with existing WebSocket protocols.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json


class OutputFormat(Enum):
    """Supported output formats."""
    WHISPERLIVE = "whisperlive"
    SEGMENTS = "segments"
    SRT = "srt"
    JSON = "json"


@dataclass
class WordTiming:
    """Word-level timing information."""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    """Standardized transcript segment."""
    id: int
    start: float
    end: float
    text: str
    language: Optional[str] = None
    confidence: float = 1.0
    no_speech_prob: float = 0.0
    engine: Optional[str] = None
    words: Optional[List[WordTiming]] = None
    
    def to_whisperlive_format(self) -> Dict[str, Any]:
        """Convert to WhisperLive compatible format."""
        segment_dict = {
            "start": f"{self.start:.3f}",
            "end": f"{self.end:.3f}", 
            "text": self.text
        }
        
        if self.no_speech_prob is not None:
            segment_dict["no_speech_prob"] = self.no_speech_prob
            
        if self.words:
            segment_dict["words"] = [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability
                }
                for word in self.words
            ]
        
        return segment_dict
    
    def to_srt_entry(self, index: int) -> str:
        """Convert to SRT format entry."""
        start_time = self._seconds_to_srt_time(self.start)
        end_time = self._seconds_to_srt_time(self.end)
        
        return f"{index}\n{start_time} --> {end_time}\n{self.text}\n"
    
    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


@dataclass
class TranscriptionOutput:
    """Complete transcription output with metadata."""
    segments: List[TranscriptSegment]
    language: str
    engine_used: str
    total_duration: float
    processing_time: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    
    def get_full_text(self, separator: str = " ") -> str:
        """Get complete transcribed text."""
        return separator.join(segment.text.strip() for segment in self.segments if segment.text.strip())
    
    def get_segment_count(self) -> int:
        """Get total number of segments."""
        return len(self.segments)
    
    def get_word_count(self) -> int:
        """Get total word count."""
        return sum(len(segment.text.split()) for segment in self.segments)


class OutputAdapter:
    """
    Adapter to convert between different ASR engine outputs and standardized formats.
    """
    
    def __init__(self):
        """Initialize the output adapter."""
        self.logger = logging.getLogger(__name__)
        
        # Format conversion statistics
        self.stats = {
            "conversions_total": 0,
            "kimi_conversions": 0,
            "whisper_conversions": 0,
            "format_distribution": {},
            "average_segments_per_output": 0.0
        }
    
    def from_kimi_result(self, kimi_result, audio_duration: float = 0.0) -> TranscriptionOutput:
        """
        Convert Kimi engine result to standardized format.
        
        Args:
            kimi_result: Result from Kimi engine
            audio_duration: Total audio duration in seconds
            
        Returns:
            TranscriptionOutput in standardized format
        """
        try:
            # TODO: Implement actual Kimi result parsing
            # This assumes kimi_result has segments and metadata
            
            segments = []
            
            # Placeholder implementation
            if hasattr(kimi_result, 'segments'):
                for i, segment in enumerate(kimi_result.segments):
                    transcript_segment = TranscriptSegment(
                        id=i + 1,
                        start=getattr(segment, 'start', 0.0),
                        end=getattr(segment, 'end', audio_duration),
                        text=getattr(segment, 'text', ''),
                        language=getattr(segment, 'language', 'zh'),
                        confidence=getattr(segment, 'confidence', 0.95),
                        engine="kimi"
                    )
                    segments.append(transcript_segment)
            else:
                # Single segment result
                segment = TranscriptSegment(
                    id=1,
                    start=0.0,
                    end=audio_duration,
                    text=str(kimi_result) if isinstance(kimi_result, str) else "Kimi transcription",
                    language="zh",
                    confidence=0.95,
                    engine="kimi"
                )
                segments.append(segment)
            
            output = TranscriptionOutput(
                segments=segments,
                language="zh",
                engine_used="kimi",
                total_duration=audio_duration,
                processing_time=getattr(kimi_result, 'processing_time', 0.1),
                confidence=0.95,
                metadata={"source": "kimi", "model_version": "kimi-audio-v1"}
            )
            
            self.stats["kimi_conversions"] += 1
            return output
            
        except Exception as e:
            self.logger.error(f"Failed to convert Kimi result: {e}")
            raise
    
    def from_whisper_result(self, whisper_segments, language: str = "auto", 
                          processing_time: float = 0.0, audio_duration: float = 0.0) -> TranscriptionOutput:
        """
        Convert Whisper engine result to standardized format.
        
        Args:
            whisper_segments: Segments from Whisper engine
            language: Detected/specified language
            processing_time: Processing time in seconds
            audio_duration: Total audio duration
            
        Returns:
            TranscriptionOutput in standardized format
        """
        try:
            segments = []
            total_confidence = 0.0
            
            for i, segment in enumerate(whisper_segments):
                # Convert words if available
                words = None
                if hasattr(segment, 'words') and segment.words:
                    words = [
                        WordTiming(
                            word=word.word,
                            start=word.start,
                            end=word.end,
                            probability=getattr(word, 'probability', 1.0)
                        )
                        for word in segment.words
                    ]
                
                # Calculate segment confidence
                segment_confidence = 1.0 - getattr(segment, 'no_speech_prob', 0.0)
                total_confidence += segment_confidence
                
                transcript_segment = TranscriptSegment(
                    id=i + 1,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    language=language,
                    confidence=segment_confidence,
                    no_speech_prob=getattr(segment, 'no_speech_prob', 0.0),
                    engine="whisper",
                    words=words
                )
                segments.append(transcript_segment)
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(segments) if segments else 0.0
            
            output = TranscriptionOutput(
                segments=segments,
                language=language,
                engine_used="whisper",
                total_duration=audio_duration,
                processing_time=processing_time,
                confidence=avg_confidence,
                metadata={"source": "whisper", "model_version": "whisper-large-v3"}
            )
            
            self.stats["whisper_conversions"] += 1
            return output
            
        except Exception as e:
            self.logger.error(f"Failed to convert Whisper result: {e}")
            raise
    
    def to_whisperlive_format(self, output: TranscriptionOutput, 
                            include_metadata: bool = False) -> Dict[str, Any]:
        """
        Convert to WhisperLive compatible format.
        
        Args:
            output: Standardized transcription output
            include_metadata: Whether to include metadata
            
        Returns:
            Dictionary in WhisperLive format
        """
        try:
            segments = [segment.to_whisperlive_format() for segment in output.segments]
            
            result = {
                "segments": segments,
                "language": output.language,
                "language_probability": output.confidence
            }
            
            if include_metadata:
                result["metadata"] = {
                    "engine_used": output.engine_used,
                    "processing_time": output.processing_time,
                    "total_duration": output.total_duration,
                    "segment_count": len(segments),
                    "word_count": output.get_word_count(),
                    **(output.metadata or {})
                }
            
            self._update_format_stats("whisperlive", len(segments))
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to convert to WhisperLive format: {e}")
            raise
    
    def to_srt_format(self, output: TranscriptionOutput) -> str:
        """
        Convert to SRT subtitle format.
        
        Args:
            output: Standardized transcription output
            
        Returns:
            SRT formatted string
        """
        try:
            srt_entries = []
            
            for i, segment in enumerate(output.segments, 1):
                srt_entry = segment.to_srt_entry(i)
                srt_entries.append(srt_entry)
            
            self._update_format_stats("srt", len(output.segments))
            return "\n".join(srt_entries)
            
        except Exception as e:
            self.logger.error(f"Failed to convert to SRT format: {e}")
            raise
    
    def to_json_format(self, output: TranscriptionOutput, 
                      include_words: bool = False) -> str:
        """
        Convert to JSON format.
        
        Args:
            output: Standardized transcription output
            include_words: Whether to include word-level timings
            
        Returns:
            JSON formatted string
        """
        try:
            # Convert to dictionary
            output_dict = asdict(output)
            
            # Remove word timings if not requested
            if not include_words:
                for segment in output_dict["segments"]:
                    segment.pop("words", None)
            
            # Add summary statistics
            output_dict["summary"] = {
                "total_segments": len(output.segments),
                "total_words": output.get_word_count(),
                "full_text": output.get_full_text()
            }
            
            self._update_format_stats("json", len(output.segments))
            return json.dumps(output_dict, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to convert to JSON format: {e}")
            raise
    
    def convert_format(self, output: TranscriptionOutput, 
                      target_format: OutputFormat, **kwargs) -> Union[Dict, str]:
        """
        Convert output to specified format.
        
        Args:
            output: Standardized transcription output
            target_format: Target output format
            **kwargs: Additional format-specific parameters
            
        Returns:
            Converted output in target format
        """
        if target_format == OutputFormat.WHISPERLIVE:
            return self.to_whisperlive_format(output, **kwargs)
        elif target_format == OutputFormat.SRT:
            return self.to_srt_format(output)
        elif target_format == OutputFormat.JSON:
            return self.to_json_format(output, **kwargs)
        elif target_format == OutputFormat.SEGMENTS:
            return output.segments
        else:
            raise ValueError(f"Unsupported output format: {target_format}")
    
    def _update_format_stats(self, format_name: str, segment_count: int):
        """Update format conversion statistics."""
        self.stats["conversions_total"] += 1
        
        if format_name not in self.stats["format_distribution"]:
            self.stats["format_distribution"][format_name] = 0
        self.stats["format_distribution"][format_name] += 1
        
        # Update average segments per output
        total_segments = (self.stats["average_segments_per_output"] * 
                         (self.stats["conversions_total"] - 1) + segment_count)
        self.stats["average_segments_per_output"] = total_segments / self.stats["conversions_total"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset conversion statistics."""
        self.stats = {
            "conversions_total": 0,
            "kimi_conversions": 0,
            "whisper_conversions": 0,
            "format_distribution": {},
            "average_segments_per_output": 0.0
        } 