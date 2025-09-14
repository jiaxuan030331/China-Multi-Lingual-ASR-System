"""
Simple Integrated ASR: 并行准备阶段，FNN决策后单链解码

核心逻辑：
1. 并行启动 CT2链(encode+LID) 和 Kimi链(tokenize)
2. 等FNN出结果：zh/en→继续Kimi，其他→继续CT2
3. 单链解码，另一链停止
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

# 设置HuggingFace缓存到workspace目录
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# 确保目录存在
os.makedirs('/workspace/.cache/huggingface/hub', exist_ok=True)
os.makedirs('/workspace/.cache/huggingface/transformers', exist_ok=True)
os.makedirs('/workspace/.cache/huggingface/datasets', exist_ok=True)

# 导入实际实现
from kimi_deployment.kimia_infer.api.kimia import KimiAudio
from WhisperLive.whisper_live.new_transcriber import WhisperModel
from kimi_deployment.kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
from kimi_deployment.app.load_model import load_kimi_model

@dataclass
class PrepareResult:
    """准备阶段结果"""
    encoder_output: Optional[object] = None
    speech_tokens: Optional[torch.Tensor] = None
    language: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class TranscribeResult:
    """转写结果"""
    text: str
    language: str
    confidence: float
    engine: str
    total_time: float




class CT2Chain:
    """CTranslate2链：编码+语言检测"""
        
    def __init__(self, whisper_model_path: str = "large-v3", lid_model_path: str = None):
        """初始化CT2模型"""
        
        self.whisper_model = WhisperModel(
                model_size_or_path=whisper_model_path,
                device="cuda",
                compute_type="bfloat16",
                language_classifier_path=lid_model_path
            )
        print(f"✅ CT2Chain initialized")
            
       
            
        self.stop_flag = threading.Event()
    
    def prepare(self, audio: np.ndarray) -> PrepareResult:
        """准备阶段：编码+语言检测"""
        
        # CT2编码
        audio_features = self.whisper_model.feature_extractor(audio)
        encoder_output = self.whisper_model.encode(audio_features)
        
        # 语言检测
        language, confidence = self.whisper_model.custom_detect_language(encoder_output)
        
        return PrepareResult(
            encoder_output=encoder_output,
            language=language,
            confidence=confidence
        )
        
    def decode(self, encoder_output, language: str) -> str:
        """解码阶段"""
        if self.stop_flag.is_set():
            return ""
        
        try:
            from WhisperLive.whisper_live.new_transcriber import TranscriptionOptions
            
            options = TranscriptionOptions(
                beam_size=1,  # 贪心解码，最快
                best_of=1,
                temperature=[0.0],
                language=language if language in ["zh", "en", "yue"] else None,
                without_timestamps=True
            )
            
            tokenizer = self.whisper_model.hf_tokenizer
            segments = list(self.whisper_model.generate_segments(
                features=self.whisper_model.feature_extractor.audio,
                tokenizer=tokenizer,
                options=options,
                encoder_output=encoder_output
            ))
            
            text = " ".join([seg.text for seg in segments]).strip()
            return text
        except Exception as e:
            print(f"❌ CT2 decode failed: {e}")
            return ""
    
    def stop(self):
        """停止信号"""
        self.stop_flag.set()


class KimiChain:
    """Kimi链：GLM4 tokenizer + Kimi LLM"""
    
    def __init__(self, kimi_model_path_or_name: str ):#load glm4 tokenizer with kimi_audio class by default
        self.kimi_engine = None
        self.kimi_model_path = kimi_model_path_or_name
        self.kimi_engine = None
        self.kimi_engine = KimiAudio(
                    model_path_or_name=kimi_model_path_or_name,
                    device="cuda",
                    torch_dtype="bfloat16",
                    load_detokenizer=False  # 节省显存
                )
        print(f"✅ Kimi engine loaded")
            
        
        self.stop_flag = threading.Event()
    
    def prepare(self, audio: np.ndarray) -> PrepareResult:
        """准备阶段：GLM4 tokenize"""
        try:
            speech_tokens = self.glm4_tokenizer.tokenize(speech=audio, sr=16000)
            return PrepareResult(speech_tokens=speech_tokens)
        except Exception as e:
            print(f"❌ Kimi prepare failed: {e}")
            return PrepareResult()
    
    def decode(self, speech_tokens: torch.Tensor) -> str:
        """解码阶段"""
        if self.stop_flag.is_set():
            return ""
        
        if self.kimi_engine is None:
            return "[Kimi engine not loaded]"
        
        try:
            # TODO: 实现真正的Kimi解码
            # result = self.kimi_engine.generate_from_speech_tokens(speech_tokens)
            return "[Kimi transcription - to be implemented]"
        except Exception as e:
            print(f"❌ Kimi decode failed: {e}")
            return ""
    
    def stop(self):
        """停止信号"""
        self.stop_flag.set()


class IntegratedASR:
    """集成ASR系统：并行准备，FNN决策，单链解码"""
    
    def __init__(
        self,
        ct2_model_path = 'large-v3', #default to large-v3 for compatibility with ctranslate2
        kimi_model_path_or_name: str = "moonshotai/Kimi-Audio-7B-Instruct",
        lid_model_path: str = "/workspace/ASR/WhisperLive/language_fnn_only2.pt",
        confidence_threshold: float = 0.9
    ):
        self.kimi_chain = KimiChain(kimi_model_path_or_name)
        self.ct2_chain = CT2Chain(ct2_model_path,lid_model_path)
        
        self.confidence_threshold = confidence_threshold
        self.executor = ThreadPoolExecutor(max_workers=1) #only one worker for 40-50G GPU memory
        self.is_initialized = False
    

    
    def transcribe(self, audio: np.ndarray) -> TranscribeResult:
        """主转写接口"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        
        # 重置停止标志
        self.ct2_chain.stop_flag.clear()
        self.kimi_chain.stop_flag.clear()
        
        try:
            print("🔄 并行启动准备阶段...")
            
            # 并行提交任务
            future_ct2 = self.executor.submit(self.ct2_chain.prepare, audio, self.lid_fnn)
            future_kimi = self.executor.submit(self.kimi_chain.prepare, audio)
            
            # 等待CT2完成（包含LID）
            ct2_result = future_ct2.result()
            language = ct2_result.language
            confidence = ct2_result.confidence
            
            print(f"🔍 语言检测: {language} (置信度: {confidence:.3f})")
            
            # 路由决策
            if language in ("zh", "en") and confidence >= self.confidence_threshold and self.kimi_chain.glm4_tokenizer:
                # 选择Kimi链
                print("🎯 路由到Kimi引擎")
                self.ct2_chain.stop()
                
                kimi_result = future_kimi.result()
                if kimi_result.speech_tokens is not None:
                    text = self.kimi_chain.decode(kimi_result.speech_tokens)
                else:
                    text = "[Kimi tokenization failed]"
                engine = "kimi"
                
            else:
                # 选择CT2链
                print("🎯 路由到Whisper引擎")
                self.kimi_chain.stop()
                
                text = self.ct2_chain.decode(ct2_result.encoder_output, language)
                engine = "whisper"
            
            total_time = time.time() - start_time
            
            result = TranscribeResult(
                text=text,
                language=language,
                confidence=confidence,
                engine=engine,
                total_time=total_time
            )
            
            print(f"✅ 转写完成: '{text}' ({engine}, {total_time:.2f}s)")
            return result
            
        except Exception as e:
            print(f"❌ 转写失败: {e}")
            return TranscribeResult(
                text="",
                language="unknown",
                confidence=0.0,
                engine="error",
                total_time=time.time() - start_time
            )
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        torch.cuda.empty_cache()
        print("🔧 资源已清理")


