"""
Simple Integrated ASR: parallel preparation phase, single-chain decode after FNN routing

Core flow:
1. Launch CT2 chain (encode + LID) and Kimi chain (tokenize) in parallel
2. Wait FNN result: zh/en → continue Kimi, otherwise → continue CT2
3. Decode only one chain, stop the other
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import torchaudio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
from faster_whisper.tokenizer import Tokenizer as FWTokenizer

# Modified backend code from faster_whisper and  Kimi_Audio
from src.backends.faster_whisper_transcriber import TranscriptionOptions, WhisperModel
from src.backends.kimia_infer.api.kimia import KimiAudio
# Set Hugging Face cache under workspace directory
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# Ensure directories exist
os.makedirs('/workspace/.cache/huggingface/hub', exist_ok=True)
os.makedirs('/workspace/.cache/huggingface/transformers', exist_ok=True)
os.makedirs('/workspace/.cache/huggingface/datasets', exist_ok=True)

# Actual implementations imported
#


@dataclass
class PrepareResult:
    encoder_output: Optional[object] = None
    audio_features: Optional[np.ndarray] = None
    speech_tokens: Optional[torch.Tensor] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    text_tokens: Optional[torch.Tensor] = None
  


@dataclass
class TranscribeResult:
    """Transcription result"""
    text: str
    language: str
    confidence: float
    engine: str
    total_time: float




class CT2Chain:
    """CTranslate2 chain: encode + language identification"""
        
    def __init__(self, whisper_model_path: str = "large-v3", lid_model_path: str = None):
        """Initialize CT2 model"""
        
        self.whisper_model = WhisperModel(
                model_size_or_path=whisper_model_path,
                device="cuda",
                compute_type="bfloat16",
                language_classifier_path=lid_model_path
            )
        print(f"✅ CT2Chain initialized")
            

    
            
        self.stop_flag = threading.Event()
    
    def prepare(self, audio: np.ndarray, sr: int = 16000) -> PrepareResult:


        # 1) Normalize to 1D float32 NumPy (force mono)
        if torch is not None and isinstance(audio, torch.Tensor):
            if audio.ndim == 2 and audio.shape[0] > 1:
                audio = audio.mean(dim=0)          # [C,T] -> mono
            else:
                audio = audio.squeeze(0) if audio.ndim == 2 else audio
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)
            if audio_np.ndim == 2:                  # e.g. [C,T] or [T,C]
                audio_np = audio_np.mean(axis=0) if audio_np.shape[0] <= audio_np.shape[1] else audio_np.mean(axis=1)
        audio_np = audio_np.astype(np.float32, copy=False)

        # 2) Optional resample to 16kHz
        if sr != 16000:
            if torchaudio is not None and torch is not None:
                wf = torch.from_numpy(audio_np)
                wf = torchaudio.functional.resample(wf, sr, 16000)
                audio_np = wf.contiguous().cpu().numpy().astype(np.float32, copy=False)
            else:
                raise ValueError("Audio sample rate != 16000. Please resample to 16kHz before calling prepare().")

        # 3) CT2 encode + language detection
        audio_features = self.whisper_model.feature_extractor(audio_np)   # expects 1D float32
        encoder_output = self.whisper_model.encode(audio_features)
        
        language, confidence = self.whisper_model.custom_detect_language(encoder_output)
        
        return PrepareResult(
        encoder_output=encoder_output,
        audio_features=audio_features,
        language=language,
        confidence=confidence)
        
    def decode(self, encoder_output, language: str) -> str:
        """Decode stage"""
        if self.stop_flag.is_set():
            return ""
        try:    

            # Language normalization: <|yue|> -> yue
            lang_map = {"<|zh|>": "zh", "<|en|>": "en", "<|yue|>": "yue"}
            lang = lang_map.get(language, language)
            if lang not in ("zh", "en", "yue"):
                lang = None

            options = TranscriptionOptions(
                beam_size=1,
                best_of=1,
                patience=1.0,
                length_penalty=1.0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True,
                prompt_reset_on_temperature=0.5,
                temperatures=[0.0],
                initial_prompt=None,
                prefix=None,
                suppress_blank=True,
                suppress_tokens=None,
                without_timestamps=True,
                max_initial_timestamp=1.0,
                word_timestamps=False,
                prepend_punctuations="\"'\"¿([{-",
                append_punctuations="\"'.。,，!！?？:：\")]}、",
                max_new_tokens=None,
                clip_timestamps="0",
                hallucination_silence_threshold=None,
            )

            # Key: construct faster_whisper Tokenizer
            tokenizer = FWTokenizer(
                self.whisper_model.hf_tokenizer,
                self.whisper_model.model.is_multilingual,
                task="transcribe",
                language=lang,
            )

            # Use features pre-injected into transcribe
            features = getattr(self.whisper_model.feature_extractor, "audio")
            segments = list(self.whisper_model.generate_segments(
                features=features,
                tokenizer=tokenizer,
                options=options,
                encoder_output=encoder_output
            ))
            return " ".join([s.text for s in segments]).strip()
        except Exception as e:
            print(f"❌ CT2 decode failed: {e}")
            return ""
    
    def stop(self):
        """Stop signal"""
        self.stop_flag.set()


class KimiChain:
    """Kimi chain: GLM4 tokenizer + Kimi LLM"""
    
    def __init__(self, kimi_model_path_or_name: str ):#load glm4 tokenizer with kimi_audio class by default
        self.kimi_model_path = kimi_model_path_or_name
        self.kimi_engine = KimiAudio(
                    model_path_or_name=kimi_model_path_or_name,
                    device="cuda",
                    torch_dtype="bfloat16",
                    load_detokenizer=False  # save VRAM
                )
        
        
        
        self.stop_flag = threading.Event()
    
    def prepare(self, audio: np.ndarray,sr:int = 16000) -> PrepareResult:
        """Preparation: dual-encoder - Whisper continuous features + GLM4 discrete tokens"""
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        if isinstance(audio, torch.Tensor):
            if audio.ndim == 2 and audio.shape[0] > 1:
                audio = audio.mean(dim=0)
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.flatten()
        speech_tokens, text_tokens = self.kimi_engine.main_encoder(audio)    

            
        return PrepareResult(
            speech_tokens=speech_tokens,
            text_tokens=text_tokens,
        )
            
       
    
    def decode(self, speech_tokens, text_tokens, waveform):
        """Decode stage"""
        if self.stop_flag.is_set():
            return ""
        
        if self.kimi_engine is None:
            return "[Kimi engine not loaded]"
        '''
        try:
            # TODO: 实现真正的Kimi解码
            # result = self.kimi_engine.generate_from_speech_tokens(speech_tokens)
            return "[Kimi transcription - to be implemented]"
        except Exception as e:
            print(f"❌ Kimi decode failed: {e}")
            return ""
        '''
        text = self.kimi_engine.main_decoder(speech_tokens, text_tokens, waveform)
        return text

    def stop(self):
        """Stop signal"""
        self.stop_flag.set()


class IntegratedASR:
    """Integrated ASR: parallel preparation, FNN routing, single-chain decoding"""
    
    def __init__(
        self,
        ct2_model_path = 'large-v3', #default to large-v3 for compatibility with ctranslate2
        kimi_model_path_or_name: str = "moonshotai/Kimi-Audio-7B-Instruct",
        lid_model_path: str = "/workspace/ASR/WhisperLive/language_fnn_only2.pt",
        confidence_threshold: float = 0.7,
        load_kimi: bool = True,
        
    ):
        self.ct2_chain = CT2Chain(ct2_model_path,lid_model_path)
        if load_kimi:
            self.kimi_chain = KimiChain(kimi_model_path_or_name)
        else:
            self.kimi_chain = None
        self.confidence_threshold = confidence_threshold
        self.executor = ThreadPoolExecutor(max_workers=2) # Two workers for parallel prepare phases
        self.is_initialized = False
        
    

    
    def transcribe(self, audio: np.ndarray, sr:int = 16000, prep_timeout=60,kimi_only:bool = False,ct2_only:bool = False) -> TranscribeResult:
        """Main transcription entry point""" 
        language = None
        confidence = None
        use_kimi = 'undecided'
        if kimi_only and ct2_only:
            print(f'Error: Fixing both Kimi and CT2 is not allowed')
            return
   
        if self.kimi_chain is None:
            print(f'Kimi chain is not used, kimi model not loaded, use CT2 only')
            kimi_only = False
            use_kimi = False
            
        # Fault tolerance: init on-the-fly if not initialized; ensure ThreadPool max_workers=2
        if not hasattr(self, "executor") or getattr(self.executor, "_max_workers", 0) < 2:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.executor = ThreadPoolExecutor(max_workers=2)

        # ⏱️ Timing logs
        start_time = time.time()
        #print(f"🕐 [TIMING] Transcribe started at {start_time:.3f}")

        # 清停标志
        self.ct2_chain.stop_flag.clear()
        if self.kimi_chain and not ct2_only:
            self.kimi_chain.stop_flag.clear()
        


        # ⏱️ Parallel prepare phase start
        prep_start = time.time()
        
        
        start = time.time()
        f_ct2 = self.executor.submit(self.ct2_chain.prepare, audio,sr)
        if self.kimi_chain and not ct2_only:
            f_kimi = self.executor.submit(self.kimi_chain.prepare, audio,sr)
        
        submit_time = time.time()
        print(f"🕐 [TIMING] Both prepare tasks submitted in {(submit_time - prep_start)*1000:.1f}ms")

        try:
            ct2_wait_start = time.time()
            ct2_res = f_ct2.result(timeout=prep_timeout)
            ct2_wait_end = time.time()
            print(f"🕐 [TIMING] CT2 prepare completed in {(ct2_wait_end - ct2_wait_start)*1000:.1f}ms")
            
        except TimeoutError:
            timeout_time = time.time()
            print(f"❌ [TIMING] CT2 prepare timeout after {(timeout_time - ct2_wait_start)*1000:.1f}ms")
            self.ct2_chain.stop(); f_ct2.cancel()
            raise
         
        # ⏱️ Decision phase
        if not self.kimi_chain:
            use_kimi = False
        elif kimi_only or ct2_only:
            use_kimi = kimi_only
        if not kimi_only:
            decision_start = time.time()
            language, confidence = ct2_res.language, ct2_res.confidence
            if use_kimi == 'undecided':
                use_kimi = (
                    language in ("zh", "en", "<|zh|>", "<|en|>") 
                    and (confidence or 0.0) >= self.confidence_threshold           
                )
            decision_end = time.time()
            print(f"🕐 [TIMING] Language decision made in {(decision_end - decision_start)*1000:.1f}ms")
            print(f'use_kimi: {use_kimi}')


        
        
        

        if use_kimi:
            self.ct2_chain.stop()
            try:
                kimi_wait_start = time.time()
                kimi_res = f_kimi.result(timeout=prep_timeout)
                kimi_wait_end = time.time()
                print(f"🕐 [TIMING] Kimi prepare completed in {(kimi_wait_end - kimi_wait_start)*1000:.1f}ms")
                
                kimi_decode_start = time.time()
                text = self.kimi_chain.decode(kimi_res.speech_tokens, kimi_res.text_tokens,ct2_res.encoder_output)
                kimi_decode_end = time.time()
                print(f"🕐 [TIMING] Kimi decode completed in {(kimi_decode_end - kimi_decode_start)*1000:.1f}ms")
                engine = "kimi"
            except TimeoutError:
                kimi_timeout_time = time.time()
                print(f"❌ [TIMING] Kimi prepare timeout after {(kimi_timeout_time - kimi_wait_start)*1000:.1f}ms")
                self.kimi_chain.stop(); f_kimi.cancel()
                text = "[Kimi timeout, fallback to Whisper]"
                
                fallback_start = time.time()
                setattr(self.ct2_chain.whisper_model.feature_extractor, "audio", ct2_res.audio_features)
                text = self.ct2_chain.decode(ct2_res.encoder_output, language)
                fallback_end = time.time()
                print(f"🕐 [TIMING] Whisper fallback decode completed in {(fallback_end - fallback_start)*1000:.1f}ms")
                engine = "whisper"
        else:
            # 确保 decode 可用到特征（老实现从 feature_extractor.audio 取）
            whisper_decode_start = time.time()
            setattr(self.ct2_chain.whisper_model.feature_extractor, "audio", ct2_res.audio_features)
            text = self.ct2_chain.decode(ct2_res.encoder_output, language)
            whisper_decode_end = time.time()
            print(f"🕐 [TIMING] Whisper decode completed in {(whisper_decode_end - whisper_decode_start)*1000:.1f}ms")
            engine = "whisper"
        '''finally:
            # 如果有一条链未被消费，确保取消
            for fut in (f_ct2):
                if hasattr(fut, "cancel"):
                    try: fut.cancel()
                    except Exception: pass'''

        # ⏱️ Final timing summary
        total_time = time.time() - start
        end_time = time.time()
        print(f"🕐 [TIMING] Transcribe completed at {end_time:.3f}")
        print(f"🕐 [TIMING] Total transcribe time: {(end_time - start_time)*1000:.1f}ms")
        print(f"🕐 [TIMING] Used engine: {engine}")
        
        return TranscribeResult(
            text=text or "",
            language= language,
            confidence=confidence,
            engine=engine,
            total_time=total_time,
        )
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        torch.cuda.empty_cache()
        print("🔧 资源已清理")


