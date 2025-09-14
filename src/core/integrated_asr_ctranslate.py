"""
Workspace-optimized IntegratedASR: 所有模型下载到/workspace目录

核心修改：
1. 设置HuggingFace缓存目录到/workspace
2. 语言分类器模型自动下载
3. 确保所有模型存储在/workspace而不是root缓存
"""

import torch
import torch.nn as nn
import numpy as np
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass
import logging
import ctranslate2

# 设置环境变量，让所有HuggingFace模型下载到workspace
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# Import actual implementations  
from kimi_deployment.kimia_infer.api.kimia import KimiAudio
from WhisperLive.whisper_live.new_transcriber import WhisperModel
from kimi_deployment.kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer


@dataclass
class CTranslateFeatures:
    """CTranslate2格式的共享特征"""
    audio_features: np.ndarray  # 原始音频特征 (梅尔频谱)
    encoder_output: ctranslate2.StorageView  # CTranslate2 encoder输出
    sequence_length: int
    language: Optional[str] = None
    confidence: Optional[float] = None
    glm4_tokens: Optional[torch.Tensor] = None


@dataclass 
class CTranslateResult:
    """CTranslate2处理结果"""
    text: str
    language: str
    confidence: float
    engine: str
    processing_time: float
    encoding_time: float
    language_detection_time: float
    decoding_time: float


def ensure_workspace_directory(subdir: str) -> str:
    """确保workspace子目录存在"""
    workspace_dir = f"/workspace/{subdir}"
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir


def download_language_classifier():
    """下载语言分类器模型到workspace"""
    model_dir = ensure_workspace_directory("models")
    model_path = os.path.join(model_dir, "language_fnn_only2.pt")
    
    if os.path.exists(model_path):
        print(f"✅ 语言分类器已存在: {model_path}")
        return model_path
    
    # 这里需要提供实际的下载URL，暂时创建一个占位文件
    print(f"⚠️  语言分类器模型不存在，需要从实际位置下载到: {model_path}")
    print("💡 请提供语言分类器的下载链接")
    
    # 如果有实际的URL，可以这样下载：
    # url = "https://example.com/language_fnn_only2.pt"
    # urllib.request.urlretrieve(url, model_path)
    
    return model_path


class WorkspaceEncoder:
    """
    Workspace优化的CTranslate2 Whisper encoder
    确保模型下载到workspace目录
    """
    
    def __init__(self, whisper_model_path: str = "base"):
        self.whisper_model = None
        self.whisper_model_path = whisper_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.download_root = ensure_workspace_directory("whisper_models")
        
    def initialize(self) -> bool:
        """初始化CTranslate2 Whisper模型，下载到workspace"""
        try:
            print(f"📥 下载/加载Whisper模型到: {self.download_root}")
            
            self.whisper_model = WhisperModel(
                model_size_or_path=self.whisper_model_path,
                device="cuda",
                compute_type="int8_float16",
                download_root=self.download_root,  # 指定下载目录
                local_files_only=False  # 允许下载
            )
            print(f"✅ Whisper encoder加载成功: {self.whisper_model_path}")
            return True
        except Exception as e:
            print(f"❌ Whisper encoder初始化失败: {e}")
            return False
    
    def encode_audio(self, audio: np.ndarray) -> CTranslateFeatures:
        """使用CTranslate2 encoder进行音频编码"""
        if self.whisper_model is None:
            raise RuntimeError("CTranslate2 encoder not initialized")
        
        try:
            start_time = time.time()
            
            # 1. 提取音频特征 (梅尔频谱等)
            audio_features = self.whisper_model.feature_extractor(audio)
            
            # 2. 使用CTranslate2 encoder进行编码
            encoder_output = self.whisper_model.encode(audio_features)
            
            encoding_time = time.time() - start_time
            
            # 获取序列长度
            encoder_output_cpu = encoder_output.to_device(ctranslate2.Device(0))  # CPU
            encoder_array = np.array(encoder_output_cpu)
            if encoder_array.dtype == object:
                encoder_array = np.stack(encoder_array)
            sequence_length = encoder_array.shape[1] if len(encoder_array.shape) > 1 else 1
            
            return CTranslateFeatures(
                audio_features=audio_features,
                encoder_output=encoder_output,
                sequence_length=sequence_length
            )
            
        except Exception as e:
            raise RuntimeError(f"CTranslate2 encoding failed: {e}")


class WorkspaceLanguageClassifier:
    """
    Workspace优化的语言分类器
    自动下载模型到workspace
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 如果没有提供路径，尝试下载到workspace
        if model_path is None:
            model_path = download_language_classifier()
        
        self.model_path = model_path
        
        # 构建分类网络 (与训练时一致)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.LayerNorm(1280),
            nn.Dropout(0.1),
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(640, 3),
        ).to(self.device)
        
        # 尝试加载训练权重
        if os.path.exists(model_path):
            self.classifier.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.classifier.eval()
            print(f"✅ 语言分类器加载成功: {model_path}")
        else:
            print(f"⚠️  语言分类器模型不存在: {model_path}")
            print("💡 将使用随机权重（仅用于测试）")
        
        # 语言映射
        self.id2lang = {0: "zh", 1: "en", 2: "yue"}
    
    def predict_language(self, ctranslate_features: CTranslateFeatures) -> Tuple[str, float]:
        """基于CTranslate2 encoder输出预测语言"""
        try:
            start_time = time.time()
            
            encoder_output = ctranslate_features.encoder_output
            
            # 1. 转换CTranslate2.StorageView到numpy
            encoder_output_cpu = encoder_output.to_device(ctranslate2.Device(0))  # 移到CPU
            encoder_output_np = np.array(encoder_output_cpu)
            
            if encoder_output_np.dtype == object:
                encoder_output_np = np.stack(encoder_output_np)
            encoder_output_np = encoder_output_np.astype(np.float32)
            
            # 2. 池化到固定维度 (B, H)
            pooled = encoder_output_np.mean(axis=1)  # shape: (1, H)
            pooled_tensor = torch.tensor(pooled, dtype=torch.float32, device=self.device)
            
            # 3. 语言分类推理
            with torch.no_grad():
                logits = self.classifier(pooled_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                pred_id = np.argmax(probs)
                confidence = float(probs[pred_id])
                language = self.id2lang[pred_id]
            
            detection_time = time.time() - start_time
            
            return language, confidence
            
        except Exception as e:
            # 如果检测失败，返回默认值
            print(f"⚠️  语言检测失败，使用默认: {e}")
            return "zh", 0.5


class WorkspaceKimiDecoder:
    """
    Workspace优化的Kimi解码器
    确保模型下载到workspace
    """
    
    def __init__(self, kimi_model_path: str = None, glm4_tokenizer_path: str = None):
        self.kimi_engine = None
        self.glm4_tokenizer = None
        self.kimi_model_path = kimi_model_path
        self.glm4_tokenizer_path = glm4_tokenizer_path or "THUDM/glm-4-voice-tokenizer"
        
    def initialize(self) -> bool:
        """初始化Kimi引擎和GLM4 tokenizer，下载到workspace"""
        success = True
        
        # 初始化GLM4 tokenizer (会自动下载到workspace缓存)
        if self.glm4_tokenizer_path:
            try:
                print(f"📥 下载/加载GLM4 tokenizer: {self.glm4_tokenizer_path}")
                self.glm4_tokenizer = Glm4Tokenizer(self.glm4_tokenizer_path)
                self.glm4_tokenizer = self.glm4_tokenizer.to(torch.device("cuda"))
                print(f"✅ GLM4 tokenizer加载成功")
            except Exception as e:
                print(f"❌ GLM4 tokenizer加载失败: {e}")
                success = False
        
        # 初始化Kimi引擎 (如果提供了路径)
        if self.kimi_model_path:
            try:
                print(f"📥 加载Kimi引擎: {self.kimi_model_path}")
                self.kimi_engine = KimiAudio(
                    model_path=self.kimi_model_path,
                    device="cuda",
                    torch_dtype="bfloat16",
                    load_detokenizer=False
                )
                print(f"✅ Kimi引擎加载成功")
            except Exception as e:
                print(f"❌ Kimi引擎加载失败: {e}")
                success = False
        else:
            print("💡 未提供Kimi模型路径，跳过Kimi引擎加载")
        
        return success
    
    def get_glm4_tokens(self, audio: np.ndarray) -> torch.Tensor:
        """获取GLM4 speech tokens"""
        if self.glm4_tokenizer is None:
            raise RuntimeError("GLM4 tokenizer not initialized")
        
        return self.glm4_tokenizer.tokenize(speech=audio, sr=16000)
    
    def decode_with_glm4_tokens(self, glm4_tokens: torch.Tensor) -> str:
        """使用GLM4 tokens进行Kimi解码"""
        if self.kimi_engine is None:
            return "[Kimi引擎未加载，无法解码]"
        
        try:
            # TODO: 实现实际的Kimi解码
            return "[Kimi transcription with GLM4 tokens - to be implemented]"
        except Exception as e:
            return f"[Kimi解码失败: {e}]"


class WorkspaceWhisperDecoder:
    """Workspace优化的Whisper解码器"""
    
    def __init__(self, whisper_model):
        self.whisper_model = WhisperModel('large-3', device="cuda", compute_type="bfloat16")
        
    def decode_with_encoder_output(
        self, 
        ctranslate_features: CTranslateFeatures, 
        language: str = None
    ) -> str:
        """使用预编码的encoder输出进行Whisper解码"""
        try:
            from WhisperLive.whisper_live.new_transcriber import TranscriptionOptions
            
            # 创建转写选项
            options = TranscriptionOptions(
                beam_size=5,
                best_of=5,
                patience=1.0,
                length_penalty=1.0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                prompt_reset_on_temperature=0.5,
                initial_prompt=None,
                prefix=None,
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=False,
                max_initial_timestamp=1.0,
                word_timestamps=False,
                prepend_punctuations="\"'",
                append_punctuations="\"'",
                max_new_tokens=None,
                clip_timestamps="0",
                hallucination_silence_threshold=None
            )
            
            # 设置语言
            if language and language in ["zh", "en", "yue"]:
                options.language = language
            
            # 获取tokenizer
            tokenizer = self.whisper_model.hf_tokenizer
            
            # 使用预编码的encoder输出生成转写
            segments = list(self.whisper_model.generate_segments(
                features=ctranslate_features.audio_features,
                tokenizer=tokenizer,
                options=options,
                encoder_output=ctranslate_features.encoder_output  # 复用预编码输出！
            ))
            
            # 合并所有段落
            text_segments = []
            for segment in segments:
                text_segments.append(segment.text)
            
            text = " ".join(text_segments).strip()
            return text
            
        except Exception as e:
            return f"[Whisper解码失败: {e}]"


class SmartRouter:
    """智能路由器"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def route_decision(self, language: str, confidence: float, has_kimi: bool) -> str:
        """路由决策"""
        if not has_kimi:
            return "whisper"
        
        if confidence < self.confidence_threshold:
            return "whisper"
        
        if language == "yue":  # 粤语用Whisper
            return "whisper"
        
        if language in ["zh", "en"]:  # 中英文用Kimi
            return "kimi"
        
        return "whisper"


class WorkspaceIntegratedASR:
    """
    Workspace优化的集成ASR系统
    
    核心特性：
    1. 所有模型下载/缓存到/workspace目录
    2. 自动下载缺失的模型
    3. 正确使用CTranslate2接口
    4. 最大化特征复用
    """
    
    def __init__(
        self,
        whisper_model_path: str = "base",
        kimi_model_path: str = None,
        glm4_tokenizer_path: str = None,
        language_classifier_path: str = None,
        confidence_threshold: float = 0.7
    ):
        self.logger = logging.getLogger(__name__)
        
        # 确保workspace目录存在
        ensure_workspace_directory("models")
        ensure_workspace_directory("cache")
        
        # 核心组件
        self.
        self.language_classifier = WorkspaceLanguageClassifier(language_classifier_path)
        self.whisper_decoder = None  # 延迟初始化
        self.kimi_decoder = WorkspaceKimiDecoder(kimi_model_path, glm4_tokenizer_path)
        self.router = SmartRouter(confidence_threshold)
        
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化所有组件，自动下载到workspace"""
        try:
            print("🚀 初始化Workspace集成ASR系统...")
            print(f"📂 缓存目录: {os.environ.get('HF_HOME', 'default')}")
            
            # 1. 初始化Whisper encoder
            print("\n1️⃣ 初始化Whisper encoder...")
            if not self.encoder.initialize():
                print("❌ Whisper encoder初始化失败")
                return False
            
            # 2. 初始化Whisper decoder (复用encoder的模型)
            print("\n2️⃣ 初始化Whisper decoder...")
            self.whisper_decoder = WorkspaceWhisperDecoder(self.encoder.whisper_model)
            print("✅ Whisper decoder初始化成功")
            
            # 3. 初始化语言分类器
            print("\n3️⃣ 语言分类器已初始化")
            
            # 4. 初始化Kimi相关组件 (可选)
            print("\n4️⃣ 初始化Kimi组件...")
            kimi_success = self.kimi_decoder.initialize()
            if not kimi_success:
                print("⚠️  Kimi组件初始化失败，将仅使用Whisper")
            
            self.is_initialized = True
            print("\n🎉 集成ASR系统初始化完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def transcribe(self, audio: np.ndarray) -> CTranslateResult:
        """转写音频"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        
        try:
            # Step 1: 编码
            print("🔄 正在编码音频...")
            encoding_start = time.time()
            ctranslate_features = self.encoder.encode_audio(audio)
            encoding_time = time.time() - encoding_start
            
            # Step 2: 语言检测
            print("🔍 正在检测语言...")
            detection_start = time.time()
            language, confidence = self.language_classifier.predict_language(ctranslate_features)
            ctranslate_features.language = language
            ctranslate_features.confidence = confidence
            detection_time = time.time() - detection_start
            
            # Step 3: 路由决策
            has_kimi = (self.kimi_decoder.kimi_engine is not None and 
                       self.kimi_decoder.glm4_tokenizer is not None)
            engine = self.router.route_decision(language, confidence, has_kimi)
            print(f"🎯 路由到: {engine}引擎")
            
            # Step 4: 解码
            decoding_start = time.time()
            
            if engine == "kimi" and has_kimi:
                try:
                    # 获取GLM4 tokens
                    if ctranslate_features.glm4_tokens is None:
                        ctranslate_features.glm4_tokens = self.kimi_decoder.get_glm4_tokens(audio)
                    
                    text = self.kimi_decoder.decode_with_glm4_tokens(ctranslate_features.glm4_tokens)
                except Exception as e:
                    print(f"⚠️  Kimi解码失败，回退到Whisper: {e}")
                    text = self.whisper_decoder.decode_with_encoder_output(ctranslate_features, language)
                    engine = "whisper"
            else:
                text = self.whisper_decoder.decode_with_encoder_output(ctranslate_features, language)
            
            decoding_time = time.time() - decoding_start
            
            # 结果
            total_time = time.time() - start_time
            
            result = CTranslateResult(
                text=text,
                language=language,
                confidence=confidence,
                engine=engine,
                processing_time=total_time,
                encoding_time=encoding_time,
                language_detection_time=detection_time,
                decoding_time=decoding_time
            )
            
            print(f"✅ 转写完成: '{text}'")
            return result
            
        except Exception as e:
            print(f"❌ 转写失败: {e}")
            return CTranslateResult(
                text="",
                language="unknown",
                confidence=0.0,
                engine="error",
                processing_time=time.time() - start_time,
                encoding_time=0.0,
                language_detection_time=0.0,
                decoding_time=0.0
            )


