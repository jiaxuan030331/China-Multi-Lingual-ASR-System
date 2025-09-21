# src/app/load_model.py

from src.core.integrated_asr import IntegratedASR
import threading

model_lock = threading.Lock()
_model_instance = None
_model_config = {}

def load_integrated_asr(
    ct2_model_path: str = 'large-v3',
    kimi_model_path_or_name: str = "moonshotai/Kimi-Audio-7B-Instruct", 
    lid_model_path: str = "/workspace/ASR/WhisperLive/language_fnn_only2.pt",
    confidence_threshold: float = 0.7
):
    """
    加载或复用一个 IntegratedASR 模型实例，支持可配置参数。
    若参数与已有模型不一致，会抛出错误防止重复加载。
    
    参数:
    - ct2_model_path: CTranslate2 Whisper模型路径
    - kimi_model_path_or_name: Kimi模型路径或名称
    - lid_model_path: 语言识别FNN模型路径  
    - confidence_threshold: 语言置信度阈值，决定是否使用Kimi
    """
    global _model_instance, _model_config

    requested_config = {
        "ct2_model_path": ct2_model_path,
        "kimi_model_path_or_name": kimi_model_path_or_name,
        "lid_model_path": lid_model_path,
        "confidence_threshold": confidence_threshold
    }

    with model_lock:
        if _model_instance is None:
            _model_instance = IntegratedASR(**requested_config)
            _model_config = requested_config
        elif _model_config != requested_config:
            raise RuntimeError(
                f"IntegratedASR 模型已加载，参数不一致。\n当前: {_model_config}\n请求: {requested_config}"
            )

    return _model_instance

def get_model_instance():
    """获取当前模型实例，如果未加载则使用默认参数加载"""
    global _model_instance
    if _model_instance is None:
        return load_integrated_asr()
    return _model_instance

def close_model():
    """关闭模型并清理资源"""
    global _model_instance, _model_config
    with model_lock:
        if _model_instance is not None:
            _model_instance.close()
            _model_instance = None
            _model_config = {} 