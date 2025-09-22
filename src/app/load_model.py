# src/app/load_model.py

from src.core.integrated_asr import IntegratedASR
import threading

model_lock = threading.Lock()
_model_instance = None
_model_config = {}

def load_integrated_asr(
    ct2_model_path: str = 'large-v3',
    kimi_model_path_or_name: str = "moonshotai/Kimi-Audio-7B-Instruct", 
    lid_model_path: str = "/workspace/ASR/models/lid/language_fnn_only2.pt",
    confidence_threshold: float = 0.7
):
    """
    Load or reuse a single IntegratedASR model instance with configurable parameters.
    If parameters differ from the existing instance, raise an error to prevent reloading.
    
    Args:
    - ct2_model_path: path or name of the CTranslate2 Whisper model
    - kimi_model_path_or_name: path or name of the Kimi model
    - lid_model_path: path to language identification FNN model
    - confidence_threshold: language confidence threshold to route to Kimi
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
                f"IntegratedASR already loaded with different parameters.\nCurrent: {_model_config}\nRequested: {requested_config}"
            )

    return _model_instance

def get_model_instance():
    """Get the current model instance; load with default params if not loaded."""
    global _model_instance
    if _model_instance is None:
        return load_integrated_asr()
    return _model_instance

def close_model():
    """Close the model and release resources."""
    global _model_instance, _model_config
    with model_lock:
        if _model_instance is not None:
            _model_instance.close()
            _model_instance = None
            _model_config = {} 