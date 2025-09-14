# app/load_model.py

from kimi_deployment.kimia_infer.api.kimia import KimiAudio
import threading

model_lock = threading.Lock()
_model_instance = None
_model_config = {}

def load_kimi_model(
    model_path: str = None,
    device: str = "cuda",
    device_index: int = 0,
    torch_dtype: str = "bfloat16",
    load_detokenizer: bool = False  # ✅ 默认不加载以节省显存
):
    """
    加载或复用一个 Kimi 模型实例，支持可配置参数。
    若参数与已有模型不一致，会抛出错误防止重复加载。

    - load_detokenizer: 设置为 False 可减少显存占用（关闭音频生成功能）
    """
    global _model_instance, _model_config

    requested_config = {
        "model_path": model_path,
        "device": device,
        "device_index": device_index,
        "torch_dtype": torch_dtype,
        "load_detokenizer": load_detokenizer
    }

    with model_lock:
        if _model_instance is None:
            _model_instance = KimiAudio(**requested_config)
            _model_config = requested_config
        elif _model_config != requested_config:
            raise RuntimeError(
                f"Kimi 模型已加载，参数不一致。\n当前: {_model_config}\n请求: {requested_config}"
            )

    return _model_instance
