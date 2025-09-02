
import os 
import sys

# === 回到项目根目录 ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)  # 确保 app 模块可导入

from app.load_model import load_kimi_model
from app.transcribe import transcribe_auto

def main():
    audio_path = "./tests/test_audios/asr_example_mandarin.wav"

    print("🌀 正在加载 Kimi 模型...")
    model = load_kimi_model()

    print("\n🎧 [TEST 1] 默认参数识别")
    result1 = transcribe_auto(model, audio_path)
    print("✅ 结果 1：", result1)

    print("\n🎧 [TEST 2] 指定 prompt")
    result2 = transcribe_auto(model, audio_path, prompt="请将以下音频内容转写为中文文本。")
    print("✅ 结果 2：", result2)

    print("\n🎧 [TEST 3] 指定 language = 'Chinese'")
    result3 = transcribe_auto(model, audio_path, language="Chinese")
    print("✅ 结果 3：", result3)

    print("\n🎧 [TEST 4] 自定义采样参数 + English")
    sampling_config = {
        "audio_temperature": 0.5,
        "audio_top_k": 3,
        "text_temperature": 0.5,
        "text_top_k": 3,
        "audio_repetition_penalty": 1.2,
        "audio_repetition_window_size": 32,
        "text_repetition_penalty": 1.2,
        "text_repetition_window_size": 32,
    }
    result4 = transcribe_auto(
        model,
        audio_path,
        transcribe_params=sampling_config,
        language="English"
    )
    print("✅ 结果 4：", result4)

if __name__ == "__main__":
    main()