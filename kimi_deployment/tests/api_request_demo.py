#测试前请运行 kimi_deployment/scripts/run_kimi_server.sh，确认测试端口一致

import requests
import numpy as np
import soundfile as sf
import os
import sys 

# === 回到项目根目录 ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)  

# === 1. 加载音频文件，转 float32 PCM ===
wav_path = "./tests/test_audios/asr_example_mandarin.wav"
if not os.path.exists(wav_path):
    print(f"❌ 音频文件未找到: {wav_path}")
    exit(1)

try:
    audio_data, sample_rate = sf.read(wav_path, dtype="float32")
except Exception as e:
    print(f"❌ 音频加载失败: {e}")
    exit(1)

pcm_bytes = audio_data.astype(np.float32).tobytes()

# === 2. 构造请求头 + 发送 POST 请求 ===
headers = {
    "name": "self_record27.wav",
    "use_custom_language_classifier": "true"
}

url = "http://localhost:8000/transcribe_websocket"  # 或换成: http://0.0.0.0:8000 或具体IP

try:
    response = requests.post(url, headers=headers, data=pcm_bytes, timeout=30)
    print("✅ 请求发送成功")
    print("状态码:", response.status_code)
    print("返回内容:", response.text)
except requests.exceptions.RequestException as e:
    print(f"❌ 请求失败: {e}")


