#Make sure the backend is running, activation: ./scripts/run_integrated_asr.sh

"""
简单的导入测试脚本
"""
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import soundfile as sf, numpy as np, librosa, requests

path = "/workspace/ASR/audio_examples/cantonese.wav"
wav, sr = sf.read(path, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=1)
if sr != 16000:
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

data = wav.astype(np.float32).tobytes()
r = requests.post(
    "http://localhost:8001/transcribe_websocket",
    headers={"Content-Type": "application/octet-stream", "prep_timeout": "60"},
    data=data
)