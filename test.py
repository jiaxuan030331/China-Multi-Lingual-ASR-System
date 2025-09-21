#!/usr/bin/env python3
"""
简单的导入测试脚本
"""
import torchaudio


print('🧪 测试ASR项目导入路径...')



from src.core.integrated_asr import IntegratedASR

asr = IntegratedASR()

audio2 = '/workspace/ASR/audio_examples/english.mp3'
audio3 = '/workspace/ASR/audio_examples/mandarin.mp3'

waveform, sr = torchaudio.load(audio2)
print(f'transcribe with kimi only')
text = asr.transcribe(audio = waveform, sr = sr,kimi_only=True)
print(text)
print(f'transcribe with ct2 only')
text = asr.transcribe(audio = waveform, sr = sr,ct2_only=True)
print(text)
print(f'transcribe with both')
text = asr.transcribe(audio = waveform, sr = sr)
print(text)

  






