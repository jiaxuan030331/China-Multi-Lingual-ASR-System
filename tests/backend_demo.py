#!/usr/bin/env python3

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torchaudio


print('🧪 测试ASR项目导入路径...')



from src.core.integrated_asr import IntegratedASR

asr = IntegratedASR()
audio1 = '/workspace/ASR/audio_examples/cantonese.wav'
audio2 = '/workspace/ASR/audio_examples/english.mp3'
audio3 = '/workspace/ASR/audio_examples/mandarin.mp3'

for audio in [audio1, audio2, audio3]:
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

  






