#!/usr/bin/env python3

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# 导入封装层
from src.app import load_integrated_asr, transcribe_auto

def main():
    print("🧪 简化测试：IntegratedASR应用层")
    
    # 1. 加载模型（使用默认参数）
    print("📥 加载模型...")
    model = load_integrated_asr()
    print("✅ 模型加载完成")
    
    # 2. 测试音频文件（如果存在）
    audio_files = [
        "/workspace/ASR/audio_examples/mandarin.mp3",
        "/workspace/ASR/audio_examples/english.mp3", 
        "/workspace/ASR/audio_examples/cantonese.wav"
    ]
    
    for audio_file in audio_files:
        try:
            print(f"\n🎵 处理: {audio_file}")
            result = transcribe_auto(model, audio_file)
            
            if result["status"] == 0:
                print(f"✅ 结果: {result['text']}")
                print(f"   引擎: {result['engine']}, 语言: {result['language']}")
            else:
                print(f"❌ 失败: {result['error']}")
                
        except Exception as e:
            print(f"💥 异常: {e}")

if __name__ == "__main__":
    main() 