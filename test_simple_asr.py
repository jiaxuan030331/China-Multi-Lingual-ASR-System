#!/usr/bin/env python3
"""
测试精简版IntegratedASR系统
"""

import sys
import os
import logging
import numpy as np

# 添加路径
sys.path.append('/workspace/ASR')
sys.path.append('/workspace/ASR/kimi_deployment')

from src.core.integrated_asr_simple import IntegratedASR

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_whisper_only():
    """测试仅使用Whisper的情况"""
    print("🧪 测试1: 仅Whisper模式")
    
    asr = IntegratedASR(
        kimi_model_path=None,  # 不加载Kimi
        whisper_model_path="large-v3",
        confidence_threshold=0.7
    )
    
    if not asr.initialize():
        print("❌ 初始化失败")
        return False
    
    # 创建测试音频 (1秒的噪声)
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    try:
        result = asr.transcribe(test_audio)
        print(f"✅ 转写结果: '{result.text}'")
        print(f"   语言: {result.language}")
        print(f"   置信度: {result.confidence:.3f}")
        print(f"   引擎: {result.engine}")
        return True
    except Exception as e:
        print(f"❌ 转写失败: {e}")
        return False

def test_with_kimi():
    """测试包含Kimi的完整系统"""
    print("\n🧪 测试2: Kimi + Whisper双引擎模式")
    
    # 使用HuggingFace缓存的Kimi模型路径
    kimi_path = "moonshotai/Kimi-Audio-7B-Instruct"
    
    asr = IntegratedASR(
        kimi_model_path=kimi_path,
        whisper_model_path="large-v3",
        confidence_threshold=0.7
    )
    
    if not asr.initialize():
        print("❌ 初始化失败")
        return False
    
    # 创建测试音频
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    try:
        result = asr.transcribe(test_audio)
        print(f"✅ 转写结果: '{result.text}'")
        print(f"   语言: {result.language}")
        print(f"   置信度: {result.confidence:.3f}")
        print(f"   引擎: {result.engine}")
        return True
    except Exception as e:
        print(f"❌ 转写失败: {e}")
        return False

def test_with_real_audio():
    """使用真实音频文件测试"""
    print("\n🧪 测试3: 真实音频文件")
    
    audio_file = "/workspace/ASR/kimi_deployment/examples/asr_example_cantonese.wav"
    
    if not os.path.exists(audio_file):
        print(f"⚠️ 音频文件不存在: {audio_file}")
        return False
    
    try:
        import librosa
        audio, sr = librosa.load(audio_file, sr=16000)
        print(f"📁 加载音频: {audio_file}")
        print(f"   时长: {len(audio)/16000:.2f}秒")
        
        # 仅使用Whisper测试
        asr = IntegratedASR(
            kimi_model_path=None,
            whisper_model_path="large-v3"
        )
        
        if not asr.initialize():
            print("❌ 初始化失败")
            return False
        
        result = asr.transcribe(audio)
        print(f"✅ 转写结果: '{result.text}'")
        print(f"   语言: {result.language}")
        print(f"   置信度: {result.confidence:.3f}")
        print(f"   引擎: {result.engine}")
        return True
        
    except ImportError:
        print("❌ 需要安装librosa: pip install librosa")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_language_detection():
    """测试语言检测功能"""
    print("\n🧪 测试4: 语言检测")
    
    asr = IntegratedASR(whisper_model_path="large-v3")
    
    if not asr.initialize():
        print("❌ 初始化失败")
        return False
    
    # 测试不同长度的音频
    for duration in [1, 3, 5]:
        test_audio = np.random.randn(16000 * duration).astype(np.float32) * 0.1
        
        try:
            language, confidence = asr.detect_language(test_audio)
            engine = asr.route_engine(language, confidence)
            
            print(f"   {duration}秒音频: {language} (置信度:{confidence:.3f}) -> {engine}")
            
        except Exception as e:
            print(f"❌ {duration}秒音频检测失败: {e}")
            return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试精简版IntegratedASR系统")
    print("=" * 50)
    
    # 检查环境
    if not os.path.exists("/workspace/ASR/WhisperLive/language_fnn_only2.pt"):
        print("❌ 语言分类器模型文件不存在")
        return
    
    # 运行测试
    tests = [
        test_whisper_only,
        test_language_detection,
        test_with_real_audio,
        # test_with_kimi,  # 注释掉，因为可能需要下载大模型
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("✅ 测试通过\n")
            else:
                print("❌ 测试失败\n")
        except Exception as e:
            print(f"💥 测试异常: {e}\n")
    
    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️ 部分测试失败")

if __name__ == "__main__":
    main() 