#!/usr/bin/env python3
"""
MP3音频处理问题诊断脚本
"""
import torchaudio
import librosa
import numpy as np
import torch

def analyze_audio_file(file_path):
    """分析音频文件的各种加载方式"""
    print(f"\n🔍 分析文件: {file_path}")
    print("="*50)
    
    # 方法1: torchaudio.load (当前test.py使用的)
    try:
        waveform_torch, sr_torch = torchaudio.load(file_path)
        print(f"📊 torchaudio.load:")
        print(f"   类型: {type(waveform_torch)}")
        print(f"   形状: {waveform_torch.shape}")
        print(f"   dtype: {waveform_torch.dtype}")
        print(f"   设备: {waveform_torch.device}")
        print(f"   采样率: {sr_torch}")
        print(f"   数值范围: [{waveform_torch.min():.6f}, {waveform_torch.max():.6f}]")
        print(f"   是否立体声: {waveform_torch.shape[0] > 1}")
        
        # 检查前10个采样点
        print(f"   前10个采样: {waveform_torch[0, :10].tolist()}")
        
    except Exception as e:
        print(f"❌ torchaudio.load失败: {e}")
        waveform_torch, sr_torch = None, None
    
    # 方法2: librosa.load (推荐方式)
    try:
        waveform_librosa, sr_librosa = librosa.load(file_path, sr=16000)
        print(f"\n📊 librosa.load(sr=16000):")
        print(f"   类型: {type(waveform_librosa)}")
        print(f"   形状: {waveform_librosa.shape}")
        print(f"   dtype: {waveform_librosa.dtype}")
        print(f"   采样率: {sr_librosa}")
        print(f"   数值范围: [{waveform_librosa.min():.6f}, {waveform_librosa.max():.6f}]")
        
        # 检查前10个采样点
        print(f"   前10个采样: {waveform_librosa[:10].tolist()}")
        
    except Exception as e:
        print(f"❌ librosa.load失败: {e}")
        waveform_librosa, sr_librosa = None, None
    
    # 方法3: librosa.load原始采样率
    try:
        waveform_orig, sr_orig = librosa.load(file_path, sr=None)
        print(f"\n📊 librosa.load(sr=None - 原始):")
        print(f"   采样率: {sr_orig}")
        print(f"   形状: {waveform_orig.shape}")
        print(f"   数值范围: [{waveform_orig.min():.6f}, {waveform_orig.max():.6f}]")
        
    except Exception as e:
        print(f"❌ librosa.load(原始)失败: {e}")
    
    # 对比分析
    if waveform_torch is not None and waveform_librosa is not None:
        print(f"\n🔄 数据对比:")
        # 转换torch数据到librosa格式进行对比
        torch_mono = waveform_torch.mean(dim=0) if waveform_torch.shape[0] > 1 else waveform_torch[0]
        torch_16k = torchaudio.functional.resample(torch_mono, sr_torch, 16000)
        torch_np = torch_16k.detach().cpu().numpy()
        
        print(f"   转换后torch形状: {torch_np.shape}")
        print(f"   librosa形状: {waveform_librosa.shape}")
        print(f"   数值差异 (前100个点MSE): {np.mean((torch_np[:100] - waveform_librosa[:100])**2):.8f}")
        
        # 检查是否完全不同
        correlation = np.corrcoef(torch_np[:min(len(torch_np), len(waveform_librosa))], 
                                 waveform_librosa[:min(len(torch_np), len(waveform_librosa))])[0,1]
        print(f"   波形相关性: {correlation:.6f}")
        
        if correlation < 0.8:
            print("   ⚠️  警告: 不同加载方式产生了显著不同的音频数据!")
        
    return waveform_torch, sr_torch, waveform_librosa, sr_librosa

def test_incorrect_transcribe_call():
    """测试错误的transcribe调用方式"""
    print(f"\n🧪 测试当前test.py的调用方式:")
    print("="*50)
    
    from src.core.integrated_asr import IntegratedASR
    
    # 模拟test.py的错误调用
    audio_file = '/workspace/ASR/audio_examples/mandarin.mp3'
    waveform, sr = torchaudio.load(audio_file)
    
    print(f"音频形状: {waveform.shape}")
    print(f"采样率: {sr}")
    print(f"数据类型: {waveform.dtype}")
    
    # 这是test.py中的错误调用方式
    print(f"\n❌ 错误调用: asr.transcribe(waveform={waveform.shape}, sr={sr})")
    print(f"   实际效果: asr.transcribe(audio={waveform.shape}, prep_timeout={sr})")
    print(f"   问题1: 音频可能是立体声torch.Tensor, 不是单声道np.ndarray")
    print(f"   问题2: 采样率{sr}被当作超时时间!")
    print(f"   问题3: 没有重采样到16kHz!")

def create_correct_loader():
    """创建正确的音频加载函数"""
    print(f"\n✅ 正确的MP3加载方式:")
    print("="*50)
    
    code = '''
def load_audio_correct(file_path):
    """正确加载MP3文件为ASR可用格式"""
    import librosa
    import numpy as np
    
    # 直接加载为16kHz单声道float32
    audio, _ = librosa.load(file_path, sr=16000, mono=True)
    return audio.astype(np.float32)

# 正确的调用方式
audio = load_audio_correct("file.mp3")
result = asr.transcribe(audio)  # 只传audio参数!
'''
    print(code)

if __name__ == "__main__":
    # 分析所有音频文件
    audio_files = [
        '/workspace/ASR/audio_examples/mandarin.mp3',
        '/workspace/ASR/audio_examples/english.mp3',
        '/workspace/ASR/audio_examples/cantonese.wav'
    ]
    
    for file_path in audio_files:
        try:
            analyze_audio_file(file_path)
        except Exception as e:
            print(f"❌ 分析{file_path}失败: {e}")
    
    test_incorrect_transcribe_call()
    create_correct_loader()
    
    print(f"\n🎯 总结:")
    print("1. test.py使用了错误的函数调用方式")
    print("2. torchaudio.load可能加载立体声数据,需要转单声道")  
    print("3. 采样率没有重采样到16kHz")
    print("4. 数据类型是torch.Tensor,不是np.ndarray")
    print("5. 建议使用librosa.load(sr=16000)直接加载") 