import os
import soundfile as sf
import sys 

# === 回到项目根目录 ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)  # 确保 app 模块可导入

from kimia_infer.api.kimia import KimiAudio
from app.transcribe import transcribe_from_waveform

if __name__ == "__main__":

    model = KimiAudio(
        model_path="moonshotai/Kimi-Audio-7B-Instruct",
        load_detokenizer=False,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    
    # 官方提供的Message格式ASR示例（仅支持路径文件）
    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
        {
            "role": "user",
            "message_type": "audio",
            "content": './tests/test_audios/asr_example_mandarin.wav',
        },
    ]

    wav, text = model.generate(messages, **sampling_params, output_type="text")
    print(">>> Official demo output text: ", text)

    # 拆分message封装自定义封装推理（支持路径文件和裸音频数据）

    #将音频加载为waveform
    audio_path = './tests/test_audios/asr_example_mandarin.wav'
    wav, sr = sf.read(audio_path)

    print(">>> Self defined api output text: ",transcribe_from_waveform(model = model,waveform = wav, prompt="请将音频内容转换为文字。", language=None, transcribe_params=sampling_params)['text'])
    


   
    
