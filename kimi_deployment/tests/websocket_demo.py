import websocket
import json
import soundfile as sf
import numpy as np
import time

def main():
    # === 1. 建立 WebSocket 连接 ===
    ws_url = "ws://127.0.0.1:9091"
    print(f"Connecting to {ws_url}")
    ws = websocket.create_connection(ws_url)

    # === 2. 发送初始化参数 ===
    options = {
        "uid": "test_user",
        "token": "xxx",
        "name": "test.wav",
        "initial_prompt": "",
        "version": "1.0",
        "model": "kimi",
        "user_id": "abc123",
        "type_name": "developer"
    }
    ws.send(json.dumps(options))
    print("Sent initialization options.")

    # === 3. 加载音频，并转成 float32 ===
    wav_path = "./tests/test_audios/asr_example_mandarin.wav"
    print(f"Loading audio from {wav_path}")
    audio_data, sample_rate = sf.read(wav_path, dtype='float32')
    assert sample_rate == 16000, "音频采样率必须为 16kHz"

    # === 4. 转成 int16 PCM（分帧发送） ===
    int16_audio = (audio_data * 32768.0).astype(np.int16)
    frame_duration = 1  # 每帧 0.4 秒，可调
    frame_size = int(sample_rate * frame_duration)  
    num_frames = len(int16_audio) // frame_size

    print(f"Sending audio in {num_frames+1} frames...")

    for i in range(num_frames + 1):
        start = i * frame_size
        end = start + frame_size
        frame = int16_audio[start:end]
        if len(frame) == 0:
            continue
        ws.send(frame.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
        time.sleep(frame_duration)  # 模拟实时发送

    # === 5. 发送结束标志 ===
    ws.send(b"END_OF_AUDIO")
    print("Sent END_OF_AUDIO.")

    # === 6. 接收并打印返回内容 ===
    print("Receiving results...")
    while True:
        try:
            msg = ws.recv()
            if not msg:
                break
            print("收到返回：", msg)
            if '"is_end": true' in msg:
                print("✅ 服务端识别结束")
                break
        except Exception as e:
            print(f"连接关闭或异常：{e}")
            break
    
    # === 7. 关闭连接 ===
    ws.close()
    print("WebSocket closed.")

if __name__ == "__main__":
    main()