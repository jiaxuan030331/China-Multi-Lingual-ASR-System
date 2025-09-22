import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import websocket._core as websocket
import json
import soundfile as sf
import numpy as np
import time
def quick_test():
        ws = websocket.create_connection("ws://127.0.0.1:9092")
        print("✅ WebSocket connection established!")
        
        # Correct initialization message
        init_msg = {
            "uid": "test_user_123",
            "token": "test_token",
            "name": "test_audio.wav", 
            "version": "1.0",  # 🔑 Required version number
            "model": "faster_whisper",
            "initial_prompt": "",
            "user_id": "test123",
            "type_name": "developer"
        }
        
        print("📤 Sending initialization message...")
        ws.send(json.dumps(init_msg))
        
        # === 3. Load audio and convert to float32 ===
        wav_path = "/workspace/ASR/audio_examples/self_record51.wav"
        print(f"Loading audio from {wav_path}")
        audio_data, sample_rate = sf.read(wav_path, dtype='float32')
        assert sample_rate == 16000, "Audio sample rate must be 16kHz"

        # === 4. Convert to int16 PCM (send in frames) ===
        int16_audio = (audio_data * 32768.0).astype(np.int16)
        frame_duration = 3  # Each frame is 0.4 seconds, adjustable
        frame_size = int(sample_rate * frame_duration)  
        num_frames = len(int16_audio) // frame_size

        print(f"Sending audio in {num_frames+1} frames...")
        #ws.send(int16_audio.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
        
        for i in range(num_frames + 1):
            start = i * frame_size
            end = start + frame_size
            frame = int16_audio[start:end]
            if len(frame) == 0:
                continue
            ws.send(frame.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
            time.sleep(frame_duration)  # Simulate real-time sending

        # === 5. Send end-of-audio flag ===
        ws.send(b"END_OF_AUDIO")
        print("Sent END_OF_AUDIO.")

        # === 6. Receive and print returned content ===
        print("Receiving results...")
        while True:
            try:
                msg = ws.recv()
                if not msg:
                    break
                print("Received response:", msg)
                if '"is_end": true' in msg:
                    print("✅ Server recognition finished")
                    break
            except Exception as e:
                print(f"Connection closed or exception: {e}")
                break
        
        # === 7. Close connection ===
        ws.close()
        print("WebSocket closed.")


quick_test()
