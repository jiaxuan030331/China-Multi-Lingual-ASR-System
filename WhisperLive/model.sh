#python3 run_server.py --port 8091 --backend faster_whisper -fw "/root/.cache/huggingface/hub/models--BELLE--Belle-whisper-large-v3-zh-punct-ct2"
#nohup python3 run_server.py --port 8091 --backend faster_whisper -fw "/root/.cache/huggingface/hub/Belle-whisper-large-v3-zh-punct-ct2_float16" &
nohup python3 model_server.py --port=8001 --model_path="/root/ASR_TTS_improvement/models/ct2-whisper-lora2" > model_server.log 2>&1 &
#nohup python3 model_server.py --port=8001 --model_path="/data/model/zh-hk-v2-0822" > model_server2.log 2>&1 &
#nohup python3 model_server.py --port=8001 --model_path="/data/model/zh-hk-v2-0822-10000" > model_server2.log 2>&1 &
#nohup python3 model_server.py --port=8001 --model_path="/data/model/zh-hk-v3-4000" > model_server2.log 2>&1 &
#python3 run_server.py --port 8091 --backend faster_whisper --faster_whisper_custom_model_path /root/.cache/huggingface/hub/Systran--faster-whisper-large-v3
