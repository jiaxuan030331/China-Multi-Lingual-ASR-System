#python3 run_server.py --port 8091 --backend faster_whisper -fw "/root/.cache/huggingface/hub/models--BELLE--Belle-whisper-large-v3-zh-punct-ct2"
#nohup python3 run_server.py --port 8091 --backend faster_whisper -fw "/root/.cache/huggingface/hub/Belle-whisper-large-v3-zh-punct-ct2_float16" &
nohup python3 run_server.py --port 8091 --backend faster_whisper -fw "/root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3" &
#python3 run_server.py --port 8091 --backend faster_whisper --faster_whisper_custom_model_path /root/.cache/huggingface/hub/Systran--faster-whisper-large-v3
