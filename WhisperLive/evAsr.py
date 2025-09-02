from faster_whisper import WhisperModel

if __name__ == "__main__":
    model_size = "/root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v2"
    #'../model/faster-whisper-large-v3'
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe("01.wav", language='zh',  beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
