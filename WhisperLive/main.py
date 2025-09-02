from whisper_live.client import TranscriptionClient


client = TranscriptionClient(
  "localhost", #ip 
  8092, #port  可以用language=“yue”指定，不指定language则yuce
  translate=False,
  model="large-v2",
  use_vad=False,
  save_output_recording=False,                         # Only used for microphone input, False by Default
  output_recording_filename="./output_recording.wav"  # Only used for microphone input
)
client("Chinese_male.wav")
#接收麦克风语音
#client() 
