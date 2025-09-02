# Whisper live项目更新 （2025.07.03）
 
1.虚拟环境改动，whisper在更新ubuntu 22.4后，使用GPU加载faster whispher可能由于内部引用原因持续调用libcublas.so.12导致与系统so.11兼容性问题，改用whisperlive环境运行

2.为兼容新系统，虚拟环境，升级fastwhisper 1.0.1 -> 1.1.1, 升级后代码改动：
 -- collect_chunks函数返回格式改动 (whisper_live/new_transcriber.py line 351-354):
    
    audio = collect_chunks(audio, speech_chunks)
    -> 
    audio,_ = collect_chunks(audio, speech_chunks)
    audio = np.concatenate(audio, axis=0)

 -- feature extractor 比原版本大3000（30秒），识别时去除 减去30秒的对齐（whisper_live/new_transcriber.py line 510):
    
    content_frames = features.shape[-1] - self.feature_extractor.nb_max_frames
    ->
    content_frames = features.shape[-1]

3. 新增new_transcriber.py支持加载/使用自定义语种识别模型

 -- 新增使用自定义语种识别方法，详见 whisper_live/new_transcriber.py line 1212 - 1252

 -- 新增创建faster whisper类对象时加载语种识别模型选项，详见 whisper_live/new_transcriber.py line 134 - 154

 -- 新增ASR识别函数是否使用自定义模型识别语种选项, 调整函数的语种决定逻辑，详见 whisper_live/new_transcriber.py line 400 - 432

 经测试，自定义单隐藏层FNN语种识别模型几乎与默认语种识别耗时相同，中，英，粤三语验证集精度99.9%以上

4. 修改model_server.py支持基于new_transcriber封装的推理函数的tornado http API

 -- 新增加载自定义语种识别模型的参数CLI: language_classifier_path,默认模型为/root/ASR_TTS_improvement/models/language_fnn_only2.pt

 -- 默认ASR模型改为：/root/ASR_TTS_improvement/models/ct2-whisper-lora2（为中英粤三语lora微调）
    
 -- 新增转写请求header参数：use_custom_language_classifier,默认为'true'

 -- 修改ASR模型返回的json格式，由于原本segment，vadoptions类无法直接由json.dump识别，对于返回内容加入提前处理，以兼容websocket服务的结构返回

5. 修改








