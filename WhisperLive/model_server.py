#coding:utf-8

import tornado.web
import tornado.httpserver
import tornado.options
import tornado.ioloop
import platform
import os
import threading
import json
from whisper_live.new_transcriber import WhisperModel
import multiprocessing
import numpy as np
from tornado.options import options , define
define("port",default=8001,help="跑在8001",type=int)
define("worker",default=2,help="跑在8001",type=int)
define("model_path",default='/root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478',help="跑在8001",type=str)
define("language_classifier_path",default=None,help="跑在8001",type=str)

import time



        
def create_model(device, model_size_or_path,language_classifier_path, device_index=0):
    print('language_classifier_path: ',language_classifier_path)
    transcriber = WhisperModel(
        model_size_or_path,
        device=device,
        device_index = device_index,
        compute_type="int8" if device == "cpu" else "float16",
        local_files_only=False,
        language_classifier_path=language_classifier_path,
    )
    #print('loaded')
    return transcriber


class Singleton(object):
    _instance = None
    _models = None
    SINGLE_MODEL_LOCK = threading.Lock()
    def __new__(cls, *args, **kwargs):
        # 实例不存在创建实例
        #key = str(args) + str(kwargs)
        #if key not in cls._instance_:
        #    cls._instance_[key] = super().__new__(cls)

        if cls._instance is None:
            cls._instance = super().__new__(cls)

            #time.sleep(3)

            model_size_or_path = '' if 'model_path' not in kwargs else kwargs['model_path']
            device = 'cpu' if 'device' not in kwargs else kwargs['device']
            device_index = 0 if 'device_index' not in kwargs else kwargs['device_index']
            language_classifier_path = None if 'language_classifier_path' not in kwargs else kwargs['language_classifier_path']
            print('init cls.models ---------')
            cls._model_id = 0 if 'model_id' not in kwargs else kwargs['model_id']
            print(model_size_or_path)
            print(cls._model_id)

            #def create_model(self, device, model_size_or_path, device_index=0):
            cls._models = create_model(device=device, model_size_or_path = model_size_or_path, device_index =device_index,language_classifier_path=language_classifier_path)
            cls.task = "transcribe"
            #cls.initial_prompt = None
            cls.vad_parameters = {"threshold": 0.5}
            cls.use_vad = True
        # 实例存在直接返回
        return cls._instance

    def get_models(self):
        return self._models

    def get_model_id(self):
        return self._model_id

    def transcribe_audio(self, input_sample, language, initial_prompt,use_custom_language_classifier):
        #print('transcribe audio : ' + str(initial_prompt))
        
        Singleton.SINGLE_MODEL_LOCK.acquire()

        try:
            
            result= self._models.transcribe(
                input_sample,
                initial_prompt=initial_prompt,
                language=language,
                use_custom_language_classifier=use_custom_language_classifier,
                task=self.task,
                vad_filter=self.use_vad,
                vad_parameters=self.vad_parameters if self.use_vad else None)
            
            # 判断返回结果类型
            if isinstance(result, tuple):
                segments, info = result
            else:
                segments = result
                info = None  

        except Exception as e:
            print(e)
            result = []
            info = None
        
        Singleton.SINGLE_MODEL_LOCK.release()
        return result, info

    def __init__(self,*args, **kwargs):
        pass


class MainHandler(tornado.web.RequestHandler):
    def initialize(self, infos):
        #self.quedict = manager.Queue()
        try:
            infos = infos['queue']
            if infos.empty() == False:
                mode_info = infos.get(block=False)
                print(mode_info)

                single1 = Singleton(device_index=mode_info['device_index'], model_path=mode_info['model_path'],
                                    device=mode_info['device'], model_id=mode_info['model_id'],language_classifier_path=mode_info['language_classifier_path'])
                model_id = single1.get_model_id()
                if model_id != mode_info['model_id']:
                    infos.put(mode_info)
                    print(mode_info)
                self.model = single1

        except Exception as e:
            print(e)
        #single1 = Singleton(device_index=0, model_path='', device='cuda')
        #self.model = single1.get_models()
        print("Init models !!!" + str(os.getpid()))

    def get(self):
        time.sleep(1)
        self.write("this is SleepHandler...")

    def post(self):
        headers = self.request.headers
        print(self.request.headers)
        try:
            file_name = headers.get('name')
            language = None if not headers.get('language') else headers.get('language')
            initial_prompt = None if not headers.get('initial_prompt') else headers.get('initial_prompt')
            use_custom_language_classifier = True if not headers.get('use_custom_language_classifier') else headers.get('use_custom_language_classifier').lower() == 'true'
            data = self.request.body

            if not data:
                self.write('{ "status" : -2, "message" : "wave data is null!" }')
            else:
                np_data = np.frombuffer(data, dtype=np.float32)
                
            segments, info = self.model.transcribe_audio(np_data, language, initial_prompt, use_custom_language_classifier)
            result = [s._asdict() if hasattr(s, "_asdict") else dict(s) for s in segments[0]]
            #print('result: ', result)
            info_dict = segments[1]._asdict()  # segments 是 TranscriptionInfo 实例

            # 转换嵌套字段
            if "transcription_options" in info_dict:
                info_dict["transcription_options"] = str(info_dict["transcription_options"])

            if "vad_options" in info_dict:
                info_dict["vad_options"] = None  # 或直接删除： del info_dict["vad_options"]

        
           

            # 你可以作为 info 字段使用
            info = info_dict
            
            # 构建最终 JSON 返回值
            json_return = {
                "result":result,
                "info": info,
                "status": 0
            }

            
            return_str = json.dumps(json_return, ensure_ascii=False)
            #print("✅ JSON Preview:", return_str)
            self.write(return_str)
        except Exception as e:
            json_return = '{ "status" : -1, "message" : "Input Data is wrong!" }'
            print(e)
            self.write(json_return)


        #resampled_file = 'test.wav'
        #with open(resampled_file, "wb") as wavfile:
        #    wavfile.write(data)

        #self.write("OK!!!!!")

def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(MainHandler)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()

def init_queue(queue_info, workers, model_path, device,language_classifier_path):
    for i in range(workers):
        model_info1 = {'device_index': 0, 'device': device, 'model_path': model_path, 'model_id': i+1, 'language_classifier_path': language_classifier_path}
        queue_info.put(model_info1)
        #print('init_queue ' + str(model_info1))
        

 
if __name__ == "__main__":
    tornado.options.parse_command_line()
    sysstr = platform.system()
    if (sysstr == "Windows"):
        multiprocessing.freeze_support()

    manager = multiprocessing.Manager()
    queue_info = manager.Queue()
    quedict = {'queue': queue_info}
    #data = que_input.get(block=False)
    init_queue(queue_info, 5, options.model_path, 'cuda', options.language_classifier_path)
    print('Server Start ' + str(options.port))
    print('Server Start ' + str(options.model_path))
    app = tornado.web.Application(
        handlers=[(r"/",MainHandler, {"infos":quedict})],
        debug = False
    )
    http_server = tornado.httpserver.HTTPServer(app)
    if (sysstr == "Windows"):
        print('----os' + str(os.getpid()))
        http_server.listen(options.port)
    else:
        http_server.bind(options.port)
        http_server.start(options.worker)

    # [I 150610 10:42:05 process:115] Starting 4 processes
    tornado.ioloop.IOLoop.instance().start()
