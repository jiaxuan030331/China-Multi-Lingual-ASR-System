#coding:utf-8
import sys
sys.path.insert(0, "/data/WhisperLive")
import tornado.web
import tornado.httpserver
import tornado.options
import tornado.ioloop
import platform
import os
import threading
import json
from whisper_live.transcriber_backup import WhisperModel
import multiprocessing

from tornado.options import options , define
define("port",default=8001,help="跑在8001",type=int)
define("worker",default=2,help="跑在8001",type=int)
define("model_path",default='/root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3',help="跑在8001",type=str)


import time
import inspect
from whisper_live.transcriber_backup import WhisperModel
print("✅ WhisperModel 使用源:", inspect.getfile(WhisperModel))

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
            print('init cls.models ---------')
            cls._model_id = 0 if 'model_id' not in kwargs else kwargs['model_id']
            print(model_size_or_path)
            print(cls._model_id)
            cls._models = cls.create_model(device, model_size_or_path, device_index)
            cls.task = "transcribe"
            cls.initial_prompt = None
            cls.vad_parameters = {"threshold": 0.5}
            cls.use_vad = True
        # 实例存在直接返回
        return cls._instance

    def get_models(self):
        return self._models

    def get_model_id(self):
        return self._model_id

    def create_model(self, device, model_size_or_path, device_index=0):
        transcriber = WhisperModel(
            model_size_or_path,
            device=device,
            device_index = device_index,
            compute_type="int8" if device == "cpu" else "float16",
            local_files_only=False,
        )
        return transcriber
    def transcribe_audio(self, input_sample, language):
        Singleton.SINGLE_MODEL_LOCK.acquire()
        result, info = self._models.transcribe(
            input_sample,
            initial_prompt=self.initial_prompt,
            language=language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None)
        Singleton.SINGLE_MODEL_LOCK.release()
        return result, info

    def __init__(self,*args, **kwargs):
        pass


class MainHandler(tornado.web.RequestHandler):
    def initialize(self, infos):
        #self.quedict = manager.Queue()
        try:
            if infos.empty() == False:
                mode_info = infos.get(block=False)
                single1 = Singleton(device_index=mode_info['device_index'], model_path=mode_info['model_path'],
                                    device=mode_info['device'], model_id=mode_info['model_id'])
                model_id = single1.get_model_id()
                if model_id != mode_info['model_id']:
                    infos.put(mode_info)
                    print(mode_info)
                print('created')
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
            data = self.request.body

            if not data:
                self.write('{ "status" : -2, "message" : "wave data is null!" }')
            else:
               
            
                result, info = self.model.transcribe_audio(data, language)
                json_return = dict()
                json_return['result'] = result
                #json_return['info'] = info
                json_return['status'] = 0
                #if 'vad_options' in json_return['info']:
                #    del json_return['info']['vad_options']                
                return_str = json.dumps(json_return, ensure_ascii=False)
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

def init_queue(queue_info, workers, model_path, device):
    for i in range(workers):
        model_info1 = {'device_index': 0, 'device': device, 'model_path': model_path, 'model_id': i+1}
        queue_info.put(model_info1)

if __name__ == "__main__":
    tornado.options.parse_command_line()
    sysstr = platform.system()
    if (sysstr == "Windows"):
        multiprocessing.freeze_support()

    manager = multiprocessing.Manager()

    queue_info = manager.Queue()
    #data = que_input.get(block=False)
    init_queue(queue_info, 5, options.model_path, 'cuda')

    app = tornado.web.Application(
        handlers=[(r"/",MainHandler, {"infos":queue_info})],
        debug = False
    )
    http_server = tornado.httpserver.HTTPServer(app)
    if (sysstr == "Windows"):
        print('----os' + str(os.getpid()))
        http_server.listen(options.port)
    else:
        http_server.bind(options.port)
        http_server.start(options.worker)
    print('Server Start ' + str(options.port))
    # [I 150610 10:42:05 process:115] Starting 4 processes
    tornado.ioloop.IOLoop.instance().start()
