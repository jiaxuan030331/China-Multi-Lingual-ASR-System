import os
import time
import threading
import json
import functools
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from enum import Enum
from typing import List, Optional
import torch
import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from whisper_live import utils
from collections import namedtuple
from whisper_live.model_db import ModelService
import requests
import configparser

# logging.basicConfig(level=logging.INFO)
utils.init_log('./logs/whiser.log', level=logging.INFO, stdout=True, backup=30)


class ClientManager:
    def __init__(self, max_clients=10, max_connection_time=800):
        self.clients = {}
        self.start_times = {}
        self.model_idx = 0
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.data_recv_thread = ThreadPoolExecutor(max_workers=max_clients)

    def get_http_url(self):
        cf = configparser.ConfigParser()
        cf.read("./conf/config.ini")
        http_urls = cf.get("model", "http_url")
        http_urls = http_urls.split(',')
        http_url = http_urls[self.model_idx%(len(http_urls))]
        self.model_idx += 1
        print(http_url)
        return http_url

    def add_client(self, websocket, client):
        self.clients[websocket] = client
        self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        if websocket in self.clients:
            return self.clients[websocket]
        return False

    def remove_client(self, websocket):
        client = self.clients.pop(websocket, None)
        if client:
            client.cleanup()
        self.start_times.pop(websocket, None)

    def get_wait_time(self):
        wait_time = None
        for start_time in self.start_times.values():
            current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        if len(self.clients) >= self.max_clients:
            wait_time = self.get_wait_time()
            response = {"uid": options["uid"], "code": StatusCode.OK,"status": StatusCode.STATUS_WAIT, "message": wait_time}
            websocket.send(json.dumps(response))
            return True
        return False

    def is_client_timeout(self, websocket):
        elapsed_time = time.time() - self.start_times[websocket]
        print('is_client_timeout :' + str(elapsed_time))
        if elapsed_time >= self.max_connection_time:
            self.clients[websocket].disconnect()
            logging.warning(f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime.")
            return True
        return False


class BackendType(Enum):
    FASTER_WHISPER = "faster_whisper"
    TENSORRT = "tensorrt"

    @staticmethod
    def valid_types() -> List[str]:
        return [backend_type.value for backend_type in BackendType]

    @staticmethod
    def is_valid(backend: str) -> bool:
        return backend in BackendType.valid_types()

    def is_faster_whisper(self) -> bool:
        return self == BackendType.FASTER_WHISPER

    def is_tensorrt(self) -> bool:
        return self == BackendType.TENSORRT

class StatusCode:
    OK = 0
    TIMEOUT = 1002
    NoToken = 1003
    Expire = 1004
    NoEnough = 1005
    NoModel = 1006
    NoVoice = 1007
    LongVoice = 1008
    FORMAT_ERR = 1009
    STATUS_RESULT = "RESULT"
    STATUS_DISCONNECT = "DISCONNECT"
    STATUS_SERVER_READY = "SERVER_READY"
    STATUS_LANGUAGE = "LANGUAGE"
    STATUS_NO_VOICE_ACTIVITY = "NO_VOICE_ACTIVITY"
    STATUS_NO_TOKEN = "NO_TOKEN"
    STATUS_TOKEN_EXPIRE = "TOKEN_EXPIRE"
    STATUS_NO_ENOUGH = "NO_ENOUGH"
    STATUS_WAIT = "WAIT"
    STATUS_LONGVOICE = "DATA TOO LONG"
    STATUS_WAV_FORMAT = "AUDIO FORMAT NOT WAV OR PCM"
    STATUS_TIMEOUT = "TIMEOUT"


class TranscriptionServer:
    RATE = 16000

    def __init__(self):
        self.client_manager = ClientManager()
        self.model_manager = None
        self.no_voice_activity_chunks = 0
        self.use_vad = True
        self.single_model = False
        self.num_workers = 1

        self.modelService = ModelService()

    def initialize_client(
            self, websocket, options, user_name,
            user_type, db_service
    ):
       
        
        client = ServeClientFasterWhisper(
            websocket,
            language=options.get("language"),
            task=options.get("task"),
            db_service=db_service,
            client_uid=options.get("uid"),
            initial_prompt=options.get("initial_prompt"),
            vad_parameters=options.get("vad_parameters"),
            use_vad=self.use_vad,
            single_model=self.single_model, 
            model_manager=self.client_manager.data_recv_thread,
            user_type=user_type
        )
        #logging.info("Running faster_whisper backend.")

        if client is None:
            raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")


        path = os.path.join('./data', user_name)
        os.makedirs(path, exist_ok=True)
        client.file_name = os.path.join(path, utils.random_key() + '_' + str(options.get('name')) + '.wav')
        logging.info("Running Begin : %s , %s" % (user_name,
                                                   client.file_name))
        client.lock.acquire()
        client.http_url = self.client_manager.get_http_url()
        client.lock.release()
        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        frame_data = websocket.recv()
        print('get audio from websocket ------------------')
        if frame_data == b"END_OF_AUDIO":
            print('get audio from websocket ------------------ END')
            return False
        # raw_data = np.frombuffer(buffer=frame_data, dtype=np.int16)
        # raw_data = raw_data.astype(np.float32) / 32768.0

        return np.frombuffer(frame_data, dtype=np.float32)

    def get_audio_bytes_from_websocket(self, websocket):
        frame_data = websocket.recv(timeout=60)
        print('get audio from websocket ------------------')
        if frame_data == b"END_OF_AUDIO" or frame_data == 'END_OF_AUDIO':
            #logging.info("End signal received ")
            print('get audio from websocket ------------------ END')
            return False, None
        try:
            raw_data = np.frombuffer(buffer=frame_data, dtype=np.int16)
            #logging.info(f"✅ 接收音频帧大小: {len(frame_data)} 字节, 采样点: {raw_data.shape[0]}")
            raw_data = raw_data.astype(np.float32) / 32768.0
        except Exception as e:
            logging.error(f"Error get_audio_bytes_from_websocket: {str(e)}")
            client = self.client_manager.get_client(websocket)
            websocket.send(json.dumps({
                "uid": client.client_uid,
                "code": StatusCode.FORMAT_ERR,
                "status":StatusCode.STATUS_WAV_FORMAT,
                "message": "Audio format is not wav or pcm!"
            }))
            return False, None

        return raw_data, frame_data
        # return np.frombuffer(frame_data, dtype=np.float32)

    def send_message(self, websocket, response):
        websocket.send(response)
    def check_token(self, token, uid):
        token_result, user_type = self.modelService.check_token(token)
        response = None
        if token_result == -1:
            logging.info('token is not exist: %s' % token)
            response = {"uid": uid, "code": StatusCode.NoToken, "status": StatusCode.STATUS_NO_TOKEN,
                        "message": "token is not exist！"}
        elif token_result == -2:
            logging.info('token is expire: %s' % token)
            response = {"uid": uid, "code": StatusCode.Expire, "status": StatusCode.STATUS_TOKEN_EXPIRE,
                        "message": "token is expire！"}
        elif token_result == -3:
            logging.info('Not enough usage times : %s' % token)
            response = {"uid": uid, "code": StatusCode.NoEnough, "status": StatusCode.STATUS_NO_ENOUGH,
                        "message": "Not enough usage times ！"}
        else:
            print('ok')
        return response, user_type

    def handle_new_connection(self, websocket, faster_whisper_custom_model_path,
                              whisper_tensorrt_path, trt_multilingual):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            logging.info(options)
            options = json.loads(options)
            self.use_vad = options.get('use_vad')

            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False  # Indicates that the connection should not continue

            #  add some thing to do
            token = options.get('token')
            version = options.get('version')
            model = options.get('model')
            # file_name = options.get('name')
            response = None
            user_type = dict()
            if version is None:
                logging.info('Please upgrade the API')
                response = {"uid": options["uid"], "code":1002,"status": "Version Error", "message": "api version is error！"}
            else:
                response, user_type = self.check_token(token, options["uid"])
            if response:
                self.send_message(websocket, json.dumps(response))
                return False
            '''
            response = None
            user_type = dict()

            if version is None:
                logging.info('Please upgrade the API')
                response = {"uid": options["uid"], "code":1002,"status": "Version Error", "message": "api version is error！"}
                self.send_message(websocket, json.dumps(response))
                return False

            # ✅ MOCK 掉 token 校验逻辑
            user_type = {
                'user_id': 'debug_user',
                'type_name': 'developer',
                'token': token or 'mock_token_123'
            }
            logging.info("Token bypassed. Mock user_type injected.")
            '''
            user_name = str(user_type['user_id']) + "_" + user_type['type_name']
            #########################

            self.initialize_client(websocket, options, user_name, user_type, self.modelService)
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        client = self.client_manager.get_client(websocket)
        try:
            frame_np, frame_data = self.get_audio_bytes_from_websocket(websocket)
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            websocket.send(json.dumps({
                "uid": client.client_uid,
                "code": StatusCode.TIMEOUT,
                "status": StatusCode.STATUS_TIMEOUT,
                "message": "Recv Time out!"
            }))
            return False

        if frame_np is False:
            if self.backend.is_tensorrt():
                client.set_eos(True)
            client.update_frames_size()
            utils.save_wav(client.file_name, frame_data, True)

            whileCount = 600
            while whileCount > 0:
                if client.canExit is True:
                    print('get audio from websocket ------END_OF_AUDIO')
                    break
                time.sleep(0.05)
                whileCount -= 1
            return False

        utils.save_wav(client.file_name, frame_data)
        client.add_frames(frame_np)
        return True

    def notify_no_voice_activity(self, websocket):
        client = self.client_manager.get_client(websocket)
        if client:
            client.websocket.send(json.dumps({
                "uid": client.client_uid,
                "code": StatusCode.NoVoice,
                "status":StatusCode.STATUS_NO_VOICE_ACTIVITY,
                "message": "NO_VOICE_ACTIVITY"
            }))

    def recv_audio(self,
                   websocket,
                   backend: BackendType = BackendType.FASTER_WHISPER,
                   faster_whisper_custom_model_path=None,
                   whisper_tensorrt_path=None,
                   trt_multilingual=False):
        self.backend = backend

        t1 = time.time()
        if not self.handle_new_connection(websocket, faster_whisper_custom_model_path,
                                          whisper_tensorrt_path, trt_multilingual):
            return

        logging.info('connecting ----------- ok :%s' % str(time.time() - t1))
        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                time.sleep(0.1)
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def run(self,
            host,
            port=9090,
            backend="tensorrt",
            faster_whisper_custom_model_path=None,
            whisper_tensorrt_path=None,
            trt_multilingual=False,
            single_model=False):
        if faster_whisper_custom_model_path is not None and not os.path.exists(faster_whisper_custom_model_path):
            raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path.")
        if whisper_tensorrt_path is not None and not os.path.exists(whisper_tensorrt_path):
            raise ValueError(f"TensorRT model '{whisper_tensorrt_path}' is not a valid path.")
        if single_model:
            if faster_whisper_custom_model_path or whisper_tensorrt_path:
                logging.info("Custom model option was provided. Switching to single model mode.")
                self.single_model = True
            else:
                logging.info("Single model mode currently only works with custom models.")
        if not BackendType.is_valid(backend):
            raise ValueError(f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")

        # self.model_manager = ModelManager(faster_whisper_custom_model_path)
        with serve(
                functools.partial(
                    self.recv_audio,
                    backend=BackendType(backend),
                    faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                    whisper_tensorrt_path=whisper_tensorrt_path,
                    trt_multilingual=trt_multilingual
                ),
                host,
                port
        ) as server:
            server.serve_forever()


    def voice_activity(self, websocket, frame_np):
        if not self.vad_detector(frame_np):
            return False
        return True

    def cleanup(self, websocket):
        if self.client_manager.get_client(websocket):
            self.client_manager.remove_client(websocket)


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    def __init__(self, client_uid, websocket):
        self.client_uid = client_uid
        self.websocket = websocket
        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_np_size = 0
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ''
        self.prev_out = ''
        self.t_start = None
        self.exit = False
        self.canExit = False
        self.same_output_threshold = 0
        self.show_prev_out_thresh = 5
        self.add_pause_thresh = 3
        self.transcript = []
        self.send_last_n_segments = 10
        self.lock = threading.Lock()
        self.http_url = 'http://127.0.0.1:8001'
        self.user_type = None
        self.file_name = None
        self.db_service = None

    def update_frames_size(self):
        if self.frames_np is not None:
            self.frames_np_size = len(self.frames_np)

    def init_frames_size(self):
        self.frames_np_size = 0

    def speech_to_text(self):
        raise NotImplementedError

    def transcribe_audio(self):
        raise NotImplementedError

    def handle_transcription_output(self):
        raise NotImplementedError

    def add_frames(self, frame_np):
        self.canExit = False
        self.init_frames_size()
        self.lock.acquire()
        if self.frames_np is not None and self.frames_np.shape[0] > 60 * self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30 * self.RATE):]
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset
            self.voice_too_long()

        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

        print('add_frames ----- length ' + str(len(self.frames_np)))
        self.lock.release()

    # 去掉过多的静音
    def clip_audio_if_no_valid_segment(self):
        if self.frames_np[int((self.timestamp_offset - self.frames_offset) * self.RATE):].shape[0] > 60 * self.RATE:
            duration = self.frames_np.shape[0] / self.RATE
            self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
        input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration, len(self.frames_np)

    def prepare_segments(self, last_segment=None):
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def get_audio_chunk_duration(self, input_bytes):
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments, is_end=False):
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "code": StatusCode.OK,
                    "status": StatusCode.STATUS_RESULT,
                    "segments": segments,
                    "is_end": is_end
                })
            )
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def disconnect(self):
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "code":StatusCode.OK,
            "status":StatusCode.STATUS_DISCONNECT,
            "message": self.DISCONNECT
        }))

    def voice_too_long(self):
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "code":StatusCode.LongVoice,
            "status":StatusCode.STATUS_LONGVOICE,
            "message": "The length of voice data exceeds the limit!"
        }))

    def cleanup(self):
        logging.info("Cleaning up.")
        self.exit = True


class ServeClientFasterWhisper(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(self, websocket, task="transcribe", db_service=None, language=None, client_uid=None, model="small.en",
                 initial_prompt=None, vad_parameters=None, use_vad=True, single_model=False, model_manager=None,
                 user_type=None):
        super().__init__(client_uid, websocket)
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3",
        ]
        self.modelsManager = model_manager
        # if not os.path.exists(model):
        #     self.model_size_or_path = self.check_valid_model(model)
        # else:
        #     self.model_size_or_path = model
        self.language = language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"threshold": 0.5}
        self.no_speech_thresh = 0.85
        self.user_type = user_type
        print('token =====' + self.user_type['token'])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_vad = use_vad
        self.db_service = db_service
        model_manager.submit(self.speech_to_text)
        # self.trans_thread = threading.Thread(target=self.speech_to_text)
        # self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "code": StatusCode.OK,
                    "status": StatusCode.STATUS_SERVER_READY,
                    "backend": "vvVoiceServer"
                }
            )
        )

    def check_valid_model(self, model_size):
        if model_size not in self.model_sizes:
            self.websocket.send(
                json.dumps(
                    {
                        "uid": self.client_uid,
                        "code": StatusCode.NoModel,
                        "status": "ERROR",
                        "message": f"Invalid model size {model_size}. Available choices: {self.model_sizes}"
                    }
                )
            )
            return None
        return model_size

    def set_language(self, info):
        if info.language_probability > 0.5:
            self.language = info.language
            logging.info(f"Detected language {self.language} with probability {info.language_probability}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "code":StatusCode.OK, "status":StatusCode.STATUS_LANGUAGE,"language": self.language, "language_prob": info.language_probability}))

    def transcribe_audio(self, input_sample):
        input_data = dict()
        t = time.time()
        input_data['name'] = str(int(round(t * 1000)))
        if self.language:
            input_data['language'] = self.language
        if self.initial_prompt:
            input_data['initial_prompt'] = self.initial_prompt
        try:
            logging.info(self.http_url + ' : ')
            res = requests.post(self.http_url, headers=input_data, data=input_sample.tobytes())
            #print(res)
            results = json.loads(res.text)
            # print(results['status'])
            if 'status' in results and results['status'] == 0:
                result = results.get('result')
                info = results.get('info')
                if self.language is None and info is not None:
                    Tanscr = namedtuple('TranscriptionInfo', info.keys())
                    print('TranscriptionInfo : ' + str(Tanscr))
                    self.set_language(Tanscr(**info))
                return result, 0
            else:
                logging.error(f"Error No Result: {str(results)}")
                return None, 0
        except Exception as e:
            logging.error(f"Error transcribe_audio: {str(e)}")
            return None, 400

    def transcribe_audio2(self, input_sample):
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
        logging.info("transcribe_audio : " + str(time.time()))
        result, info = self.transcriber.transcribe(
            input_sample,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None)
        logging.info("transcribe_audio result : " + str(time.time()))
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()

        if self.language is None and info is not None:
            self.set_language(info)
        return result

    def get_previous_output(self):
        segments = []
        if self.t_start is None:
            self.t_start = time.time()
        if time.time() - self.t_start < self.show_prev_out_thresh:
            segments = self.prepare_segments()

        if len(self.text) and self.text[-1] != '':
            if time.time() - self.t_start > self.add_pause_thresh:
                self.text.append('')
        return segments

    def handle_transcription_output(self, result, duration, is_end=False):
        segments = []
        # new_result = []
        if len(result):
            self.t_start = None
            print('handle_transcription_output result is not empty: ' + str(result))
            last_segment = self.update_segments(result, duration)
            #print('last_segment : ' + str(last_segment))
            segments = self.prepare_segments(last_segment)
            
        else:
            print('handle_transcription_output result is empty: ')
            segments = self.get_previous_output()

        if len(segments) or is_end:
            # segments.append(new_result)
            print('handle_transcription_output sending segments: ' + str(segments))
            self.send_transcription_to_client(segments, is_end)
        else:
            print('handle_transcription_output not sending ')
        return segments

    def speech_to_text(self):
        print(self.client_uid + ":"+self.user_type['token'])
        is_end = False
        result_none = 0
        no_frames = 0
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread : " + self.client_uid)
                break

            if self.frames_np is None:
                time.sleep(0.01)
                no_frames +=1
                if no_frames > 7777:
                    self.exit = True
                continue
            no_frames = 0
            self.clip_audio_if_no_valid_segment()
            input_bytes, duration, all_length = self.get_audio_chunk_for_processing()
            # tmp = len(input_bytes)
            print('--------------------------' + str(self.timestamp_offset) + '; ' + str(self.frames_offset))
            print(str(duration) + ' --- ' + str(all_length))
            if all_length == self.frames_np_size and self.frames_np_size > 0:
                is_end = True
            if duration < 0.5:

                if is_end is True:
                    self.canExit = True
                    self.exit = True
                continue
            try:
                t1 = time.time()
                new_result = []
                if duration < 0.5:
                    if is_end is False:
                        time.sleep(0.01)
                        continue
                else:
                    input_sample = input_bytes.copy()
                    result, code = self.transcribe_audio(input_sample)
                    
                    print('transcribe_audio : ' + str(time.time() - t1))
                    if result is None: #or self.language is None:

                        if code == 0:
                            if result_none % 100 > 10:
                                self.timestamp_offset += 0.5
                                result_none = 0
                            else:
                                result_none += 1
                        else:
                            if result_none > 500:
                                self.canExit = True
                                self.exit = True
                                break
                            else:
                                result_none += 100

                        time.sleep(0.2)
                        continue
                    result_none = 0

                    for item in result:
                            Segment_info = namedtuple('Segment', item.keys())
                            new_result.append(Segment_info(**item))
                    #print('new_result : ' + str(new_result))

                    
               
                all_result = self.handle_transcription_output(new_result, duration, is_end)

                if is_end:
                    # save log
                    # result_str, sen_id, user_type
                    result_str = json.dumps(all_result, ensure_ascii=False)
                    iret = self.db_service.add_result(result_str, self.file_name, self.user_type)
                    logging.info("%s , %s, code=%s, %s, %s" % (str(self.user_type.get('type_name')),
                                                              str(self.user_type.get('user_id')),
                                                              str(iret), self.file_name, result_str))
                    self.canExit = True
                    time.sleep(0.1)
                    self.exit = True
                else:
                    self.canExit = False
            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def format_segment(self, start, end, text):
        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text
        }

    def update_segments(self, segments, duration):
        offset = None
        self.current_out = ''
        if len(segments) > 1:
            for i, s in enumerate(segments[:-1]):
                text_ = s.text
                self.text.append(text_)
                start, end = self.timestamp_offset + s.start, self.timestamp_offset + min(duration, s.end)

                if start >= end:
                    continue
                if s.no_speech_prob > self.no_speech_thresh:
                    continue
                temp_result = self.format_segment(start, end, text_)
                self.transcript.append(temp_result)
                # new_result.append(temp_result)

                offset = min(duration, s.end)

        self.current_out += segments[-1].text
        last_segment = self.format_segment(
            self.timestamp_offset + segments[-1].start,
            self.timestamp_offset + min(duration, segments[-1].end),
            self.current_out
        )
        # new_result.append(last_segment)
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_threshold += 1
        else:
            self.same_output_threshold = 0

        if self.same_output_threshold > 5:
            if not len(self.text) or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                self.transcript.append(self.format_segment(
                    self.timestamp_offset,
                    self.timestamp_offset + duration,
                    self.current_out
                ))
            self.current_out = ''
            offset = duration
            self.same_output_threshold = 0
            last_segment = None
        else:
            self.prev_out = self.current_out

        if offset is not None:
            self.timestamp_offset += offset

        return last_segment


