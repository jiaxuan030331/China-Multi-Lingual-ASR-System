
# 🎧 Kimi Audio ASR 推理服务

本项目基于 [MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) 开源语音识别模型，封装为 FastAPI 服务，支持上传 .wav / .mp3 音频文件并返回文本转写结果。
Kimi项目原定逻辑为发送文件路径，prompt等完整请求得到模型回复的多模态模型，该项目修改了部分源代码，通过模型请求实现直接通过裸数据（PCM,numpy array)进行ASR

适用于Websocket实时识别，语音识别推理测试、API 部署演示、上流输入系统集成等场景。

---

## 📁 项目结构

```
kimi_deployment/
├── app/
│   ├── fastapi_api.py       # FastAPI 主接口文件
│   ├── load_model.py        # 封装模型加载
│   ├── transcribe.py        # 封装识别逻辑
|   |-- server.py            # websocket客户端
├── test/
│   ├── test_transcribe.py   # 测试脚本
│   └── test_audios/         # 示例音频
├── kimi_cloned/             # 克隆的 Kimi-Audio 模型代码
|-- Whisperlive
├── run_kimi_server.sh       # 启动 FastAPI 的脚本
|-- run_server.py            # 启动server.py 提供websocket服务
├── requirements.txt
└── setup.py

```

---

## ✨ Kimi环境依赖安装

建议使用 Conda 虚拟环境，Python ≥ 3.10：

```bash
conda create -n kimi python=3.10 -y
conda activate kimi
pip install -r requirements.txt
```


### 额外依赖 (必须安装)

```bash
# 高性能应用需要 flash-attn 2.x
pip install flash-attn --no-build-isolation

# CUDA kernel 构建需要 Ninja
pip install ninja
```


## Websocket 服务使用：

项目结构：前端录入音频 -> 同步至websocket客户端 -> 发送PCM数据请求至后端kimi fastapi的transcribe-websocket接口 -> 调用kimi模型

注： 

启动步骤：

1.连接项目根目录，启动Kimi虚拟环境
```bash
cd /root/ASR_TTS_improvement/kimi_deployment
conda activate kimi
```

2.拉起fastapi提供的后端http api接口(模型路径：/root/.cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b)
```bash
./run_kimi_server.sh
```
启动参数：
KIMI_PORT: 启动端口，默认8000
KIMI_NUM_WORKERS: 启动的进程数，默认1（当前显存仅支持单进程）
KIMI_MODEL_PATH: 模型路径，默认'/root/.cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b'
KIMI_TORCH_DTYPE: 加载数据类型，默认'bfloat16'
KIMI_DEVICE,KIMI_DEVICE_INDEX: 加载的设备，默认'cuda',0

请求示例：
```python
requests.post("http://127.0.0.1:8000/transcribe_websocket", headers=headers, data=pcm_bytes)#headers为传入的参数
```

3.拉起websocket服务
```bash
python run_server.py
```
启动参数：
--port: 启动端口，默认9091
--omp_num_threads: openmp线程数，默认1
--no_single_model:是否为每个连接创建实例
监听的后端接口可在./conf/config.ini修改

请求示例：
```python
ws = websocket.create_connection("ws://127.0.0.1:9091")
ws.send(json.dumps(options))#转录参数
ws.send(pcm_bytes, opcode=websocket.ABNF.OPCODE_BINARY)
ws.send(b"END_OF_AUDIO")
```

转录参数：
prompt：默认'请转写音频'
VAD配置

注：第三步需要验证token,若后端测试可以注释server.py 249-258行，并去掉258-275行的注释



## Fastapi 启动服务

```bash
bash kimi_deployment/run_kimi_server.sh
```

服务默认启动于:

```
http://localhost:8000
```

首次启动时会自动从 Huggingface 下载 Kimi-Audio 预训练模型。

---

## 💪 API 使用法

### Swagger UI 互动页面：

打开浏览器：

```
http://localhost:8000/docs
```

可以直接上传音频进行测试，结果展示在页面下方。

### 使用 CURL 接口请求

```bash
curl -X POST http://localhost:8000/transcribe   -F "audio_file=@kimi_deployment/test/test_audios/asr_example.wav"
```

---


## 🔧 单元测试

支持自动运行脚本进行输入测试：

```bash
python kimi_deployment/test/test_transcribe.py
```

---

\
