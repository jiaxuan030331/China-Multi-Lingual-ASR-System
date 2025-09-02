import os
import logging 
import tqdm
import torch
from loguru import logger
from huggingface_hub import cached_assets_path
from transformers import AutoModelForCausalLM
import torchaudio
from kimia_infer.models.detokenizer import get_audio_detokenizer
from .prompt_manager import KimiAPromptManager
from kimia_infer.utils.sampler import KimiASampler
from huggingface_hub import snapshot_download
import numpy as np
from kimia_infer.utils.data import KimiAContent
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from transformers import AutoTokenizer
from kimia_infer.models.tokenizer.glm4.speech_tokenizer.utils import extract_speech_token
import torch
import librosa


class KimiAudio(object):
    def __init__(
        self,
        model_path: str,
        load_detokenizer: bool = True,
        device: str = "cuda",
        device_index: int = 0,
        torch_dtype: str = "bfloat16"
    ):

        
        
        if os.path.exists(model_path):
            print("Loading model from local path.")
            cache_path = model_path
        else:
            cache_path = snapshot_download(model_path)

        logger.info(f"Looking for resources in {cache_path}")
        logger.info(f"Loading whisper model")

        # 安全解析 device 和 dtype
        dtype = getattr(torch, torch_dtype, torch.float16)
        device_str = f"{device}:{device_index}" if device == "cuda" else device
        torch_device = torch.device(device_str)

        logger.info(f"Using device: {torch_device}, dtype: {dtype}")
        
        self.alm = AutoModelForCausalLM.from_pretrained(
            cache_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        self.alm = self.alm.to(torch_device)
        
        model_config = self.alm.config
        
        self.kimia_token_offset = model_config.kimia_token_offset
       
        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path,
            kimia_token_offset=self.kimia_token_offset
        )
        
        if load_detokenizer:
            logger.info(f"Loading detokenizer")
            self.detokenizer = get_audio_detokenizer(cache_path)
        else:
            self.detokenizer = None
        
        self.extra_tokens = self.prompt_manager.extra_tokens
        self.kimia_text_audiodelaytokens = 6
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]  
         
    '''
    def __init__(self, model_path: str, load_detokenizer: bool = True):
        logger.info(f"Loading kimi-audio main model")

        if os.path.exists(model_path):
            # local path
            cache_path = model_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_path)
    
        logger.info(f"Looking for resources in {cache_path}")
        logger.info(f"Loading whisper model")
        self.alm = AutoModelForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.alm = self.alm.to(torch.cuda.current_device())

        model_config = self.alm.config
        self.kimia_token_offset = model_config.kimia_token_offset

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path, kimia_token_offset=self.kimia_token_offset
        )

        if load_detokenizer:
            logger.info(f"Loading detokenizer")
            # need to compile extension moudules for the first time, it may take several minutes.
            self.detokenizer = get_audio_detokenizer(cache_path)
        else:
            # in this case, you're not allowed to generate audio(wav)
            self.detokenizer = None

        self.extra_tokens = self.prompt_manager.extra_tokens
        self.kimia_text_audiodelaytokens = 6
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]
        '''
    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,  # input audio tokens
        text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        output_type: str = "text",
    ):

        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        text_stream_is_finished = False
        previous_audio_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        text_previous_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
            )
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1

        valid_text_length = 0
        valid_audio_length = 0

        for i in tqdm.tqdm(
            range(max_new_tokens), desc="Generating tokens", disable=False
        ):
            audio_logits, text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            # Sample text token using the sampler
            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )

            # Sample audio token using the sampler
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )

            if text_stream_is_finished:
                next_token_text.fill_(self.extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
            else:
                valid_text_length += 1

            text_previous_tokens[i : i + 1] = next_token_text

            if i < self.kimia_text_audiodelaytokens:
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1

            previous_audio_tokens[i : i + 1] = next_audio_token

            audio_stream_is_finished = next_audio_token.item() in self.eod_ids

            if (
                output_type == "text"
                and text_stream_is_finished
                or output_type == "both"
                and audio_stream_is_finished
            ):
                return_text_tokens = (
                    text_previous_tokens[:valid_text_length]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return_audio_tokens = (
                    previous_audio_tokens[
                        self.kimia_text_audiodelaytokens : valid_audio_length
                        + self.kimia_text_audiodelaytokens
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return return_audio_tokens, return_text_tokens
            else:
                decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                decoder_input_text_ids = next_token_text.unsqueeze(1)

                decoder_position_ids = (
                    torch.zeros(1, 1, device=torch.cuda.current_device())
                    .fill_(last_position_id + 1)
                    .long()
                    .view(1, 1)
                )
                last_position_id += 1

                decoder_input_whisper_feature = None
                decoder_is_continuous_mask = None

        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return_audio_tokens = (
            previous_audio_tokens[
                self.kimia_text_audiodelaytokens : valid_audio_length
                + self.kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        return return_audio_tokens, return_text_tokens

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
    ):
        ## TODO: 需要一个check函数，检查输入的history格式是否合法
        ## 比如，对于ASR任务，一定是: text-instruction/audio-instruction + audio-content, 我理解content和instruction是不能换位置的
        ## assistant前必须有user等等，我觉得最好做一下check

        assert output_type in ["text", "both"]

        history = self.prompt_manager.get_prompt(chats, output_type=output_type)

        audio_input_ids, text_input_ids, is_continuous_mask = history.to_tensor()
        audio_features = history.continuous_feature

        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                max_new_tokens = 7500 - audio_input_ids.shape[1]

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            output_type=output_type,
        )

        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]  #  filter out the illegal tokens

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset

        generated_text_tokens = [
            t for t in generated_text_tokens if t < self.kimia_token_offset
        ]
        generated_text = self.detokenize_text(generated_text_tokens)
        if self.detokenizer is not None and output_type == "both":
            generated_wav = self.detokenize_audio(generated_wav_tokens)
        else:
            generated_wav = None

        return generated_wav, generated_text
    
    def generate_from_waveform(
        self, 
        waveform: torch.Tensor, 
        sr: int = 16000, 
        prompt: str = "Please transcribe the audio: ",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1
    ):
        """
        简化版语音转写（ASR）接口：
        - 直接支持 waveform 输入，而不是文件路径
        - 自动完成 audio token、text token、Whisper 特征提取
        - 不依赖 message 构造，直接创建 KimiAContent 实例
        - 自动生成 assistant 起始 token
        """

        ### -------- Step 1: 音频处理（离散 token + whisper 特征） -------- ###

        # 1.1 将 waveform 封装为 batch 格式
        glm_audio = torch.tensor(waveform).unsqueeze(0)
        glm_audio_info = (glm_audio, sr)

        # 1.2 通过 GLM 音频 tokenizer 提取 audio token（类似音频编码）
        glm_audio_tokens = extract_speech_token(
            self.prompt_manager.audio_tokenizer.whisper_model,
            self.prompt_manager.audio_tokenizer.feature_extractor,
            [glm_audio_info]
        )[0]

        # 1.3 转为 tensor 并加 kimia_token_offset（防止和 text 冲突）
        glm_audio_tokens = torch.tensor(glm_audio_tokens).unsqueeze(0)
        wav_tokens = glm_audio_tokens + self.prompt_manager.kimia_token_offset
        speech_tokens = wav_tokens.squeeze(0).cpu().numpy().tolist()

        # 1.4 编码文本 prompt 为 text_token_ids
        text_tokens = self.prompt_manager.text_tokenizer.encode(prompt, bos=False, eos=False)

        # 1.5 提取 Whisper 连续特征（float 特征，用于上下文引导）
        wav_tensor = torch.tensor(waveform).unsqueeze(0)[:, :].to(torch.cuda.current_device())
        whisper_feature = self.prompt_manager.whisper_model.tokenize_waveform(wav_tensor)
        whisper_feature = whisper_feature.reshape(
            whisper_feature.shape[0],
            whisper_feature.shape[1] // 4,
            whisper_feature.shape[2] * 4,
        )

        ### -------- Step 2: 构造 KimiAContent 实例 -------- ###

        msg = KimiAContent()

        # 添加 role 起始标记（user）
        msg.audio_append(self.prompt_manager.extra_tokens.kimia_user_msg_start)
        msg.text_append(self.prompt_manager.extra_tokens.kimia_text_blank)

        # 添加文本 prompt → text token
        msg.text_extend(text_tokens)
        msg.audio_extend([self.prompt_manager.extra_tokens.kimia_text_blank] * len(text_tokens))

        # 添加音频 token（带 media_begin/media_end 控制符）
        msg.audio_append(self.prompt_manager.extra_tokens.media_begin)
        msg.audio_extend(speech_tokens, is_continuous=True)
        msg.audio_append(self.prompt_manager.extra_tokens.media_end)
        msg.text_extend([self.prompt_manager.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2))

        # 添加切换标记，代表“语音内容结束，可生成回答”
        msg.audio_append(self.prompt_manager.extra_tokens.kimia_speech_ct_id)
        msg.text_append(self.prompt_manager.extra_tokens.kimia_text_blank)

        # 添加消息终止 token
        msg.audio_append(self.prompt_manager.extra_tokens.msg_end)
        msg.text_append(self.prompt_manager.extra_tokens.kimia_text_blank)

        # 添加 Whisper 特征
        msg.continuous_feature.append(whisper_feature)

        # 添加 assistant 响应开始（表示模型开始生成）
        msg_assist = self.prompt_manager.tokenize_message(
            message={"role": "assistant", "message_type": None},
            tokenize_role=True,
            has_ct_token=False,
            has_msg_end_token=False,
        )
        msg.merge(msg_assist)
        assert msg.is_valid()

        ### -------- Step 3: 模型推理 -------- ###

        # 将内容结构化为 tensor，供模型使用
        audio_input_ids, text_input_ids, is_continuous_mask = msg.to_tensor()
        audio_features = [f.to(torch.cuda.current_device()) for f in msg.continuous_feature]

        # 自动设置生成长度限制
        
        max_new_tokens = 7500 - audio_input_ids.shape[1]

        # 进入模型核心解码逻辑（逐步生成 text token）
        _, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids.to(torch.cuda.current_device()),
            text_input_ids=text_input_ids.to(torch.cuda.current_device()),
            max_new_tokens=max_new_tokens,
            is_continuous_mask=is_continuous_mask.to(torch.cuda.current_device()),
            continous_feature=audio_features,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size, 
            output_type='text',
        )

        # 去除非法 token（如 audio token 混入）
        generated_text_tokens = [t for t in generated_text_tokens if t < self.kimia_token_offset]

        # 解码为字符串
        generated_text = self.detokenize_text(generated_text_tokens)

        return generated_text
        
    def detokenize_audio(self, audio_tokens):
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states()
        chunk_size = 30  # hard-coded right now
        first_chunk_size = 30
        cache_speech_collection = []
        audio_tokens = audio_tokens.to(torch.cuda.current_device())
        audio_tokens = audio_tokens.long()
        num_audio_tokens = audio_tokens.size(1)
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)
