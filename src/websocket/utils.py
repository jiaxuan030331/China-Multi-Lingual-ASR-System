import os
import sys
import textwrap
import scipy
import ffmpeg
import logging
import logging.handlers
import random
import time
import wave


def clear_screen():
    """Clears the console screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_transcript(text):
    """Prints formatted transcript text."""
    wrapper = textwrap.TextWrapper(width=60)
    for line in wrapper.wrap(text=" ".join(text)):
        print(line)


def format_time(s):
    """Convert seconds (float) to SRT time format."""
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    seconds = int(s % 60)
    milliseconds = int((s - int(s)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def create_srt_file(segments, output_file):
    with open(output_file, 'w', encoding='utf-8') as srt_file:
        segment_number = 1
        for segment in segments:
            start_time = format_time(float(segment['start']))
            end_time = format_time(float(segment['end']))
            text = segment['text']

            srt_file.write(f"{segment_number}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

            segment_number += 1


def resample(file: str, sr: int = 16000):
    """
    # https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/audio.py#L22
    Open an audio file and read as mono waveform, resampling as necessary,
    save the resampled audio

    Args:
        file (str): The audio file to open
        sr (int): The sample rate to resample the audio if necessary

    Returns:
        resampled_file (str): The resampled audio file
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    np_buffer = np.frombuffer(out, dtype=np.int16)

    resampled_file = f"{file.split('.')[0]}_resampled.wav"
    scipy.io.wavfile.write(resampled_file, sr, np_buffer.astype(np.int16))
    return resampled_file


'''

import librosa
import numpy as np

def resample_audio_data(audio_data, orig_sample_rate, target_sample_rate):
    """
    Resample audio data to a target sample rate.
    
    Args:
        audio_data (np.ndarray): Original audio data.
        orig_sample_rate (int or float): Original sample rate of the audio data.
        target_sample_rate (int or float): Target sample rate.
    
    Returns:
        np.ndarray: Resampled audio data.
    """
    if isinstance(orig_sample_rate, np.ndarray):
        if orig_sample_rate.size > 1:
            raise ValueError("orig_sample_rate must be a single scalar value, not an array.")
        orig_sample_rate = orig_sample_rate.item()
    if isinstance(target_sample_rate, np.ndarray):
        if target_sample_rate.size > 1:
            raise ValueError("target_sample_rate must be a single scalar value, not an array.")
        target_sample_rate = target_sample_rate.item()
        
    if orig_sample_rate != target_sample_rate:
        # Convert audio data to floating point
        audio_data = audio_data.astype(np.float32)
        audio_data = librosa.resample(y=audio_data, orig_sr=orig_sample_rate, target_sr=target_sample_rate)
    return audio_data

'''

import librosa
import numpy as np


def resample_audio_data(audio_data, orig_sample_rate, target_sample_rate):
    """
    Resample audio data to a target sample rate.
    
    Args:
        audio_data (np.ndarray): Original audio data.
        orig_sample_rate (int or float): Original sample rate of the audio data.
        target_sample_rate (int or float): Target sample rate.
    
    Returns:
        np.ndarray: Resampled audio data.
    """
    if isinstance(orig_sample_rate, np.ndarray):
        if orig_sample_rate.size > 1:
            raise ValueError("orig_sample_rate must be a single scalar value, not an array.")
        orig_sample_rate = orig_sample_rate.item()
    if isinstance(target_sample_rate, np.ndarray):
        if target_sample_rate.size > 1:
            raise ValueError("target_sample_rate must be a single scalar value, not an array.")
        target_sample_rate = target_sample_rate.item()

    if orig_sample_rate != target_sample_rate:
        # Convert audio data to floating point
        audio_data = audio_data.astype(np.float32)
        audio_data = librosa.resample(y=audio_data, orig_sr=orig_sample_rate, target_sr=target_sample_rate)
    return audio_data


def init_log(
        log_path,
        level=logging.INFO,
        stdout=False,
        when="D",
        backup=7,
        format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"):
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    dir = os.path.dirname(log_path)
    if not dir:
        dir = './'
    if not os.path.isdir(dir):
        os.makedirs(dir)

    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + ".log", when=when, backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.suffix = '%Y-%m-%d'
    logger.addHandler(handler)
    err_handler = logging.handlers.TimedRotatingFileHandler(
        log_path + ".log.wf", when=when, backupCount=backup)
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    handler.suffix = '%Y-%m-%d'
    logger.addHandler(err_handler)
    return None


def random_key():
    random_list = random.sample('zYxwvuTsRQpOnmlKjIHgfeDcbA', 3)
    vkey = str(int(time.time()))
    for v in random_list:
        vkey += v
    return vkey


def save_wave(filename, audio_data):
    with wave.open(filename, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(audio_data)
        f.close

def save_wav(filename, audio_data, is_finish = False):
    if is_finish:
        with open(filename, 'rb') as rf:
            buffer = rf.read()
            save_wave(filename, buffer)
            rf.close()
    else:
        with open(filename, "ab") as f:
            f.write(audio_data)
            f.close()

