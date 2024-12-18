# -*- encoding: utf-8 -*-
import os
import time
import websockets
import ssl
import asyncio
import json
import traceback
from multiprocessing import Process
from queue import Queue
import logging
from dataclasses import dataclass
from typing import List, Optional, Union
import streamlit as st

logging.basicConfig(level=logging.ERROR)

@dataclass
class ASRConfig:
    host: str = "localhost"
    port: int = 10095
    chunk_size: List[int] = None  # Will be initialized in __post_init__
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 0
    chunk_interval: int = 10
    hotword: str = ""
    audio_in: Optional[str] = None
    audio_fs: int = 16000
    send_without_sleep: bool = True
    thread_num: int = 1
    words_max_print: int = 10000
    output_dir: Optional[str] = None
    ssl_enabled: bool = True
    use_itn: bool = True
    mode: str = "2pass"

    def __post_init__(self):
        if self.chunk_size is None:
            self.chunk_size = [5, 10, 5]

class FunASRClient:
    def __init__(self, config: ASRConfig, text_queue=None):
        self.config = config
        self.text_queue = text_queue
        self.voices = Queue()
        self.offline_msg_done = False
        self.websocket = None
        self.connection_retries = 3  # 添加重试次数
        self.connection_timeout = 30  # 添加连接超时时间
        
        if self.config.output_dir and not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

    async def message(self, id):
        global voices, offline_msg_done
        text_print = ""
        text_print_2pass_online = ""
        text_print_2pass_offline = ""
        if self.config.output_dir is not None:
            ibest_writer = open(
                os.path.join(self.config.output_dir, "text.{}".format(id)), "a", encoding="utf-8"
            )
        else:
            ibest_writer = None
        try:
            while True:

                meg = await self.websocket.recv()
                meg = json.loads(meg)
                wav_name = meg.get("wav_name", "demo")
                text = meg["text"]
                timestamp = ""
                offline_msg_done = meg.get("is_final", False)
                if "timestamp" in meg:
                    timestamp = meg["timestamp"]

                if ibest_writer is not None:
                    if timestamp != "":
                        text_write_line = "{}\t{}\t{}\n".format(wav_name, text, timestamp)
                    else:
                        text_write_line = "{}\t{}\n".format(wav_name, text)
                    ibest_writer.write(text_write_line)

                if "mode" not in meg:
                    continue
                if meg["mode"] == "online":
                    text_print += "{}".format(text)
                    text_print = text_print[-self.config.words_max_print :]
                    os.system("clear")
                    print("\rpid" + str(id) + ": " + text_print)
                elif meg["mode"] == "offline":
                    if "stamp_sents" in meg:
                        for segment in meg["stamp_sents"]:
                            start_time = segment["start"] / 1000
                            end_time = segment["end"] / 1000
                            text_seg = segment["text_seg"]
                            punc = segment.get("punc", "")
                            
                            text_print += f"[{start_time:.2f}-{end_time:.2f}] {text_seg}{punc}\n"
                    else:
                        if timestamp != "":
                            text_print += "{} timestamp: {}".format(text, timestamp)
                        else:
                            text_print += "{}".format(text)

                    if self.text_queue:
                        self.text_queue.put(text_print)
                    
                    print("\rpid" + str(id) + ": " + wav_name + ": " + text_print)
                    offline_msg_done = True
                else:
                    if meg["mode"] == "2pass-online":
                        text_print_2pass_online += "{}".format(text)
                        text_print = text_print_2pass_offline + text_print_2pass_online
                    else:
                        text_print_2pass_online = ""
                        text_print = text_print_2pass_offline + "{}".format(text)
                        text_print_2pass_offline += "{}".format(text)
                    text_print = text_print[-self.config.words_max_print :]
                    os.system("clear")
                    print("\rpid" + str(id) + ": " + text_print)
                    # offline_msg_done=True

        except Exception as e:
            print("Exception:", e)
            # traceback.print_exc()
            # await websocket.close()

    async def record_microphone(self):
        is_finished = False
        import pyaudio

        # print("2")
        global voices
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        chunk_size = 60 * self.config.chunk_size[1] / self.config.chunk_interval
        CHUNK = int(RATE / 1000 * chunk_size)

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
        )
        # hotwords
        fst_dict = {}
        hotword_msg = ""
        if self.config.hotword.strip() != "":
            if os.path.exists(self.config.hotword):
                f_scp = open(self.config.hotword)
                hot_lines = f_scp.readlines()
                for line in hot_lines:
                    words = line.strip().split(" ")
                    if len(words) < 2:
                        print("Please checkout format of hotwords")
                        continue
                    try:
                        fst_dict[" ".join(words[:-1])] = int(words[-1])
                    except ValueError:
                        print("Please checkout format of hotwords")
                hotword_msg = json.dumps(fst_dict)
            else:
                hotword_msg = self.config.hotword

        use_itn = True
        if self.config.use_itn == 0:
            use_itn = False

        message = json.dumps(
            {
                "mode": self.config.mode,
                "chunk_size": self.config.chunk_size,
                "chunk_interval": self.config.chunk_interval,
                "encoder_chunk_look_back": self.config.encoder_chunk_look_back,
                "decoder_chunk_look_back": self.config.decoder_chunk_look_back,
                "wav_name": "microphone",
                "is_speaking": True,
                "hotwords": hotword_msg,
                "itn": use_itn,
            }
        )
        # voices.put(message)
        await self.websocket.send(message)
        while True:
            data = stream.read(CHUNK)
            message = data
            # voices.put(message)
            await self.websocket.send(message)
            await asyncio.sleep(0.005)

    async def record_from_scp(self, chunk_begin, chunk_size):
        global voices
        is_finished = False
        if self.config.audio_in.endswith(".scp"):
            f_scp = open(self.config.audio_in)
            wavs = f_scp.readlines()
        else:
            wavs = [self.config.audio_in]

        # hotwords
        fst_dict = {}
        hotword_msg = ""
        if self.config.hotword.strip() != "":
            if os.path.exists(self.config.hotword):
                f_scp = open(self.config.hotword)
                hot_lines = f_scp.readlines()
                for line in hot_lines:
                    words = line.strip().split(" ")
                    if len(words) < 2:
                        print("Please checkout format of hotwords")
                        continue
                    try:
                        fst_dict[" ".join(words[:-1])] = int(words[-1])
                    except ValueError:
                        print("Please checkout format of hotwords")
                hotword_msg = json.dumps(fst_dict)
            else:
                hotword_msg = self.config.hotword
            print(hotword_msg)

        sample_rate = self.config.audio_fs
        wav_format = "pcm"
        use_itn = True
        if self.config.use_itn == 0:
            use_itn = False

        if chunk_size > 0:
            wavs = wavs[chunk_begin : chunk_begin + chunk_size]
        for wav in wavs:
            wav_splits = wav.strip().split()

            wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
            wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
            if not len(wav_path.strip()) > 0:
                continue
            if wav_path.endswith(".pcm"):
                with open(wav_path, "rb") as f:
                    audio_bytes = f.read()
            elif wav_path.endswith(".wav"):
                import wave

                with wave.open(wav_path, "rb") as wav_file:
                    params = wav_file.getparams()
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_bytes = bytes(frames)
            else:
                wav_format = "others"
                with open(wav_path, "rb") as f:
                    audio_bytes = f.read()

            stride = int(60 * self.config.chunk_size[1] / self.config.chunk_interval / 1000 * sample_rate * 2)
            chunk_num = (len(audio_bytes) - 1) // stride + 1
            # print(stride)

            # send first time
            message = json.dumps(
                {
                    "mode": self.config.mode,
                    "chunk_size": self.config.chunk_size,
                    "chunk_interval": self.config.chunk_interval,
                    "encoder_chunk_look_back": self.config.encoder_chunk_look_back,
                    "decoder_chunk_look_back": self.config.decoder_chunk_look_back,
                    "audio_fs": sample_rate,
                    "wav_name": wav_name,
                    "wav_format": wav_format,
                    "is_speaking": True,
                    "hotwords": hotword_msg,
                    "itn": use_itn,
                }
            )

            # voices.put(message)
            await self.websocket.send(message)
            is_speaking = True
            for i in range(chunk_num):

                beg = i * stride
                data = audio_bytes[beg : beg + stride]
                message = data
                # voices.put(message)
                await self.websocket.send(message)
                if i == chunk_num - 1:
                    is_speaking = False
                    message = json.dumps({"is_speaking": is_speaking})
                    # voices.put(message)
                    await self.websocket.send(message)

                sleep_duration = (
                    0.001
                    if self.config.mode == "offline"
                    else 60 * self.config.chunk_size[1] / self.config.chunk_interval / 1000
                )

                await asyncio.sleep(sleep_duration)

        if not self.config.mode == "offline":
            await asyncio.sleep(2)
        # offline model need to wait for message recved

        if self.config.mode == "offline":
            global offline_msg_done
            while not offline_msg_done:
                await asyncio.sleep(1)

        await self.websocket.close()

    async def ws_client(self, id, chunk_begin, chunk_size):
        if self.config.audio_in is None:
            chunk_begin = 0
            chunk_size = 1
        global voices, offline_msg_done

        for i in range(chunk_begin, chunk_begin + chunk_size):
            offline_msg_done = False
            voices = Queue()
            if self.config.ssl_enabled == 1:
                ssl_context = ssl.SSLContext()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                uri = "wss://{}:{}".format(self.config.host, self.config.port)
            else:
                uri = "ws://{}:{}".format(self.config.host, self.config.port)
                ssl_context = None
            print("connect to", uri)
            async with websockets.connect(
                uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
            ) as self.websocket:
                if self.config.audio_in is not None:
                    task = asyncio.create_task(self.record_from_scp(i, 1))
                else:
                    task = asyncio.create_task(self.record_microphone())
                task3 = asyncio.create_task(self.message(str(id) + "_" + str(i)))  # processid+fileid
                await asyncio.gather(task, task3)

    def one_thread(self, id, chunk_begin, chunk_size):
        asyncio.get_event_loop().run_until_complete(self.ws_client(id, chunk_begin, chunk_size))
        # asyncio.get_event_loop().run_forever()

    def start_recognition(self):
        # for microphone
        if self.config.audio_in is None:
            self.one_thread(0, 0, 0)  # 直接调用，不创建新进程
        else:
            # calculate the number of wavs for each preocess
            if self.config.audio_in.endswith(".scp"):
                f_scp = open(self.config.audio_in)
                wavs = f_scp.readlines()
            else:
                wavs = [self.config.audio_in]
            for wav in wavs:
                wav_splits = wav.strip().split()
                wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
                wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
                audio_type = os.path.splitext(wav_path)[-1].lower()

            total_len = len(wavs)
            if total_len >= self.config.thread_num:
                chunk_size = int(total_len / self.config.thread_num)
                remain_wavs = total_len - chunk_size * self.config.thread_num
            else:
                chunk_size = 1
                remain_wavs = 0

            process_list = []
            chunk_begin = 0
            for i in range(self.config.thread_num):
                now_chunk_size = chunk_size
                if remain_wavs > 0:
                    now_chunk_size = chunk_size + 1
                    remain_wavs = remain_wavs - 1
                # process i handle wavs at chunk_begin and size of now_chunk_size
                self.one_thread(i, chunk_begin, now_chunk_size)  # 直接调用，不创建新进程
                chunk_begin = chunk_begin + now_chunk_size

            print("end")

    def start(self):
        """Start the client in a blocking manner."""
        asyncio.get_event_loop().run_until_complete(self.start_recognition())

def create_client(
    host: str = "localhost",
    port: int = 10095,
    audio_in: str = None,
    text_queue = None,
    **kwargs
) -> FunASRClient:
    """Create a FunASR client with the given configuration."""
    config = ASRConfig(
        host=host,
        port=port,
        audio_in=audio_in,
        **kwargs
    )
    return FunASRClient(config, text_queue)
