# -*- encoding: utf-8 -*-
import os
import time
import websockets, ssl
import asyncio

# import threading
import argparse
import json
import traceback
from multiprocessing import Process

# from funasr.fileio.datadir_writer import DatadirWriter

import logging
import pyaudio
import wave
from datetime import datetime
from pynput import keyboard  # 替换 keyboard 库
from threading import Event
from ctypes import *
import soundcard as sc  # 需要安装: pip install soundcard
import pulsectl

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="192.168.213.11", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=10096, required=False, help="grpc server port")
parser.add_argument("--chunk_size", type=str, default="5, 10, 5", help="chunk")
parser.add_argument("--encoder_chunk_look_back", type=int, default=4, help="chunk")
parser.add_argument("--decoder_chunk_look_back", type=int, default=0, help="chunk")
parser.add_argument("--chunk_interval", type=int, default=10, help="chunk")
parser.add_argument(
    "--hotword",
    type=str,
    default="",
    help="hotword file path, one hotword perline (e.g.:阿里巴巴 20)",
)
parser.add_argument("--audio_in", type=str, default=None, help="audio_in")
parser.add_argument("--audio_fs", type=int, default=16000, help="audio_fs")
parser.add_argument(
    "--send_without_sleep",
    action="store_true",
    default=True,
    help="if audio_in is set, send_without_sleep",
)
parser.add_argument("--thread_num", type=int, default=1, help="thread_num")
parser.add_argument("--words_max_print", type=int, default=10000, help="chunk")
parser.add_argument("--output_dir", type=str, default="output_dir", help="output_dir")
parser.add_argument("--ssl", type=int, default=0, help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--use_itn", type=int, default=1, help="1 for using itn, 0 for not itn")
parser.add_argument("--mode", type=str, default="2pass", help="offline, online, 2pass")
parser.add_argument(
    "--record_mode",
    type=str,
    default="soundcard",
    choices=["mic", "file", "soundcard"],
    help="录音模式：mic(麦克风)，file(文件)，soundcard(声卡)",
)

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)
# voices = asyncio.Queue()
from queue import Queue

voices = Queue()
offline_msg_done = False

if args.output_dir is not None:
    # if os.path.exists(args.output_dir):
    #     os.remove(args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


async def record_microphone():
    global voices, websocket
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()
    stream = None
    recording = False
    audio_data = []  # 用于存储录音数据
    space_pressed = Event()  # 用于跨线程通信
    
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                space_pressed.set()  # 设置事件
        except AttributeError:
            pass
            
    # 设置键盘监听器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    # 准备热词信息
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            with open(args.hotword) as f_scp:
                hot_lines = f_scp.readlines()
                fst_dict = {}
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
            hotword_msg = args.hotword

    use_itn = True if args.use_itn != 0 else False
    
    try:
        while True:
            if space_pressed.is_set():  # 检查是否按下空格键
                space_pressed.clear()  # 重置事件
                if not recording:
                    # 开始录音
                    print("\n开始录音...")
                    recording = True
                    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                                  input=True, frames_per_buffer=CHUNK)
                    audio_data = []  # 清空之前的录音数据
                    
                    # 发送开始消息
                    message = json.dumps({
                        "mode": args.mode,
                        "chunk_size": args.chunk_size,
                        "chunk_interval": args.chunk_interval,
                        "encoder_chunk_look_back": args.encoder_chunk_look_back,
                        "decoder_chunk_look_back": args.decoder_chunk_look_back,
                        "wav_name": "microphone",
                        "is_speaking": True,
                        "hotwords": hotword_msg,
                        "itn": use_itn,
                    })
                    await websocket.send(message)
                    await asyncio.sleep(0.3)  # 防止空格键多次触发
                    
                else:
                    # 停止录音
                    print("\n停止录音...")
                    recording = False
                    if stream:
                        stream.stop_stream()
                        stream.close()
                        
                    # 发送结束消息
                    message = json.dumps({"is_speaking": False})
                    await websocket.send(message)
                    
                    # 保存音频文件
                    if audio_data:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wav_path = os.path.join(args.output_dir, f"record_{timestamp}.wav")
                        with wave.open(wav_path, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(audio_data))
                        print(f"录音已保存: {wav_path}")
                    
                    await asyncio.sleep(0.3)  # 防止空格键多次触发
            
            if recording and stream:
                try:
                    data = stream.read(CHUNK)
                    audio_data.append(data)  # 保存录音数据
                    await websocket.send(data)  # 发送音频数据进行识别
                except Exception as e:
                    print(f"录音错误: {e}")
                    recording = False
                    if stream:
                        stream.stop_stream()
                        stream.close()
            
            await asyncio.sleep(0.001)  # 避免CPU占用过高
            
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        listener.stop()  # 停止键盘监听


async def record_from_scp(chunk_begin, chunk_size):
    global voices
    is_finished = False
    if args.audio_in.endswith(".scp"):
        f_scp = open(args.audio_in)
        wavs = f_scp.readlines()
    else:
        wavs = [args.audio_in]

    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            f_scp = open(args.hotword)
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
            hotword_msg = args.hotword
        print(hotword_msg)

    sample_rate = args.audio_fs
    wav_format = "pcm"
    use_itn = True
    if args.use_itn == 0:
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

        stride = int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * sample_rate * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1
        # print(stride)

        # send first time
        message = json.dumps(
            {
                "mode": args.mode,
                "chunk_size": args.chunk_size,
                "chunk_interval": args.chunk_interval,
                "encoder_chunk_look_back": args.encoder_chunk_look_back,
                "decoder_chunk_look_back": args.decoder_chunk_look_back,
                "audio_fs": sample_rate,
                "wav_name": wav_name,
                "wav_format": wav_format,
                "is_speaking": True,
                "hotwords": hotword_msg,
                "itn": use_itn,
            }
        )

        # voices.put(message)
        await websocket.send(message)
        is_speaking = True
        for i in range(chunk_num):

            beg = i * stride
            data = audio_bytes[beg : beg + stride]
            message = data
            # voices.put(message)
            await websocket.send(message)
            if i == chunk_num - 1:
                is_speaking = False
                message = json.dumps({"is_speaking": is_speaking})
                # voices.put(message)
                await websocket.send(message)

            sleep_duration = (
                0.001
                if args.mode == "offline"
                else 60 * args.chunk_size[1] / args.chunk_interval / 1000
            )

            await asyncio.sleep(sleep_duration)

    if not args.mode == "offline":
        await asyncio.sleep(2)
    # offline model need to wait for message recved

    if args.mode == "offline":
        global offline_msg_done
        while not offline_msg_done:
            await asyncio.sleep(1)

    await websocket.close()


async def record_from_soundcard():
    """从声卡录制音频进行实时识别"""
    global voices, websocket
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    # 首先列出所有可用的音频设备
    print("\n=== 系统音频设备信息 ===")
    with pulsectl.Pulse('list-devices') as pulse:
        print("\n可用的音频输出设备及其监听源:")
        for sink in pulse.sink_list():
            print(f"\n输出设备: {sink.name}")
            print(f"描述: {sink.description}")
            print(f"监听源: {sink.monitor_source_name}")
        
        # 获取默认输出设备的监听源
        default_sink = pulse.get_sink_by_name(pulse.server_info().default_sink_name)
        monitor_source = default_sink.monitor_source_name
        print(f"\n将使用默认输出设备的监听源: {monitor_source}")

    p = pyaudio.PyAudio()
    
    # 查找对应的PyAudio设备索引
    output_device_index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"\n设备 {i}: {device_info['name']}")
        if monitor_source in device_info['name']:
            output_device_index = i
            print(f"找到匹配的设备索引: {i}")
            break
    
    if output_device_index is None:
        print("\n未找到匹配的监听设备，将使用默认输入设备")
        output_device_index = p.get_default_input_device_info()['index']

    # 准备热词信息
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            with open(args.hotword) as f_scp:
                hot_lines = f_scp.readlines()
                fst_dict = {}
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
            hotword_msg = args.hotword

    use_itn = True if args.use_itn != 0 else False
    audio_data = []  # 用于存储录音数据
    
    try:
        # 发送开始消息
        message = json.dumps({
            "mode": args.mode,
            "chunk_size": args.chunk_size,
            "chunk_interval": args.chunk_interval,
            "encoder_chunk_look_back": args.encoder_chunk_look_back,
            "decoder_chunk_look_back": args.decoder_chunk_look_back,
            "wav_name": "soundcard",
            "is_speaking": True,
            "hotwords": hotword_msg,
            "itn": use_itn,
        })
        await websocket.send(message)
        
        print("\n开始从声卡录制音频...")
        
        # 开始录制
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=output_device_index,
            frames_per_buffer=CHUNK
        )
        
        while True:
            try:
                # 读取声卡数据
                data = stream.read(CHUNK)
                audio_data.append(data)  # 保存录音数据
                await websocket.send(data)  # 发送音频数据进行识别
                
                # 每隔一段时间保存一次音频文件
                if len(audio_data) >= (RATE * 60) // CHUNK:  # 每60秒保存一次
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wav_path = os.path.join(args.output_dir, f"soundcard_{timestamp}.wav")
                    with wave.open(wav_path, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(audio_data))
                    print(f"录音已保存: {wav_path}")
                    audio_data = []  # 清空缓存
                    
            except KeyboardInterrupt:
                print("\n停止录音...")
                break
            except Exception as e:
                print(f"录音错误: {e}")
                break
            
            await asyncio.sleep(0.001)  # 避免CPU占用过高
            
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        
        # 发送结束消息
        message = json.dumps({"is_speaking": False})
        await websocket.send(message)
        
        # 保存最后的音频数据
        if audio_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = os.path.join(args.output_dir, f"soundcard_{timestamp}.wav")
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(audio_data))
            print(f"录音已保存: {wav_path}")


async def message(id):
    global websocket, voices, offline_msg_done
    text_print = ""
    text_print_2pass_online = ""
    text_print_2pass_offline = ""
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, "text.{}".format(id))
        ibest_writer = open(output_file, "a", encoding="utf-8", buffering=1)
    else:
        ibest_writer = None
    try:
        while True:

            meg = await websocket.recv()
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
                # 先写入调试信息
                ibest_writer.write("\n=== Debug Info ===\n")
                ibest_writer.write(f"Message: {json.dumps(meg, ensure_ascii=False, indent=2)}\n")
                ibest_writer.write(f"Writing line: {text_write_line}")
                ibest_writer.write("=== Debug End ===\n\n")
                # 写入实际内容
                ibest_writer.write(text_write_line)
                ibest_writer.flush()

            if "mode" not in meg:
                continue
            if meg["mode"] == "online":
                text_print += "{}".format(text)
                text_print = text_print[-args.words_max_print :]
                os.system("clear")
                print("\rpid" + str(id) + ": " + text_print)
            elif meg["mode"] == "offline":
                if timestamp != "":
                    text_print += "{} timestamp: {}".format(text, timestamp)
                else:
                    text_print += "{}".format(text)

                # text_print = text_print[-args.words_max_print:]
                # os.system('clear')
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
                text_print = text_print[-args.words_max_print :]
                os.system("clear")
                print("\rpid" + str(id) + ": " + text_print)
                # offline_msg_done=True

    except Exception as e:
        print("Exception:", e)
        # traceback.print_exc()
        # await websocket.close()

    finally:
        if ibest_writer is not None:
            ibest_writer.close()


async def ws_client(id, chunk_begin, chunk_size):
    if args.audio_in is None:
        chunk_begin = 0
        chunk_size = 1
    global websocket, voices, offline_msg_done

    for i in range(chunk_begin, chunk_begin + chunk_size):
        offline_msg_done = False
        voices = Queue()
        if args.ssl == 1:
            ssl_context = ssl.SSLContext()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            uri = "wss://{}:{}".format(args.host, args.port)
        else:
            uri = "ws://{}:{}".format(args.host, args.port)
            ssl_context = None
        print("connect to", uri)
        async with websockets.connect(
            uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
        ) as websocket:
            if args.audio_in is not None:
                task = asyncio.create_task(record_from_scp(i, 1))
            elif args.record_mode == "soundcard":
                task = asyncio.create_task(record_from_soundcard())
            else:
                task = asyncio.create_task(record_microphone())
            task3 = asyncio.create_task(message(str(id) + "_" + str(i)))
            await asyncio.gather(task, task3)
    exit(0)


def one_thread(id, chunk_begin, chunk_size):
    asyncio.get_event_loop().run_until_complete(ws_client(id, chunk_begin, chunk_size))
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    # for microphone
    if args.audio_in is None:
        p = Process(target=one_thread, args=(0, 0, 0))
        p.start()
        p.join()
        print("end")
    else:
        # calculate the number of wavs for each preocess
        if args.audio_in.endswith(".scp"):
            f_scp = open(args.audio_in)
            wavs = f_scp.readlines()
        else:
            wavs = [args.audio_in]
        for wav in wavs:
            wav_splits = wav.strip().split()
            wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
            wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
            audio_type = os.path.splitext(wav_path)[-1].lower()

        total_len = len(wavs)
        if total_len >= args.thread_num:
            chunk_size = int(total_len / args.thread_num)
            remain_wavs = total_len - chunk_size * args.thread_num
        else:
            chunk_size = 1
            remain_wavs = 0

        process_list = []
        chunk_begin = 0
        for i in range(args.thread_num):
            now_chunk_size = chunk_size
            if remain_wavs > 0:
                now_chunk_size = chunk_size + 1
                remain_wavs = remain_wavs - 1
            # process i handle wavs at chunk_begin and size of now_chunk_size
            p = Process(target=one_thread, args=(i, chunk_begin, now_chunk_size))
            chunk_begin = chunk_begin + now_chunk_size
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

        print("end")
