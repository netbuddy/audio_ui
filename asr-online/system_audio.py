import subprocess
import numpy as np
from datetime import datetime
import os
import signal
import threading
from queue import Queue
import time
import asyncio
import json
import websockets
import argparse

class AudioCapture:
    def __init__(self, websocket, monitor_source="alsa_output.pci-0000_00_1f.3.hdmi-stereo.monitor", 
                 sample_rate=16000, channels=1, format="s16le"):
        """初始化音频捕获
        Args:
            websocket: WebSocket连接对象，用于发送音频数据
            monitor_source: PulseAudio监听源名称
            sample_rate: 采样率
            channels: 通道数
            format: 音频格式
        """
        self.websocket = websocket
        self.monitor_source = monitor_source
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.process = None
        self.is_running = False
        self.audio_queue = Queue()
        self.chunk_size = int(0.1 * sample_rate)  # 0.1秒的数据
        
    async def start(self, mode="2pass", chunk_size=[5,10,5], chunk_interval=10,
                   encoder_chunk_look_back=4, decoder_chunk_look_back=0, use_itn=1):
        """开始捕获音频并进行识别"""
        if self.is_running:
            return
            
        # 发送开始消息
        start_message = json.dumps({
            "mode": mode,
            "chunk_size": chunk_size,
            "chunk_interval": chunk_interval,
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "decoder_chunk_look_back": decoder_chunk_look_back,
            "wav_name": "system_audio",
            "is_speaking": True,
            "hotwords": "",
            "itn": use_itn != 0,
        })
        await self.websocket.send(start_message)
            
        # 构建parec命令
        cmd = [
            'parec',
            '-d', self.monitor_source,
            '--rate', str(self.sample_rate),
            '--channels', str(self.channels),
            '--format', self.format,
            '--raw'
        ]
        
        try:
            # 启动parec进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.is_running = True
            
            # 启动数据读取线程
            self.read_thread = threading.Thread(target=self._read_audio)
            self.read_thread.daemon = True
            self.read_thread.start()
            
            print(f"开始捕获系统音频 - 设备: {self.monitor_source}")
            
            # 开始发送音频数据
            while self.is_running:
                audio_chunk = self.get_audio_chunk()
                if audio_chunk is not None:
                    # 发送音频数据到识别服务
                    await self.websocket.send(audio_chunk.tobytes())
                await asyncio.sleep(0.001)  # 避免CPU占用过高
            
        except Exception as e:
            print(f"启动音频捕获失败: {e}")
            self.stop()
            
    async def stop(self):
        """停止捕获音频"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            
        # 发送结束消息
        end_message = json.dumps({"is_speaking": False})
        await self.websocket.send(end_message)

    def _read_audio(self):
        """读取音频数据的线程函数"""
        bytes_per_sample = 2  # 16位音频
        chunk_bytes = self.chunk_size * bytes_per_sample
        
        while self.is_running and self.process:
            try:
                # 读取一个数据块
                audio_data = self.process.stdout.read(chunk_bytes)
                if not audio_data:
                    break
                    
                # 转换为numpy数组
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # 放入队列
                self.audio_queue.put(audio_array)
                
            except Exception as e:
                print(f"读取音频数据出错: {e}")
                break
                
        self.stop()
        
    def get_audio_chunk(self, timeout=1):
        """获取一个音频数据块
        Returns:
            numpy array: 音频数据，如果没有数据返回None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except:
            return None
            
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

async def run_audio_capture(args):
    """运行音频捕获和识别"""
    # 建立WebSocket连接
    uri = f"{'wss' if args.ssl else 'ws'}://{args.host}:{args.port}"
    ssl_context = None
    if args.ssl:
        ssl_context = ssl.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    async with websockets.connect(uri, subprotocols=["binary"], 
                                ping_interval=None, ssl=ssl_context) as websocket:
        # 创建音频捕获对象
        capture = AudioCapture(websocket)
        
        # 启动消息接收任务
        message_task = asyncio.create_task(handle_messages(websocket))
        
        try:
            # 启动音频捕获
            await capture.start(
                mode=args.mode,
                chunk_size=args.chunk_size,
                chunk_interval=args.chunk_interval,
                encoder_chunk_look_back=args.encoder_chunk_look_back,
                decoder_chunk_look_back=args.decoder_chunk_look_back,
                use_itn=args.use_itn
            )
        except KeyboardInterrupt:
            print("\n停止捕获...")
        finally:
            await capture.stop()
            message_task.cancel()

async def handle_messages(websocket):
    """处理识别结果消息"""
    try:
        while True:
            message = await websocket.recv()
            result = json.loads(message)
            
            # 处理识别结果
            if "text" in result:
                mode = result.get("mode", "")
                text = result["text"]
                timestamp = result.get("timestamp", "")
                
                if timestamp:
                    print(f"\r{mode} [{timestamp}]: {text}")
                else:
                    print(f"\r{mode}: {text}")
                
    except websockets.exceptions.ConnectionClosed:
        print("连接已关闭")
    except Exception as e:
        print(f"处理消息时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    # 添加与funasr_wss_client.py相同的参数
    parser.add_argument("--host", type=str, default="192.168.213.11", help="host ip")
    parser.add_argument("--port", type=int, default=10096, help="port")
    parser.add_argument("--mode", type=str, default="2pass", help="offline, online, 2pass")
    parser.add_argument("--chunk_size", type=str, default="5,10,5", help="chunk size")
    parser.add_argument("--chunk_interval", type=int, default=10, help="chunk interval")
    parser.add_argument("--encoder_chunk_look_back", type=int, default=4)
    parser.add_argument("--decoder_chunk_look_back", type=int, default=0)
    parser.add_argument("--ssl", type=int, default=0, help="1 for ssl")
    parser.add_argument("--use_itn", type=int, default=1, help="1 for using itn")
    
    args = parser.parse_args()
    args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
    
    # 运行音频捕获和识别
    asyncio.run(run_audio_capture(args))

if __name__ == "__main__":
    main()
