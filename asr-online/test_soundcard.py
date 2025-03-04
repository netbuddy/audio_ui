import alsaaudio
import wave
import time
import numpy as np
from datetime import datetime

def list_audio_devices():
    """列出所有音频设备"""
    print("\n=== 可用的音频设备 ===")
    # 列出所有捕获设备（录音设备）
    print("\n捕获设备:")
    for device in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
        print(f"- {device}")
    
    # 列出所有播放设备
    print("\n播放设备:")
    for device in alsaaudio.pcms(alsaaudio.PCM_PLAYBACK):
        print(f"- {device}")

def record_from_soundcard(duration=10, device='default'):
    """
    从声卡录制音频
    duration: 录制时长（秒）
    device: 录制设备名称
    """
    # 音频参数
    CHANNELS = 1
    RATE = 16000
    FORMAT = alsaaudio.PCM_FORMAT_S16_LE
    PERIOD_SIZE = 160  # 每次读取的帧数
    
    try:
        # 打开音频输入
        inp = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NONBLOCK,
            device=device
        )
        
        # 设置参数
        inp.setchannels(CHANNELS)
        inp.setrate(RATE)
        inp.setformat(FORMAT)
        inp.setperiodsize(PERIOD_SIZE)
        
        print(f"\n开始录制音频，设备: {device}")
        print(f"采样率: {RATE}Hz")
        print(f"通道数: {CHANNELS}")
        print(f"录制时长: {duration}秒")
        
        # 准备保存数据
        frames = []
        start_time = time.time()
        
        while True:
            current_time = time.time()
            if current_time - start_time > duration:
                break
                
            # 读取音频数据
            length, data = inp.read()
            if length:
                frames.append(data)
                # 打印音量级别
                if len(frames) % 100 == 0:  # 每100帧打印一次
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = np.abs(audio_data).mean()
                    print(f"\r当前音量: {volume:.0f}", end='')
        
        print("\n\n录制完成!")
        
        # 保存为WAV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = f"soundcard_test_{timestamp}.wav"
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16位采样
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"文件已保存: {wav_path}")
        
    except Exception as e:
        print(f"\n录制出错: {e}")
    finally:
        if 'inp' in locals():
            inp.close()

if __name__ == "__main__":
    # 列出所有设备
    list_audio_devices()
    
    # 选择要使用的设备
    print("\n请选择录制设备:")
    print("1. 默认设备 (default)")
    print("2. 系统声音监听 (hw:Loopback,0)")
    print("3. 手动输入设备名称")
    
    choice = input("\n请输入选项 (1/2/3): ")
    
    if choice == '1':
        device = 'default'
    elif choice == '2':
        device = 'hw:Loopback,0'
    else:
        device = input("请输入设备名称: ")
    
    # 设置录制时长
    try:
        duration = int(input("\n请输入录制时长(秒): "))
    except:
        duration = 10
        print(f"使用默认时长: {duration}秒")
    
    # 开始录制
    record_from_soundcard(duration, device) 