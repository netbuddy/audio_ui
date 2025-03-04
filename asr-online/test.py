import sounddevice as sd
import numpy as np

# 设置你想要使用的PulseAudio sink名称
# 你需要根据之前查询到的实际sink名称进行修改
device_name = 'your_loopback_sink_name_here'

# 查找设备ID
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['name'] == device_name:
        device_id = i
        break
else:
    raise ValueError(f"Device {device_name} not found!")

# 开始录音
def callback(indata, frames, time, status):
    if status:
        print(status)
    print('Received audio data:', indata)

with sd.InputStream(callback=callback, channels=2, device=device_id):
    print("Listening... Press Ctrl+C to stop.")
    try:
        sd.sleep(10000)  # 监听10秒或直到手动停止
    except KeyboardInterrupt:
        print("Recording stopped manually.")