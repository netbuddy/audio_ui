import streamlit as st
import tempfile
import os
from typing import Optional, Tuple
from moviepy.video.io.VideoFileClip import VideoFileClip
import requests
from config_manager import config
from dataclasses import dataclass, asdict
import asyncio
from comp_audio_model import build_model_config
import websockets
import json
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
import wave
import io
import ffmpeg

# 从环境变量获取服务地址
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://localhost:8001")

@dataclass
class RecognitionConfig:
    """识别配置"""
    batch_size_s: int = 300
    hotword: Optional[str] = None
    use_timestamp: bool = False
    use_itn: bool = True
    max_single_segment: int = 10000

def format_text_with_options(text: str, segments: list, show_timestamp: bool, show_speaker: bool) -> str:
    """根据选项格式化文本"""
    if not segments:
        return text
        
    formatted_text = ""
    current_speaker = None
    
    for segment in segments:
        start_time = segment["start"] / 1000
        end_time = segment["end"] / 1000
        speaker = str(segment.get("speaker", "0"))
        if speaker.startswith("speaker"):
            speaker = speaker.replace("speaker", "")
            
        # 处理说话人显示
        if show_speaker and speaker != current_speaker:
            current_speaker = speaker
            display_name = st.session_state.speaker_mapping.get(
                speaker, f"说话人 {speaker}"
            )
            formatted_text += f"\n[{display_name}]\n"
            
        # 处理文本显示
        line = ""
        if show_timestamp:
            line += f"[{start_time:.2f}-{end_time:.2f}] "
        line += segment["text"]
        formatted_text += line + "\n"
        
    return formatted_text.strip()

def start_recognition(audio_file=None):
    """开始音频识别"""
    try:
        # 检查服务是否运行
        if st.session_state.service_status != "运行中":
            st.error("请先启动语音识别服务")
            return
            
        # 获取音频文件
        if audio_file is None:
            # 检查是否有上传的文件
            if "audio_upload" not in st.session_state or st.session_state.audio_upload is None:
                st.error("请先上传音频文件")
                return
            audio_file = st.session_state.audio_upload
            
        # 调用识别接口
        with st.spinner('正在识别音频...'):
            # 检查是否使用微调模型
            selected_models = st.session_state.get("selected_models", [])
            is_finetune = any(model == "finetune_model" for model in selected_models)
            
            if is_finetune:
                # 使用微调模型路径
                finetune_dir = st.session_state.get("finetune_model_dir")
                model_config = {
                    "model": finetune_dir,
                    "model_revision": "local"
                }
            else:
                # 使用原有的模型配置
                model_config = build_model_config(selected_models)
            
            response = requests.post(
                f"{ASR_SERVICE_URL}/recognize",
                files={
                    'audio_file': (
                        audio_file.name,
                        audio_file.getvalue(),
                        "audio/wav"
                    )
                },
                json={
                    **asdict(RecognitionConfig(
                        batch_size_s=300,
                        use_timestamp=True,
                        use_itn=True,
                        max_single_segment=10000
                    )),
                    **model_config
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    # 保存原始数据到 session state
                    st.session_state.recognition_text = result["result"]["text"]
                    st.session_state.recognition_segments = result["result"].get("segments", [])
                    
                    # 根据当前选项格式化显示
                    formatted_text = format_text_with_options(
                        st.session_state.recognition_text,
                        st.session_state.recognition_segments,
                        st.session_state.get("show_timestamp", True),
                        st.session_state.get("show_speaker", True)
                    )
                    st.session_state.original_text = formatted_text
                    
                else:
                    st.error(f"识别失败: {result.get('message', '未知错误')}")
            else:
                st.error(f"请求失败: {response.status_code}")
                if response.status_code == 422:
                    error_detail = response.json()
                    st.error(f"请求参数错误: {error_detail}")
                
    except Exception as e:
        st.error(f"识别程发生错误: {str(e)}")

def render_audio_section():
    """渲染音频部分"""
    st.subheader("音频")
    uploaded_file = st.file_uploader(
        "上传音频文件", 
        type=['mp3', 'wav'], 
        key="audio_upload"
    )
    st.button(
        "开始识别", 
        on_click=start_recognition, 
        key="audio_recognition_btn"
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
    
    # 使用 session_state 跟踪麦克风状态
    if "last_mic_id" not in st.session_state:
        st.session_state.last_mic_id = None
    
    mic_value = st.audio_input("麦克风")
    if mic_value:
        # 检查是否是新的录音
        current_mic_id = id(mic_value)
        if current_mic_id != st.session_state.last_mic_id:
            st.session_state.last_mic_id = current_mic_id
            asyncio.run(process_audio_input(mic_value))

async def process_audio_input(audio_data):
    """处理音频输入"""
    try:
        # 初始化 session state
        if "streaming_text" not in st.session_state:
            st.session_state.streaming_text = ""
            
        # 获取音频二进制数据
        audio_bytes = audio_data.getvalue()
        
        # 使用 wave 模块解析音频格式
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
            # 获取音频参数
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            print(f"原始音频格式信息:")
            print(f"- 声道数: {channels}")
            print(f"- 采样宽度: {sample_width * 8}bit")
            print(f"- 采样率: {frame_rate}Hz")
            print(f"- 总帧数: {n_frames}")
            print(f"- 时长: {n_frames/frame_rate:.2f}秒")
            
        # 转换音频格式
        try:
            # 使用 ffmpeg 转换音频格式
            process = (
                ffmpeg
                .input('pipe:', format='wav')  # 从内存读取
                .output(
                    'pipe:',  # 输出到内存
                    format='wav',        # WAV 格式
                    acodec='pcm_s16le',  # 16bit
                    ac=1,               # 单声道
                    ar=16000           # 16kHz
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            
            # 写入音频数据
            stdout_data, stderr_data = process.communicate(input=audio_bytes)
            
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg 转换失败: {stderr_data.decode()}")
                
            # 转换后的音频数据
            audio_bytes = stdout_data
            
            print("音频格式转换完成")
            
        except Exception as e:
            raise RuntimeError(f"音频格式转换失败: {str(e)}")
            
        # 通过 WebSocket 连接服务器
        ws_url = f"ws://{ASR_SERVICE_URL.replace('http://', '')}/stream"
        async with websockets.connect(ws_url) as websocket:
            # 发送音频数据
            await websocket.send(audio_bytes)
            print("音频数据发送完成")
            
            # 接收识别结果
            full_text = ""
            while True:
                try:
                    result = await websocket.recv()
                    print(f"接收到原始数据: {result}")
                    result = json.loads(result)
                    print(f"解析后的数据: {result}")
                    
                    if "text" in result and result["text"]:
                        print(f"收到文本: {result['text']}")
                        full_text += result["text"]
                        st.session_state.original_text = full_text
                        print(f"当前完整文本: {full_text}")
                    else:
                        print("接收到的数据中没有文本内容")
                    
                    if result.get("is_final", False):
                        print("识别完成")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket连接已关闭")
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    break
                    
    except Exception as e:
        print(f"错误: {str(e)}")
        st.error(f"处理录音时发生错误: {str(e)}")
