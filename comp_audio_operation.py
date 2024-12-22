import streamlit as st
import tempfile
import os
from typing import Optional, Tuple
from moviepy.video.io.VideoFileClip import VideoFileClip
import requests
from config_manager import config
from dataclasses import dataclass, asdict
import asyncio
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
            response = requests.post(
                f"{ASR_SERVICE_URL}/recognize",
                files={
                    'audio_file': (
                        audio_file.name,
                        audio_file.getvalue(),
                        "audio/wav"
                    )
                },
                json=asdict(RecognitionConfig(
                    batch_size_s=300,
                    use_timestamp=True,
                    use_itn=True,
                    max_single_segment=10000
                ))
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
    
    mic_value = st.audio_input("麦克风")
    if mic_value:
        process_audio_input(mic_value)



def process_audio_input(audio_data):
    # """处理音频输入"""
    # try:
    #     # 创建临时文件保存录音
    #     with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
    #         # 写入录音数据
    #         tmp_file.write(audio_data)
    #         audio_path = tmp_file.name
            
    #     # 开始识别
    #     asyncio.run(process_audio(audio_path))
        
    #     # 清理临时文件
    #     os.unlink(audio_path)
        
    # except Exception as e:
    #     st.error(f"处理录音时发生错误: {str(e)}")
    pass
