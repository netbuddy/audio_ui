import streamlit as st
import os
import tempfile
import subprocess
import logging
import json
import requests
from comp_audio_operation import format_text_with_options
from comp_audio_operation import start_recognition

# 从环境变量获取服务地址
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://localhost:8001")

def extract_audio_from_video(video_file) -> tuple:
    """从视频文件中提取音频，返回临时文件名和文对象"""
    try:
        # 创建临时文件用于保存视频
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
            # 保存上传的视频文件
            temp_video.write(video_file.getvalue())
            
        # 创建临时文件用于保存音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
        # 构建 ffmpeg 命令
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', temp_video_path,  # 使用临时保存的视频文件路径
            '-vn',                  # 不处理视频
            '-acodec', 'pcm_s16le', # 16位音频编码
            '-ar', '16000',         # 采样率16kHz
            '-ac', '1',             # 单声道
            '-y',                   # 覆盖已存在的文件
            temp_audio_path         # 输出音频文件
        ]
        
        # 执行 ffmpeg 命令
        process = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 检查命令执行结果
        if process.returncode != 0:
            error_msg = process.stderr.decode()
            logging.error(f"FFmpeg error: {error_msg}")
            raise Exception(f"Failed to extract audio: {error_msg}")
        
        # 创建 UploadedFile 对象
        with open(temp_audio_path, 'rb') as f:
            audio_content = f.read()
            
        # 创建类似于 st.file_uploader 返回的对象
        class TemporaryUploadedFile:
            def __init__(self, content, filename):
                self._content = content
                self.name = filename
                
            def getvalue(self):
                return self._content
                
        temp_file = TemporaryUploadedFile(audio_content, "video_audio.wav")
        return temp_audio_path, temp_file
        
    except Exception as e:
        st.error(f"提取音频失败: {str(e)}")
        return None, None
        
    finally:
        # 清理临时视频文件
        if 'temp_video_path' in locals():
            try:
                os.remove(temp_video_path)
            except Exception as e:
                logging.warning(f"Error removing temporary video file: {e}")

def start_recognition_from_url():
    """从流媒体URL开始识别"""
    try:
        # 检查服务是否运行
        if st.session_state.service_status != "运行中":
            st.error("请先启动语音识别服务")
            return
            
        # 检查是否有URL
        video_url = st.session_state.get("video_url")
        if not video_url:
            st.error("请先输入视频URL")
            return
            
        # 创建占位符用于显示实时转录结果
        transcript_placeholder = st.empty()
        
        # 使用 SSE 接收实时转录结果
        with st.spinner('正在识别音频...'):
            with requests.get(
                f"{ASR_SERVICE_URL}/recognize_stream",
                params={"url": video_url},
                stream=True,
                headers={"Accept": "text/event-stream"}
            ) as response:
                
                if response.status_code != 200:
                    st.error(f"请求失败: {response.status_code}")
                    return
                    
                # 处理 SSE 事件
                current_text = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            event_data = json.loads(line.decode().split("data: ")[1])
                            if event_data["status"] == "success":
                                result = event_data["result"]
                                
                                # 更新显示的文本
                                if "text" in result:
                                    current_text += result["text"]
                                    transcript_placeholder.text(current_text)
                                    
                                # 如果是最终结果，保存到 session state
                                if event_data.get("is_final"):
                                    st.session_state.recognition_text = current_text
                                    st.session_state.recognition_segments = result.get("segments", [])
                                    
                                    # 格式化显示
                                    formatted_text = format_text_with_options(
                                        current_text,
                                        st.session_state.recognition_segments,
                                        st.session_state.get("show_timestamp", True),
                                        st.session_state.get("show_speaker", True)
                                    )
                                    st.session_state.original_text = formatted_text
                                    
                        except Exception as e:
                            st.error(f"处理转录结果失败: {str(e)}")
                            
    except Exception as e:
        st.error(f"流媒体识别失败: {str(e)}")

def start_recognition_from_video():
    """从视频开始识别"""
    try:
        # 检查服务是否运行
        if st.session_state.service_status != "运行中":
            st.error("请先启动语音识别服务")
            return
            
        # 检查是否有上传的视频文件或URL
        video_url = st.session_state.get("video_url")
        has_upload = "video_upload" in st.session_state and st.session_state.video_upload is not None
        
        if not has_upload and not video_url:
            st.error("请先上传视频文件或输入视频URL")
            return
            
        # 处理本地上传的视频
        if has_upload:
            video_file = st.session_state.video_upload
            # 提取音频
            with st.spinner('正在从视频提取音频...'):
                temp_path, audio_file = extract_audio_from_video(video_file)
                
            if not audio_file:
                return
                
            try:
                # 直接调用 start_recognition 并传入音频文件
                start_recognition(audio_file)
                
            finally:
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                    
        # 处理视频URL
        else:
            # 使用 SSE 接收实时转录结果
            start_recognition_from_url()
                
    except Exception as e:
        st.error(f"视频识别失败: {str(e)}")

def render_video_section():
    st.subheader("视频")
    
    # 在这里添加 JavaScript 注入
    # inject_audio_capture_js()
    
    # 添加视频URL输入
    video_url = st.text_input(
        "视频URL", 
        key="video_url",
        placeholder="输入YouTube、Bilibili等视频链接",
        # 当有本地视频时禁用URL输入
        disabled=bool(st.session_state.get("video_upload"))
    )
    
    # 添加本地视频上传
    uploaded_video = st.file_uploader(
        "上传视频文件", 
        type=['mp4', 'avi'], 
        key="video_upload",
        # 当有URL时禁用文件上传
        disabled=bool(video_url)
    )
    
    # 添加两个按钮
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "识别本地视频", 
            on_click=start_recognition_from_video, 
            key="local_video_btn",
            disabled=not st.session_state.get("video_upload")
        )
    with col2:
        st.button(
            "识别在线视频", 
            on_click=start_recognition_from_video,
            key="url_video_btn",
            disabled=not (video_url and video_url.startswith(('http://', 'https://')))
        )
    
    # 显示视频
    if uploaded_video:
        st.video(uploaded_video)
    elif video_url and video_url.startswith(('http://', 'https://')):
        try:
            st.video(video_url)
        except Exception as e:
            st.error(f"无法加载视频: {str(e)}")
            st.info("提示：目前支持YouTube、Bilibili等主流视频平台的视频链接")