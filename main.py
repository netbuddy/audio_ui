import streamlit as st
import model_metadata
import requests
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import re
import os
from dotenv import load_dotenv
from litellm import completion
import litellm
import tempfile
import subprocess
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip
import asyncio

# 加载环境变量
load_dotenv()

# 定义模型供应商和模型
LLM_PROVIDERS = {
    "OpenAI": {
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "env_vars": ["OPENAI_API_KEY", "OPENAI_API_BASE"]
    },
    "Anthropic": {
        "models": ["claude-3-opus", "claude-3-sonnet", "claude-2.1"],
        "env_vars": ["ANTHROPIC_API_KEY"]
    },
    "Ollama": {
        "models": ["llama2", "mistral", "codellama", "gemma", "qwen2.5:7b"],
        "env_vars": ["OLLAMA_HOST"]
    },
    "智谱 AI": {
        "models": ["chatglm-turbo", "chatglm-pro", "chatglm-std", "chatglm-lite"],
        "env_vars": ["ZHIPU_API_KEY"]
    },
    "百度文心": {
        "models": ["ERNIE-Bot-4", "ERNIE-Bot", "ERNIE-Bot-turbo"],
        "env_vars": ["BAIDU_API_KEY", "BAIDU_SECRET_KEY"]
    },
    "OpenRouter": {
        "models": [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.1-405b-instruct:free"
        ],
        "env_vars": ["OPENROUTER_API_KEY"]
    }
}

def check_provider_config(provider: str) -> bool:
    """检查供应商的环境变量配置是否完整"""
    if provider not in LLM_PROVIDERS:
        return False
    return all(os.getenv(env_var) for env_var in LLM_PROVIDERS[provider]["env_vars"])

def get_available_providers() -> list:
    """获取已配置的供应商列表"""
    return [provider for provider in LLM_PROVIDERS if check_provider_config(provider)]

# 设置服务地址
ASR_SERVICE_URL = "http://localhost:8001"

@dataclass
class RecognitionConfig:
    """识别配置"""
    batch_size_s: int = 300
    hotword: Optional[str] = None
    use_timestamp: bool = False
    use_itn: bool = True
    max_single_segment: int = 10000

def get_speakers_from_text(text: str) -> set:
    """从文本中提取所有说话人ID"""
    speakers = set()
    # 匹配 [说话人 数字] 格式
    pattern = r'\[说话人 (\d+)\]'
    matches = re.findall(pattern, text)
    return set(matches)

def get_model_type(model_name: str) -> str:
    """根据模型名称获取模型类型"""
    if "asr" in model_name.lower():
        return "asr"
    elif "vad" in model_name.lower():
        return "vad"
    elif "punc" in model_name.lower():
        return "punc"
    elif "spk" in model_name.lower() or "cam++" in model_name.lower():
        return "spk"
    return "unknown"

def build_model_config(selected_models: List[str]) -> Dict:
    """根据选择的模型构建配置"""
    config = {}
    
    for model in selected_models:
        model_type = get_model_type(model)
        if model_type == "asr":
            config["model"] = model
            config["model_revision"] = "v2.0.4"
        elif model_type == "vad":
            config["vad_model"] = model
            config["vad_model_revision"] = "v2.0.4"
        elif model_type == "punc":
            config["punc_model"] = model
            config["punc_model_revision"] = "v2.0.4"
        elif model_type == "spk":
            config["spk_model"] = model
            config["spk_model_revision"] = "v2.0.2"
            
    return config

def start_service():
    """启动语音识别服务"""
    try:
        # 获取选中的模型
        selected_models = st.session_state.get("selected_models", [])
        
        if not selected_models:
            st.error("请至少选择一个模型")
            return
            
        # 检查是否包含 ASR 模型
        has_asr = any(get_model_type(model) == "asr" for model in selected_models)
        if not has_asr:
            st.error("必须选择一个语音识别(ASR)模型")
            return
            
        # 构建配置
        config = build_model_config(selected_models)
        
        # 调用服务启动接口
        response = requests.post(f"{ASR_SERVICE_URL}/start", json=config)
        result = response.json()
        
        if result["status"] == "success":
            st.session_state.service_status = "运行中"
            st.success("服务启动成功")
        else:
            st.session_state.service_status = "启动失败"
            st.error(result["message"])
            
    except Exception as e:
        st.session_state.service_status = "启动失败"
        st.error(f"启动服务失败: {str(e)}")

# 停止服务函数
def stop_service():
    """停止语音识别服务"""
    try:
        response = requests.post(f"{ASR_SERVICE_URL}/stop")
        result = response.json()
        
        if result["status"] == "success":
            st.session_state.service_status = "已停止"
        else:
            st.error(result["message"])
            
    except Exception as e:
        st.error(f"停止服务失败: {str(e)}")

# 检查服务状态函数
def check_service_status():
    """检查服务状态"""
    try:
        response = requests.get(f"{ASR_SERVICE_URL}/status")
        result = response.json()
        st.session_state.service_status = "运行中" if result["status"] == "running" else "已停止"
    except Exception:
        st.session_state.service_status = "未知"

# 初始化 session state
if 'service_status' not in st.session_state:
    st.session_state.service_status = "未知"
    check_service_status()

# 初始化说话人映射的 session state
if 'speaker_mapping' not in st.session_state:
    st.session_state.speaker_mapping = {}

def update_speaker_mapping():
    """更新说话人映射"""
    # 获取所有输入的映射
    new_mapping = {}
    for key in st.session_state:
        if key.startswith('speaker_name_'):
            speaker_id = key.replace('speaker_name_', '')
            name = st.session_state[key]
            if name:  # 只保存非空的映射
                new_mapping[speaker_id] = name
    
    st.session_state.speaker_mapping = new_mapping
    
    # 如果有识别结果，更新显示
    if 'original_text' in st.session_state and st.session_state.original_text:
        update_display_with_mapping()

def update_display_with_mapping():
    """使用说话人映射更新显示"""
    text = st.session_state.original_text
    mapping = st.session_state.speaker_mapping
    
    # 替换所有的说话人标记
    for speaker_id, name in mapping.items():
        text = text.replace(f"[说话人 {speaker_id}]", f"[{name}]")
    
    st.session_state.original_text = text

# 设置页面配置
st.set_page_config(layout="wide")

# 侧边栏区域
with st.sidebar:
    # 音视频服务部分
    with st.container():
        st.subheader("音视频服务")
        
        # 模型列表
        st.markdown("##### 模型列表")
        
        # 按类型分组显示模型
        asr_models = [model for model in model_metadata.audio_model_name_maps.keys() 
                     if get_model_type(model) == "asr"]
        vad_models = [model for model in model_metadata.audio_model_name_maps.keys() 
                     if get_model_type(model) == "vad"]
        punc_models = [model for model in model_metadata.audio_model_name_maps.keys() 
                      if get_model_type(model) == "punc"]
        spk_models = [model for model in model_metadata.audio_model_name_maps.keys() 
                     if get_model_type(model) == "spk"]
        
        # 使用 selectbox 进行单选
        st.markdown("**语音识别(ASR)模型**")
        selected_asr = st.selectbox(
            "",
            options=["不使用"] + asr_models,
            key="asr_model"
        )
        
        st.markdown("**语音活动检测(VAD)模型**")
        selected_vad = st.selectbox(
            "",
            options=["不使用"] + vad_models,
            key="vad_model"
        )
        
        st.markdown("**标点恢复(PUNC)模型**")
        selected_punc = st.selectbox(
            "",
            options=["不使用"] + punc_models,
            key="punc_model"
        )
        
        st.markdown("**说话人分离(SPK)模型**")
        selected_spk = st.selectbox(
            "",
            options=["不使用"] + spk_models,
            key="spk_model"
        )
        
        # 合并所有选择的模型
        st.session_state.selected_models = [
            model for model, selected in [
                (selected_asr, selected_asr != "不使用"),
                (selected_vad, selected_vad != "不使用"),
                (selected_punc, selected_punc != "不使用"),
                (selected_spk, selected_spk != "不使用")
            ] if selected
        ]
        
        # 显示选择模型能力
        if st.session_state.selected_models:
            capabilities = set()
            for model in st.session_state.selected_models:
                capabilities.update(
                    model_metadata.audio_model_capability_maps.get(model, [])
                )
            st.markdown("##### 已选模型能力")
            st.write(", ".join(sorted(capabilities)))
        
        # 启动和关闭按钮
        col3, col4 = st.columns(2)
        with col3:
            st.button("启动", on_click=start_service)
        with col4:
            st.button("关闭", on_click=stop_service)
        
        # 显示服务状态
        status = st.session_state.service_status
        color = {
            "运行中": "green",
            "已停止": "red",
            "未知": "gray",
            "启动失败": "red"
        }.get(status, "gray")
        st.markdown(f'<p style="color: {color}">音视频服务状态: {status}</p>', 
                   unsafe_allow_html=True)
    
    # 语言模型部分
    with st.container():
        st.subheader("语言模型")
        
        # 获取已配置的供应商
        available_providers = get_available_providers()
        
        if not available_providers:
            st.warning("未找到已配置的模型供应商。请在 .env 文件中配置相关参数。")
            st.markdown("""
            示例配置：        ```env
            OPENAI_API_KEY=sk-xxx
            ANTHROPIC_API_KEY=sk-xxx
            OLLAMA_HOST=http://localhost:11434
            ZHIPU_API_KEY=xxx
            BAIDU_API_KEY=xxx
            BAIDU_SECRET_KEY=xxx        ```
            """)
        else:
            # 模型供应商选择
            st.markdown("##### 模型供应商")
            selected_provider = st.selectbox(
                "",
                options=["请选择"] + available_providers,
                key="llm_provider"
            )
            
            # 模型列表选择
            st.markdown("##### 模型列表")
            if selected_provider and selected_provider != "请选择":
                available_models = LLM_PROVIDERS[selected_provider]["models"]
                selected_model = st.selectbox(
                    "",
                    options=["请选择"] + available_models,
                    key="llm_model"
                )
                
                # 显示当前配置
                if selected_model and selected_model != "请选择":
                    st.markdown("##### 当前配置")
                    config_md = f"""
                    - **供应商**: {selected_provider}
                    - **模型**: {selected_model}
                    - **接口地址**: {os.getenv(LLM_PROVIDERS[selected_provider]['env_vars'][0])}
                    """
                    st.markdown(config_md)
            
            # 系统提示词
            system_prompt_text = """你是一位专业的中文文本校对助手，擅长识别和纠正语音识别的文本错误。我将提供一段语音识别后的中文文本，请你按照以下步骤进行仔细校对：

校对指南：
1. 全面检查文本，重点关注以下方面：
   - 同音字替换
   - 语法错误
   - 标点使用
   - 语义准确性

2. 具体校对步骤：
   a) 首先快速通读整段文本，理解原始语境和表达意图
   b) 逐字逐句进行仔细校对
   c) 识别可能的错误类型，包括但不限于：
      - 近音字替换（如"几"和"鸡"）
      - 语气词使用错误
      - 分词不准确
      - 标点符号使用不当

3. 修正原则：
   - 只进行同音、近音字和标点符号替换
   - 对于语法不当的位置可以提出修改建议，但不能修改原文
   - 绝对不要进行基于行文流畅的文本润色，绝对不要为了语法结构完整、通顺而加字、减字

4. 输出要求：
   - 提供修正后的完整文本
   - 在文本末尾附上修改说明
   - 对于不确定的地方，用注释标注

5. 特别提示：
   - 对于专业术语，确保准确性
   - 注意保持原文的语气和语调
   - 如有重大疑问，请说明需要进一步确认

请开始校对。我会提供需要校对的语音识别文本。"""
            system_prompt = st.text_area(
                "系统提示词",
                value=system_prompt_text,
                key="system_prompt_input",
                height=100
            )

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

def extract_audio_from_video(video_file) -> tuple:
    """从视频文件中提取音频，返回临时文件名和文件对象"""
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
            with st.spinner('正在识别在线视频音频...'):
                response = requests.post(
                    f"{ASR_SERVICE_URL}/recognize_stream",
                    json={"url": video_url},
                    stream=True,
                    headers={"Accept": "text/event-stream"}
                )
                
                if response.status_code != 200:
                    st.error(f"请求失败: {response.status_code}")
                    return
                    
                # 创建占位符用于显示实时转录结果
                transcript_placeholder = st.empty()
                current_text = ""
                
                # 处理 SSE 事件
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode()
                            # 只处理数据行
                            if line_text.startswith("data: "):
                                data = line_text.replace("data: ", "")
                                event_data = json.loads(data)
                                
                                if event_data["status"] == "success":
                                    result = event_data["result"]
                                    
                                    # 更新显示的文本
                                    if isinstance(result, list) and result and "text" in result[0]:
                                        current_text += result[0]["text"]
                                        transcript_placeholder.text(current_text)
                                        
                                    # 如果是最终结果，保存到 session state
                                    if event_data.get("is_final"):
                                        st.session_state.recognition_text = current_text
                                        if isinstance(result, list) and result:
                                            st.session_state.recognition_segments = result[0].get("segments", [])
                                        else:
                                            st.session_state.recognition_segments = []
                                        
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
                            st.error(f"Error processing transcript: {e}", exc_info=True)
                
    except Exception as e:
        st.error(f"视频识别失败: {str(e)}")

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

# 主区域
# 创建两列布局
col_audio, col_video = st.columns(2)

# 音频部分
with col_audio:
    st.subheader("音频")
    uploaded_file = st.file_uploader("上传音频文件", type=['mp3', 'wav'], key="audio_upload")
    st.button("开始识别", on_click=start_recognition, key="audio_recognition_btn")
    st.audio(uploaded_file)
    audio_value = st.audio_input("音频输入")

def inject_audio_capture_js():
    """注入基于 WebRTC 的音频捕获 JavaScript 代码"""
    st.markdown("""
    <script>
    class AudioStreamProcessor {
        constructor() {
            this.mediaRecorder = null;
            this.websocket = null;
            this.audioContext = null;
            this.isRecording = false;
            this.chunks = [];
            this.CHUNK_SIZE = 2048; // 每个音频块的大小
        }
        
        async initAudioCapture(videoElement) {
            try {
                // 获取视频的音频轨道
                const stream = videoElement.captureStream();
                const audioTrack = stream.getAudioTracks()[0];
                
                if (!audioTrack) {
                    console.error('No audio track found in video');
                    return;
                }
                
                // 创建音频上下文
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000,
                    channelCount: 1
                });
                
                // 创建 MediaStream
                const audioStream = new MediaStream([audioTrack]);
                
                // 创建 MediaRecorder
                this.mediaRecorder = new MediaRecorder(audioStream, {
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: 16000
                });
                
                // 连接 WebSocket
                this.websocket = new WebSocket('ws://localhost:8001/stream');
                
                // 处理音频数据
                this.mediaRecorder.ondataavailable = async (event) => {
                    if (event.data.size > 0 && this.websocket.readyState === WebSocket.OPEN) {
                        // 将 Blob 转换为 ArrayBuffer
                        const arrayBuffer = await event.data.arrayBuffer();
                        const audioData = await this.audioContext.decodeAudioData(arrayBuffer);
                        const pcmData = this.convertToMono16(audioData);
                        
                        // 分块发送数据
                        for (let i = 0; i < pcmData.length; i += this.CHUNK_SIZE) {
                            const chunk = pcmData.slice(i, i + this.CHUNK_SIZE);
                            this.websocket.send(chunk.buffer);
                        }
                    }
                };
                
                // 开始录制
                this.mediaRecorder.start(100); // 每 100ms 触发一次 ondataavailable
                this.isRecording = true;
                
                // 监听视频事件
                videoElement.addEventListener('pause', () => this.pauseRecording());
                videoElement.addEventListener('play', () => this.resumeRecording());
                videoElement.addEventListener('ended', () => this.stopRecording());
                
            } catch (error) {
                console.error('Error initializing audio capture:', error);
            }
        }
        
        convertToMono16(audioBuffer) {
            const samples = audioBuffer.getChannelData(0);
            const pcmData = new Int16Array(samples.length);
            
            for (let i = 0; i < samples.length; i++) {
                pcmData[i] = Math.max(-1, Math.min(1, samples[i])) * 0x7FFF;
            }
            
            return pcmData;
        }
        
        pauseRecording() {
            if (this.isRecording && this.mediaRecorder) {
                this.mediaRecorder.pause();
                this.isRecording = false;
            }
        }
        
        resumeRecording() {
            if (!this.isRecording && this.mediaRecorder) {
                this.mediaRecorder.resume();
                this.isRecording = true;
            }
        }
        
        async stopRecording() {
            if (this.mediaRecorder) {
                this.mediaRecorder.stop();
                this.isRecording = false;
            }
            
            if (this.websocket) {
                // 发送结束信号
                await this.websocket.send(JSON.stringify({
                    type: 'end'
                }));
                this.websocket.close();
            }
            
            if (this.audioContext) {
                await this.audioContext.close();
            }
        }
    }
    
    // 创建音频处理器实例
    const audioProcessor = new AudioStreamProcessor();
    
    // 监听视频元素
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes) {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeName === 'VIDEO') {
                        audioProcessor.initAudioCapture(node);
                    }
                });
            }
        });
    });
    
    // 开始监听 DOM 变化
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    </script>
    """, unsafe_allow_html=True)

# 视频部分
with col_video:
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

on_timestamp = st.toggle("时间戳", value=True)
on_speaker = st.toggle("说话人", value=True)

# 当选项改变时更新显示
if "recognition_text" in st.session_state and "recognition_segments" in st.session_state:
    st.session_state.show_timestamp = on_timestamp
    st.session_state.show_speaker = on_speaker
    
    formatted_text = format_text_with_options(
        st.session_state.recognition_text,
        st.session_state.recognition_segments,
        on_timestamp,
        on_speaker
    )
    st.session_state.original_text = formatted_text

# 音视频文本部分
st.subheader("音视频文本")
col_original, col_corrected = st.columns(2)

with col_original:
    st.markdown("##### 原始文本")
    st.text_area("", height=200, key="original_text")

with col_corrected:
    st.markdown("##### 校正文本")
    st.text_area("", height=200, key="corrected_text")

# 说话人映射部分
st.subheader("说话人映射")

# 获取当前文本中的说话人
if 'original_text' in st.session_state and st.session_state.original_text:
    speakers = get_speakers_from_text(st.session_state.original_text)
    
    if speakers:
        # 计算每行显示的列数（最多4列）
        cols_per_row = min(4, len(speakers))
        rows_needed = (len(speakers) + cols_per_row - 1) // cols_per_row
        
        # 按行显示说话人输入框
        for row in range(rows_needed):
            cols = st.columns(cols_per_row)
            start_idx = row * cols_per_row
            
            # 填充当前的列
            for col_idx in range(cols_per_row):
                speaker_idx = start_idx + col_idx
                if speaker_idx < len(speakers):
                    speaker_id = sorted(speakers)[speaker_idx]
                    with cols[col_idx]:
                        st.text_input(
                            f"说话人 {speaker_id}",
                            key=f"speaker_name_{speaker_id}",
                            on_change=update_speaker_mapping,
                            placeholder="请输入说话人名字"
                        )
    else:
        st.info("当前文中没有检测到说话人")
else:
    st.info("请先进行语音识别")

# 初始化 LiteLLM
def init_litellm():
    """初始化 LiteLLM 配置"""
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        litellm.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_API_BASE"):
            litellm.api_base = os.getenv("OPENAI_API_BASE")
            
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        litellm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
    # 智谱 AI
    if os.getenv("ZHIPU_API_KEY"):
        litellm.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        
    # 百度文心
    if os.getenv("BAIDU_API_KEY") and os.getenv("BAIDU_SECRET_KEY"):
        litellm.baidu_api_key = os.getenv("BAIDU_API_KEY")
        litellm.baidu_secret_key = os.getenv("BAIDU_SECRET_KEY")
    
    # OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        litellm.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        # 不需要设置 api_base，让 litellm 使用默认值

# 调用 LLM 的函数
async def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str) -> str:
    """调用语言模型"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if provider == "OpenRouter":
            # 修改 OpenRouter 的调用方式
            response = completion(
                model="openrouter/" + model,  # 添加 openrouter/ 前缀
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                api_base="https://openrouter.ai/api/v1",
                headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "FunASR Text Correction"
                }
            )
        elif provider == "Ollama":
            # 特殊处理 qwen2.5:7b 模型
            if model == "qwen2.5:7b":
                model_name = "ollama/qwen2.5:7b"
                model_max_tokens = 8192
            else:
                model_name = f"ollama/{model}"
            
            # Ollama 特定配置
            response = completion(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=model_max_tokens,
                api_base=os.getenv("OLLAMA_HOST"),  # 使用环境变量中的地址
                stop=["Human:", "Assistant:"]  # Qwen 特定的停止标记
            )
        else:
            # 其他模型的处理保持不变
            if provider == "OpenAI":
                model_name = f"openai/{model}"
            elif provider == "Anthropic":
                model_name = f"anthropic/{model}"
            elif provider == "智谱 AI":
                model_name = f"zhipu/{model}"
            elif provider == "百度文心":
                model_name = f"baidu/{model}"
            
            response = completion(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"调用语言模型失败: {str(e)}")
        return None

# 修改处理校正提示词的函数为异步函数
async def process_correction():
    """处理校正提示词"""
    try:
        # 检查是否选择了模型
        if not st.session_state.get("llm_provider") or not st.session_state.get("llm_model"):
            st.error("请先选择语言模型")
            return
            
        # 检查是否有原始文本
        if not st.session_state.get("original_text"):
            st.error("请先进行语音识别")
            return
            
        # 获取系统提示词
        system_prompt = st.session_state.get("system_prompt_input", "")
        if not system_prompt.strip():
            st.error("请先输入系统提示词")
            return
        
        # 构建用户提示词
        correction_prompt = st.session_state.get("correction_prompt", "")
        user_prompt = f"""请校正以下文本：

{st.session_state.original_text}

校正要求：{correction_prompt if correction_prompt else '请进行基本的文本校正'}"""
        
        # 调用语言模型
        corrected_text = await call_llm(
            st.session_state.llm_provider,
            st.session_state.llm_model,
            system_prompt,
            user_prompt
        )
        
        # 更新校正文本
        if corrected_text:
            st.session_state.corrected_text = corrected_text
            
    except Exception as e:
        st.error(f"处理校正提示词失败: {str(e)}")

# 创建同步包装函数
def handle_correction():
    """同步处理校正提示词"""
    import asyncio
    asyncio.run(process_correction())

# 修改按钮的回调函数
col_prompt, col_button = st.columns([5,1])
with col_prompt:
    correction_prompt = st.text_input(
        "校正提示词",
        key="correction_prompt",
        placeholder="请输入具体的校正要求，例如：'保持口语化风格，仅修正明显错误'"
    )
with col_button:
    st.button("发送", on_click=handle_correction)  # 使用同步包装函数
