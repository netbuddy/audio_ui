import streamlit as st
import requests
import json
import os
import tempfile
import subprocess
from typing import Dict, List, Tuple
import uuid

# 从环境变量获取服务地址
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://localhost:8001")

def split_audio(audio_file: str, segments: List[Tuple[int, int]], output_dir: str) -> List[str]:
    """使用 ffmpeg 分割音频"""
    audio_files = []
    
    for i, (start_ms, end_ms) in enumerate(segments):
        # 转换为秒
        start_sec = start_ms / 1000
        duration_sec = (end_ms - start_ms) / 1000
        
        # 构建输出文件名
        output_file = os.path.join(output_dir, f"segment_{i:03d}.wav")
        
        # 构建 ffmpeg 命令
        cmd = [
            'ffmpeg',
            '-i', audio_file,
            '-ss', str(start_sec),
            '-t', str(duration_sec),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_file
        ]
        
        # 执行命令
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            audio_files.append(output_file)
        except subprocess.CalledProcessError as e:
            st.error(f"分割音频段 {i} 失败: {e.stderr.decode()}")
            continue
            
    return audio_files

def recognize_audio_segment(audio_file: str) -> str:
    """识别单个音频段"""
    try:
        with open(audio_file, 'rb') as f:
            files = {
                'audio_file': (
                    os.path.basename(audio_file),
                    f.read(),
                    'audio/wav'
                )
            }
            
            response = requests.post(
                f"{ASR_SERVICE_URL}/recognize",
                files=files,
                json={
                    "batch_size_s": 300,
                    "use_timestamp": False,
                    "use_itn": True,
                    "max_single_segment": 10000
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    return result["result"]["text"]
            
            return ""
            
    except Exception as e:
        st.error(f"识别音频段失败: {str(e)}")
        return ""

def start_vad_detection():
    """开始音频分割"""
    try:          
        # 检查是否选择了保存目录
        if not hasattr(st.session_state, 'save_dir'):
            st.error("请先选择保存目录")
            return
            
        save_dir = st.session_state.save_dir
            
        # 启动 VAD 服务
        response = requests.post(f"{ASR_SERVICE_URL}/vad/start")
        if response.status_code != 200:
            st.error(f"启动 VAD 服务失败: {response.status_code}")
            return
            
        # 保存上传的音频文件
        audio_file = st.session_state.train_audio_upload
        input_audio = os.path.join(save_dir, "input.wav")
        with open(input_audio, "wb") as f:
            f.write(audio_file.getvalue())
        
        # 发送音频文件进行分割
        files = {
            'audio_file': (
                audio_file.name,
                audio_file.getvalue(),
                'audio/wav'
            )
        }
        
        with st.spinner('正在进行音频分割...'):
            response = requests.post(
                f"{ASR_SERVICE_URL}/vad/detect",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    segments = result["result"]["segments"]
                    
                    # 分割音频
                    audio_files = split_audio(input_audio, segments, save_dir)
                    
                    # 识别每个音频段
                    with st.spinner('正在识别音频段...'):
                        segment_texts = []
                        for audio_file in audio_files:
                            text = recognize_audio_segment(audio_file)
                            segment_texts.append(text)
                    
                    # 更新 session state
                    st.session_state.audio_segments = [
                        {
                            "file": file,
                            "start": start,
                            "end": end,
                            "text": text
                        }
                        for file, (start, end), text in zip(audio_files, segments, segment_texts)
                    ]
                    
                    st.success(f"音频分割完成，共 {len(audio_files)} 个片段")
                else:
                    st.error(f"分割失败: {result.get('message', '未知错误')}")
            else:
                st.error(f"请求失败: {response.status_code}")
            
    except Exception as e:
        st.error(f"音频分割失败: {str(e)}")
    finally:
        # 停止 VAD 服务
        try:
            requests.post(f"{ASR_SERVICE_URL}/vad/stop")
        except Exception:
            pass

def generate_unique_id() -> str:
    """生成全局唯一ID"""
    return f"ID{uuid.uuid4().hex[:8]}"

def export_train_data():
    """导出训练数据"""
    try:
        # 检查是否有音频段数据
        if not hasattr(st.session_state, 'audio_segments'):
            st.error("没有可导出的音频段数据")
            return
            
        audio_segments = st.session_state.audio_segments
        if not audio_segments:
            st.error("没有可导出的音频段数据")
            return
            
        # 获取保存路径
        save_path = st.session_state.get("train_data_path", "/sata/train_data")
            
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 构建训练数据
        train_data = []
        for segment in audio_segments:
            # 获取音频文件的绝对路径
            audio_path = os.path.abspath(segment["file"])
            text = segment.get("text", "").strip()
            
            if not text:  # 跳过没有文本的片段
                continue
                
            item = {
                "key": generate_unique_id(),
                "source": audio_path,
                "source_len": len(audio_path),
                "target": text,
                "target_len": len(text)
            }
            train_data.append(item)
        
        if not train_data:
            st.error("没有有效的训练数据可导出")
            return
            
        # 保存为 jsonl 文件
        output_file = os.path.join(save_path, "train.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        st.success(f"成功导出 {len(train_data)} 条训练数据到 {output_file}")
        
    except Exception as e:
        st.error(f"导出训练数据失败: {str(e)}")

def render_train_data_create():
    """渲染音频模型侧边栏"""
    with st.container():
        # 保存目录选择
        with st.expander("保存目录设置", expanded=True):
            # 音频段保存目录
            save_dir = st.text_input(
                "音频段保存目录",
                value="/sata/audio_segments",
                key="save_dir_input"
            )
            if st.button("确认目录"):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    st.session_state.save_dir = save_dir
                    st.success(f"将保存到目录: {save_dir}")
                except Exception as e:
                    st.error(f"创建目录失败: {str(e)}")
            
            # 训练数据保存目录
            st.text_input(
                "训练数据保存路径",
                value="/sata/train_data",
                key="train_data_path"
            )
        
        # 从 session state 获取音频段
        audio_segments = st.session_state.get("audio_segments", [])
        
        # 生成音频段选项列表
        segment_options = [
            f"片段 {i+1}" for i in range(len(audio_segments))
        ]
        
        # 初始化变量
        segment = None
        current_text = ""
        
        # 获取当前选中的片段索引
        current_segment_idx = -1
        
        if segment_options:
            # 保存上一次的修改
            if "last_segment_idx" in st.session_state and "text_area" in st.session_state:
                last_idx = st.session_state.last_segment_idx
                if 0 <= last_idx < len(audio_segments):
                    audio_segments[last_idx]["text"] = st.session_state.text_area
                    st.session_state.audio_segments = audio_segments
            
            selection = st.segmented_control(
                "音频段",
                segment_options,
                selection_mode="single",
                key="segment_selector"
            )
            
            # 如果有选择，获取对应的文本
            if selection:
                current_segment_idx = int(selection.split()[-1]) - 1
                segment = audio_segments[current_segment_idx]
                current_text = segment.get("text", "")
                # 保存当前选中的片段索引
                st.session_state.last_segment_idx = current_segment_idx
        
        # 创建两列布局
        col_audio, col_text = st.columns(2)
        
        with col_audio:
            train_audio_upload = st.file_uploader(
                "上传音频文件", 
                type=['mp3', 'wav'], 
                key="train_audio_upload"
            )

            st.button(
                "音频分割", 
                on_click=start_vad_detection, 
                key="train_audio_vad_detection_btn"
            )
            
            # 如果选择了音频段，显示信息和播放器
            if segment:
                st.markdown(f"""
                **开始时间**: {segment['start']/1000:.2f}秒  
                **结束时间**: {segment['end']/1000:.2f}秒  
                **持续时间**: {(segment['end'] - segment['start'])/1000:.2f}秒
                """)
                
                # 播放音频段
                st.audio(segment["file"])

        with col_text:
            st.text_area(
                "文本", 
                value=current_text,
                key="text_area",
                height=200
            )

        # 添加导出按钮
        if st.session_state.get("audio_segments"):
            st.button(
                "导出训练数据",
                on_click=export_train_data,
                key="export_train_data_btn"
            )
