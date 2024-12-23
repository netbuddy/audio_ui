import streamlit as st
import requests
import json
from typing import Optional, List, Dict
import re
import os
from dotenv import load_dotenv
import tempfile
import subprocess
import logging
from comp_audio_model import render_audio_model_sidebar
from comp_lang_model import render_lang_model_sidebar
from service_status import check_service_status
from comp_audio_operation import render_audio_section
from comp_video_operation import render_video_section
from comp_text_postprocess import render_text_postprocess
from comp_train_data_create import render_train_data_create
from comp_finetune import render_finetune
# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(layout="wide")


# 初始化 session state
if 'service_status' not in st.session_state:
    st.session_state.service_status = "未知"
    check_service_status()

# 初始化说话人映射的 session state
if 'speaker_mapping' not in st.session_state:
    st.session_state.speaker_mapping = {}

# 创建 tab 页面
tab1, tab2 = st.tabs(["音频分析", "模型微调"])

# 音频分析 tab
with tab1:
    # 侧边栏区域
    with st.sidebar:
        # 音频模型部分
        render_audio_model_sidebar()
        # 语言模型部分
        render_lang_model_sidebar()

    # 主区域
    # 创建两列布局
    col_audio, col_video = st.columns(2)

    # 音频部分
    with col_audio:
        render_audio_section()

    # 视频部分
    with col_video:
        render_video_section()

    # 文本后处理部分
    render_text_postprocess()

# 模型微调 tab
with tab2:
    render_train_data_create()
    render_finetune()