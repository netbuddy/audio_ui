import streamlit as st
import asyncio
import tempfile
import os
import json
from funasr_wss_client import create_client, ASRConfig
import logging
import traceback
import multiprocessing as mp
import requests

# 初始化session state
if 'asr_text' not in st.session_state:
    st.session_state.asr_text = ""

# 添加服务管理相关的常量
ASR_SERVICE_URL = "http://localhost:10097"

class StreamlitASRClient:
    def __init__(self, config: ASRConfig, text_queue: mp.Queue):
        self.client = create_client(
            host=config.host,
            port=config.port,
            audio_in=config.audio_in,
            output_dir=config.output_dir,
            mode=config.mode,
            text_queue=text_queue
        )

async def process_audio(audio_path: str):
    """处理音频文件"""
    try:
        # 创建进程间通信的队列
        text_queue = mp.Queue()
        
        config = ASRConfig(
            host="192.168.213.11",
            port=10098,
            audio_in=audio_path,
            mode="offline"
        )
        
        client = StreamlitASRClient(config, text_queue)
        
        # 添加进度条
        with st.spinner('正在处理音频文件...'):
            # 启动识别进程，传入队列
            p = mp.Process(target=client.client.start_recognition, args=())
            p.start()
            
            # 主进程监听队列更新
            while p.is_alive():
                if not text_queue.empty():
                    text = text_queue.get()
                    st.session_state.asr_text = text
                    # st.rerun()
                await asyncio.sleep(0.1)
                
            p.join()
            
    except ConnectionError as e:
        st.error(f"连接服务器失败: {str(e)}")
    except asyncio.TimeoutError:
        st.error("处理超时，请检查网络连接或重试")
    except Exception as e:
        st.error(f"处理音频时发生错误: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")

def handle_analyze_click():
    """处理解析按钮点击事件"""
    if st.session_state.uploaded_file is None:
        st.warning("请先上传音频文件")
        return
        
    try:
        # 创建临时文件保存上传的音频
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(st.session_state.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 清空之前的结果
        st.session_state.asr_text = ""
        
        # 运行异步处理
        asyncio.run(process_audio(tmp_file_path))
        
        # 更新文本区域
        st.session_state.text_area_1 = st.session_state.asr_text
        
    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_file_path)
        except Exception:
            pass

def run_model():
    """启动模型服务"""
    try:
        response = requests.post(f"{ASR_SERVICE_URL}/start")
        result = response.json()
        
        if result["status"] == "success":
            st.session_state.model_status = "运行中"
        else:
            st.session_state.model_status = "启动失败"
            st.error(result["message"])
            
    except Exception as e:
        st.session_state.model_status = "启动失败"
        st.error(f"启动服务失败: {str(e)}")

def stop_model():
    """停止模型服务"""
    try:
        response = requests.post(f"{ASR_SERVICE_URL}/stop")
        result = response.json()
        
        if result["status"] == "success":
            st.session_state.model_status = "已停止"
        else:
            st.error(result["message"])
            
    except Exception as e:
        st.error(f"停止服务失败: {str(e)}")

def check_model_status():
    """检查模型服务状态"""
    try:
        response = requests.get(f"{ASR_SERVICE_URL}/status")
        result = response.json()
        st.session_state.model_status = "运行中" if result["status"] == "running" else "已停止"
    except Exception:
        st.session_state.model_status = "未知"

# 初始化模型状态
if 'model_status' not in st.session_state:
    st.session_state.model_status = ["未知", "未知", "未知"]  # 对应三个模型的状态

def update_model_status(model_index: int):
    """更新指定模型的状态"""
    try:
        response = requests.get(f"{ASR_SERVICE_URL}/status/{model_index}")
        result = response.json()
        st.session_state.model_status[model_index] = "运行中" if result["status"] == "running" else "已停止"
    except Exception:
        st.session_state.model_status[model_index] = "未知"

def on_model_select():
    """模型选择变更时的回调函数"""
    selected_index = st.session_state.model_select
    update_model_status(selected_index)

# 页面布局代码
st.set_page_config(layout="wide")

# 创建标签页
tabs = st.tabs(["音频解析", "标签页2", "标签页3"])

# 在第一个标签页中创建音频解析界面
with tabs[0]:
    # 创建三列布局用于顶部控件
    col1, col2, col3 = st.columns(3)

    with col1:
        # 文件上传区域
        uploaded_file = st.file_uploader("上传音频文件", type=None, key="uploaded_file")
        st.audio(uploaded_file)
        st.button("解析", key="analyze", on_click=handle_analyze_click)

    with col2:
        # 模型选择区域
        st.selectbox(
            "模型选择",
            options=range(3),
            format_func=lambda x: f"选项{x+1}",
            key="model_select",
            on_change=on_model_select
        )
        
        col_run_model, col_stop_model, col_model_status = st.columns(3)
        with col_run_model:
            st.button("启动服务", key="run_model", on_click=run_model)
        with col_stop_model:
            st.button("停止服务", key="stop_model", on_click=stop_model)
        with col_model_status:
            # 根据状态显示不同颜色
            status = st.session_state.model_status[st.session_state.model_select]
            color = {
                "运行中": "green",
                "已停止": "red",
                "未知": "gray"
            }.get(status, "gray")
            st.markdown(f'<p style="color: {color}">服务状态: {status}</p>', unsafe_allow_html=True)
        
        # 扩展功能区域
        st.write("扩展功能：")
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.checkbox("标点恢复")
            st.checkbox("说话人区分")
        with col2_2:
            st.checkbox("时间戳输出")

    with col3:
        # 输出格式区域
        st.selectbox("输出格式", ["选项1", "选项2", "选项3"])
        st.text("保存位置：")
        st.button("路径选择对话框")
        st.button("导出")

    # 创建两列布局用于文本显示区域
    col_text1, col_text2 = st.columns(2)

    with col_text1:
        st.text("原始文本：")
        st.text_area("", key="text_area_1", value=st.session_state.get("asr_text", ""), height=200)

    with col_text2:
        st.text("校正文本：")
        st.text_area("", key="text_area_2", value="Sample Text.\nSample Text.\nSample Text.", height=200)

    # 底部控件区域
    col_bottom1, col_bottom2 = st.columns(2)

    with col_bottom1:
        st.selectbox("模型选择：", ["选项1", "选项2", "选项3"], key="bottom_model")

    with col_bottom2:
        # 提示词输入区域
        text_col, button_col = st.columns([4, 1])
        with text_col:
            st.text_input("提示词：", value="Text")
        with button_col:
            st.button("发送", key="send")

# 在第二个标签页中添加内容
with tabs[1]:
    st.write("这是标签页2的内容")

# 在第三个标签页中添加内容
with tabs[2]:
    st.write("这是标签页3的内容")