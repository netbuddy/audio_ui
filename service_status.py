import streamlit as st
import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取服务地址
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://localhost:8001")

def check_service_status():
    """检查服务状态，带超时控制"""
    try:
        # 设置超时时间为3秒
        response = requests.get(
            f"{ASR_SERVICE_URL}/status", 
            timeout=3.0  # 连接超时和读取超时都设为3秒
        )
        result = response.json()
        st.session_state.service_status = "运行中" if result["status"] == "running" else "已停止"
    except requests.exceptions.Timeout:
        # 超时异常单独处理
        st.session_state.service_status = "连接超时"
        st.warning(f"连接服务器超时: {ASR_SERVICE_URL}")
    except requests.exceptions.ConnectionError:
        # 连接错误单独处理
        st.session_state.service_status = "无法连接"
        st.warning(f"无法连接到服务器: {ASR_SERVICE_URL}")
    except Exception as e:
        # 其他异常
        st.session_state.service_status = "未知"
        st.error(f"检查服务状态时发生错误: {str(e)}") 