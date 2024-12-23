import streamlit as st
import subprocess
import os
import asyncio
import sys

def init_session_state():
    """初始化 session_state"""
    if "log_content" not in st.session_state:
        st.session_state.log_content = ""
    if "finetune_running" not in st.session_state:
        st.session_state.finetune_running = False

def set_running_state(state: bool):
    """统一设置运行状态"""
    st.session_state.finetune_running = state

async def read_stream(stream, prefix=""):
    """异步读取流"""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode().strip()
        if text:
            # 只保留最新的1000行日志
            lines = st.session_state.log_content.splitlines()[-999:]
            st.session_state.log_content = "\n".join(lines + [f"{prefix}{text}"])

async def run_finetune_command():
    """异步运行微调命令"""
    try:
        # 切换到 finetune 目录
        finetune_dir = os.path.join(os.getcwd(), "finetune")
        os.chdir(finetune_dir)
        
        # 运行命令
        process = await asyncio.create_subprocess_exec(
            "bash", "finetune.sh",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # 异步读取输出
        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, prefix="错误: ")
        )
        
        # 等待进程结束
        await process.wait()
        
        # 检查返回码
        if process.returncode == 0:
            st.session_state.log_content += "\n训练完成！"
        else:
            st.session_state.log_content += f"\n训练异常退出，返回码: {process.returncode}"
            
    except Exception as e:
        st.session_state.log_content += f"\nError running finetune command: {str(e)}"
        
    finally:
        # 切回原目录
        os.chdir(os.path.dirname(finetune_dir))
        # 标记训练完成
        set_running_state(False)

def render_finetune():
    """渲染微调界面"""
    # 确保 session_state 已初始化
    init_session_state()
    
    st.subheader("模型微调")
    
    # 显示当前状态
    status = "运行中..." if st.session_state.finetune_running else "就绪"
    st.markdown(f"**状态**: {status}")
    
    # 开始微调按钮
    if st.button(
        "开始微调",
        key="start_finetune_btn",
        disabled=st.session_state.finetune_running
    ):
        set_running_state(True)
        st.session_state.log_content = ""
        # 启动异步任务
        asyncio.run(run_finetune_command())
    
    # 显示日志
    st.text_area(
        "训练日志",
        value=st.session_state.log_content,
        height=400,
        key="finetune_log_area"
    )
