import streamlit as st
import os
import requests
import json
from typing import Optional, List, Dict
from service_status import check_service_status
from config_manager import config

# 从环境变量获取服务地址
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://localhost:8001")

def get_model_type(model_name: str) -> str:
    """根据模型名称或路径获取模型类型"""
    # 如果是本地模型路径
    if os.path.isdir(model_name):
        config_path = os.path.join(model_name, "configuration.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if config.get("task") == "auto-speech-recognition":
                        return "asr"
            except Exception:
                pass
    # 原有的模型名称判断逻辑
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
    result = {}
    model_revisions = config.get_model_revisions()
    
    for model in selected_models:
        # 处理微调模型
        if model == "finetune_model":
            result["model"] = st.session_state.get("finetune_model_dir")
            result["model_revision"] = "local"
            continue
            
        model_type = get_model_type(model)
        if model_type == "asr":
            result["model"] = model
            result["model_revision"] = model_revisions.get("asr", "v2.0.4")
        elif model_type == "vad":
            result["vad_model"] = model
            result["vad_model_revision"] = model_revisions.get("vad", "v2.0.4")
        elif model_type == "punc":
            result["punc_model"] = model
            result["punc_model_revision"] = model_revisions.get("punc", "v2.0.4")
        elif model_type == "spk":
            result["spk_model"] = model
            result["spk_model_revision"] = model_revisions.get("spk", "v2.0.2")
            
    return result

def start_service():
    """启动语音识别服务"""
    try:
        # 获取选中的模型
        selected_models = st.session_state.get("selected_models", [])
        
        if not selected_models:
            st.error("请至少选择一个模型")
            return
            
        # 检查是否包含 ASR 模型或微调模型
        has_asr = any(model == "finetune_model" or get_model_type(model) == "asr" 
                     for model in selected_models)
        if not has_asr:
            st.error("必须选择一个语音识别(ASR)模型")
            return
            
        # 构建配置
        model_config = build_model_config(selected_models)
        
        # 调用服务启动接口
        response = requests.post(f"{ASR_SERVICE_URL}/start", json=model_config)
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

def render_audio_model_sidebar():
    """渲染音频模型侧边栏"""
    with st.container():
        st.subheader("音视频服务")
        
        # 推理框架选择
        st.markdown("##### 推理框架")
        providers = config.audio_model_providers
        provider_names = list(providers.keys())
        
        selected_provider = st.selectbox(
            "",
            options=["请选择"] + provider_names,
            key="audio_provider"
        )
        
        if selected_provider and selected_provider != "请选择":
            provider_config = providers[selected_provider]
            
            # 显示框架描述
            st.markdown(f"*{provider_config['description']}*")
            
            # 模型列表
            st.markdown("##### 模型列表")
            
            # 获取当前框架的模型
            model_names = provider_config["models"]["name_maps"]
            
            # 添加本地微调模型选项
            if selected_provider == "FunASR":
                st.markdown("**本地微调模型**")
                finetune_dir = st.text_input(
                    "微调模型路径",
                    value="./finetune/outputs",
                    key="finetune_model_dir"
                )
                
                # 检查是否存在微调模型
                if os.path.exists(finetune_dir):
                    config_path = os.path.join(finetune_dir, "configuration.json")
                    if os.path.exists(config_path):
                        st.success("检测到微调模型")
                        # 将微调模型添加到模型列表
                        model_names = {"finetune_model": finetune_dir, **model_names}
                    else:
                        st.warning("未检测到微调模型配置文件")
                else:
                    st.warning("微调模型路径不存在")
            
            # 按类型分组显示模型
            asr_models = [model for model in model_names.keys() 
                         if get_model_type(model_names[model] if model == "finetune_model" else model) == "asr"]
            vad_models = [model for model in model_names.keys() 
                         if get_model_type(model_names[model] if model == "finetune_model" else model) == "vad"]
            punc_models = [model for model in model_names.keys() 
                          if get_model_type(model_names[model] if model == "finetune_model" else model) == "punc"]
            spk_models = [model for model in model_names.keys() 
                         if get_model_type(model_names[model] if model == "finetune_model" else model) == "spk"]
            
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
                capability_maps = provider_config["models"]["capability_maps"]
                for model in st.session_state.selected_models:
                    capabilities.update(capability_maps.get(model, []))
                st.markdown("##### 已选模型能力")
                st.write(", ".join(sorted(capabilities)))
            
            # 启动和关按钮
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
