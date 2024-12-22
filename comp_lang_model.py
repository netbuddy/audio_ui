import streamlit as st
import os
from typing import Optional, List, Dict
from config_manager import config

def render_lang_model_sidebar():
    """渲染语言模型侧边栏"""
    with st.container():
        st.subheader("语言模型")
        
        # 获取已配置的供应商
        available_providers = config.llm_providers
        available_providers_name = available_providers.keys()
        
        if not available_providers_name:
            st.warning("未找到已配置的模型供应商。请在 models.json 文件中配置相关参数。")
        else:
            # 模型供应商选择
            st.markdown("##### 模型供应商")
            selected_provider = st.selectbox(
                "",
                options=["请选择"] + list(available_providers_name),
                key="llm_provider"
            )
            
            # 模型列表选择
            st.markdown("##### 模型列表")
            if selected_provider and selected_provider != "请选择":
                available_models = available_providers[selected_provider]["models"]
                selected_model = st.selectbox(
                    "",
                    options=["请选择"] + available_models,
                    key="llm_model"
                )
                
                # 显示当前配置
                if selected_model and selected_model != "请选择":
                    st.markdown("##### 当前配置")
                    env_var = available_providers[selected_provider]['env_vars'][0]
                    config_md = f"""
                    - **供应商**: {selected_provider}
                    - **模型**: {selected_model}
                    - **接口地址**: {os.getenv(env_var)}
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
            st.text_area(
                "系统提示词",
                value=system_prompt_text,
                key="system_prompt_input",
                height=100
            ) 