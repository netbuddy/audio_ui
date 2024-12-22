import streamlit as st
import os
from litellm import completion
import litellm
from comp_audio_operation import format_text_with_options
import re

def get_speakers_from_text(text: str) -> set:
    """从文本中提取所有说话人ID"""
    speakers = set()
    # 匹配 [说话人 数字] 格式
    pattern = r'\[说话人 (\d+)\]'
    matches = re.findall(pattern, text)
    return set(matches)

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
                temperature=0.1,
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
                temperature=0.1,
                max_tokens=model_max_tokens,
                api_base=os.getenv("OLLAMA_HOST"),  # 使用环境变量中的地址
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

def render_text_postprocess():
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
    st.subheader("音频文本")
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
