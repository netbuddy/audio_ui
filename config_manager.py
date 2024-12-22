import streamlit as st
import json
from typing import Dict, List

class ModelConfig:
    """模型配置管理类"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._config = self._load_config()
            self._initialized = True
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open('models.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"加载配置失败: {str(e)}")
            return {}
    
    @property
    def llm_providers(self) -> Dict:
        """获取 LLM 供应商配置"""
        return self._config.get("LLM_PROVIDERS", {})
    
    @property
    def audio_models(self) -> Dict:
        """获取音频模型配置"""
        return self._config.get("AUDIO_MODELS", {})
    
    def get_model_revisions(self) -> Dict:
        """获取模型版本信息"""
        return self.audio_models.get("model_revisions", {})
    
    def get_model_names(self) -> Dict:
        """获取模型名称映射"""
        return self.audio_models.get("name_maps", {})
    
    def get_model_capabilities(self) -> List:
        """获取模型能力列表"""
        return self.audio_models.get("capabilities", [])
    
    def get_model_capability_maps(self) -> Dict:
        """获取模型能力映射"""
        return self.audio_models.get("capability_maps", {})

# 创建全局配置实例
config = ModelConfig() 