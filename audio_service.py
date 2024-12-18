from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import os
import logging
from funasr import AutoModel
import tempfile
import asyncio
from enum import Enum
import wave
import numpy as np
from queue import Queue
from threading import Lock
import json
import time
from model_metadata import audio_model_name_maps
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    """模型类型"""
    ASR = "asr"
    VAD = "vad"
    PUNC = "punc"
    SPK = "spk"

class ModelConfig(BaseModel):
    """模型配置"""
    # model是必选参数
    model: str
    model_revision: Optional[str] = "v2.0.4"
    vad_model: Optional[str] = None
    vad_model_revision: Optional[str] = "v2.0.4"
    punc_model: Optional[str] = None
    punc_model_revision: Optional[str] = "v2.0.4"
    spk_model: Optional[str] = None
    spk_model_revision: Optional[str] = "v2.0.2"

class AudioFormat(str, Enum):
    """支持的音频格式"""
    WAV = "wav"
    PCM = "pcm"
    MP3 = "mp3"

class RecognitionConfig(BaseModel):
    """识别配置"""
    batch_size_s: int = 300
    hotword: Optional[str] = None
    use_timestamp: bool = False
    use_itn: bool = True
    max_single_segment: int = 10000

class StreamingConfig(BaseModel):
    """流式识别配置"""
    chunk_size: List[int] = [0, 10, 5]  # [0, 10, 5] 600ms
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 1
    sample_rate: int = 16000
    hotword: Optional[str] = None
    use_itn: bool = True

class StreamingSession:
    """流式会话管理"""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.cache = {}
        self.audio_buffer = []
        self.is_final = False
        self.chunk_stride = 0
        self.start_time = time.time()

class ASRService:
    def __init__(self):
        self.model = None
        self.is_running = False
        self.config = None
        self.temp_dir = tempfile.mkdtemp()
        self.supported_formats = {".wav", ".mp3", ".pcm"}
        self.streaming_sessions = {}
        self.session_lock = Lock()
        
    async def start(self, config: ModelConfig) -> Dict:
        """启动服务"""
        try:
            if self.is_running:
                return {"status": "success", "message": "Service is already running"}
                
            # 构建模型参数
            # 将config.model作为key在model_metadata.py的audio_model_name_maps中获取model_name
            model_name = audio_model_name_maps.get(config.model)
            model_kwargs = {
                "model": model_name,
                "model_revision": config.model_revision,
                "disable_update": True
            }
            
            # 添加可选模型
            if config.vad_model:
                model_kwargs["vad_model"] = config.vad_model
                model_kwargs["vad_model_revision"] = config.vad_model_revision
                
            if config.punc_model:
                model_kwargs["punc_model"] = config.punc_model
                model_kwargs["punc_model_revision"] = config.punc_model_revision
                
            if config.spk_model:
                model_kwargs["spk_model"] = config.spk_model
                model_kwargs["spk_model_revision"] = config.spk_model_revision
            
            # 初始化模型
            self.model = AutoModel(**model_kwargs)
            self.config = config
            self.is_running = True
            
            return {"status": "success", "message": "Service started successfully"}
            
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def stop(self) -> Dict:
        """停止服务"""
        try:
            if not self.is_running:
                return {"status": "success", "message": "Service is not running"}
            
            # 需要删除self.model
            del self.model
            self.model = None
            self.is_running = False
            self.config = None
            
            return {"status": "success", "message": "Service stopped successfully"}
            
        except Exception as e:
            logger.error(f"Error stopping service: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def _convert_audio(self, input_path: str) -> str:
        """转换音频格式为 wav"""
        try:
            import ffmpeg
            output_path = os.path.join(self.temp_dir, "converted.wav")
            
            # 使用 ffmpeg 转换音频
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path,
                                 acodec='pcm_s16le',
                                 ac=1,
                                 ar=16000)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to convert audio format"
            )
            
    def _validate_audio_file(self, filename: str):
        """验证音频文件格式"""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {ext}. Supported formats: {', '.join(self.supported_formats)}"
            )
            
    async def process_audio(self, audio_file: UploadFile, config: RecognitionConfig) -> Dict:
        """处理音频文件"""
        input_path = None
        converted_path = None
        
        try:
            if not self.is_running:
                raise HTTPException(
                    status_code=400,
                    detail="Service is not running"
                )
                
            # 验证文件格式
            self._validate_audio_file(audio_file.filename)
                
            # 保存上传的文件
            input_path = os.path.join(self.temp_dir, audio_file.filename)
            with open(input_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
                
            # 转换音频格式
            converted_path = await self._convert_audio(input_path)
                
            # 处理音频
            result = self.model.generate(
                input=converted_path,
                batch_size_s=config.batch_size_s,
                hotword=config.hotword if config.hotword else None,
                use_timestamp=config.use_timestamp,
                use_itn=config.use_itn
            )
            
            # 处理结果
            processed_result = self._process_result(result)
            
            return {
                "status": "success",
                "result": processed_result
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # 清理临时文件
            for path in [input_path, converted_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Error removing temporary file {path}: {e}")
                        
    def _process_result(self, result: List) -> Dict:
        """处理识别结果"""
        if not result:
            return {"text": "", "segments": []}
            
        processed = {
            "text": result[0].get("text", ""),
            #timestamp格式为[[70, 230], [230, 410], [410, 530], [530, 650], [650, 770], [770, 1010], [1090, 1250], [13450, 13690], [14110, 14290], [14290, 14410], [14410, 14650], [14670, 14890], ...]
            "timestamp": result[0].get("timestamp", []),
            "segments": []
        }
        
        # 处理分段信息
        if "sentence_info" in result[0]:
            for segment in result[0]["sentence_info"]:
                processed["segments"].append({
                    "text": segment.get("text", ""),
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "speaker": segment.get("spk", "")
                })
                
        return processed

    async def handle_websocket(self, websocket: WebSocket, config: StreamingConfig):
        """处理 WebSocket 连接"""
        try:
            await websocket.accept()
            session = StreamingSession(websocket)
            session.chunk_stride = config.chunk_size[1] * 960  # 600ms
            
            with self.session_lock:
                self.streaming_sessions[id(websocket)] = session
                
            try:
                while True:
                    message = await websocket.receive()
                    
                    if message["type"] == "websocket.disconnect":
                        break
                        
                    if message["type"] == "websocket.receive":
                        if "text" in message:
                            # 处理控制消息
                            await self._handle_control_message(session, message["text"])
                        elif "bytes" in message:
                            # 处理音频数据
                            await self._handle_streaming_audio(session, message["bytes"], config)
                            
            finally:
                with self.session_lock:
                    if id(websocket) in self.streaming_sessions:
                        del self.streaming_sessions[id(websocket)]
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close()
            
    async def _handle_control_message(self, session: StreamingSession, message: str):
        """处理控制消息"""
        try:
            data = json.loads(message)
            if data.get("type") == "end":
                session.is_final = True
                # 处理剩余的音频数据
                if session.audio_buffer:
                    await self._process_streaming_chunk(session, b''.join(session.audio_buffer), True)
                    session.audio_buffer.clear()
                    
        except Exception as e:
            logger.error(f"Error handling control message: {e}")
            
    async def _handle_streaming_audio(self, session: StreamingSession, audio_data: bytes, config: StreamingConfig):
        """处理流式音频数据"""
        try:
            session.audio_buffer.append(audio_data)
            buffer_size = sum(len(chunk) for chunk in session.audio_buffer)
            
            # 当缓冲区达到指定大小时处理
            if buffer_size >= session.chunk_stride:
                audio_chunk = b''.join(session.audio_buffer)
                session.audio_buffer.clear()
                
                await self._process_streaming_chunk(session, audio_chunk, session.is_final)
                
        except Exception as e:
            logger.error(f"Error handling streaming audio: {e}")
            
    async def _process_streaming_chunk(self, session: StreamingSession, audio_chunk: bytes, is_final: bool):
        """处理音频数据块"""
        try:
            # 转换音频数据
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # 使用模型处理
            result = self.model.generate(
                input=audio_array,
                cache=session.cache,
                is_final=is_final,
                chunk_size=config.chunk_size,
                encoder_chunk_look_back=config.encoder_chunk_look_back,
                decoder_chunk_look_back=config.decoder_chunk_look_back
            )
            
            # 发送结果
            if result:
                await session.websocket.send_json({
                    "status": "success",
                    "is_final": is_final,
                    "result": self._process_result(result)
                })
                
        except Exception as e:
            logger.error(f"Error processing streaming chunk: {e}")
            await session.websocket.send_json({
                "status": "error",
                "message": str(e)
            })

# 创建 FastAPI 应用
app = FastAPI(title="FunASR Service")
service = ASRService()

@app.post("/start")
async def start_service(config: ModelConfig):
    """启动服务"""
    return await service.start(config)

@app.post("/stop")
async def stop_service():
    """停止服务"""
    return await service.stop()

@app.get("/status")
async def get_status():
    """获取服务状态"""
    return {
        "status": "running" if service.is_running else "stopped",
        "config": service.config.dict() if service.config else None
    }

@app.post("/recognize")
async def recognize_audio(
    audio_file: UploadFile = File(...),
    config: RecognitionConfig = RecognitionConfig()
):
    """处理音频文件"""
    return await service.process_audio(audio_file, config)

@app.get("/models")
async def list_models():
    """获取支持的模型列表"""
    return {
        "asr_models": [
            {"name": "paraformer-zh", "description": "中文语音识别"},
            {"name": "paraformer-zh-streaming", "description": "中文流式识别"}
        ],
        "vad_models": [
            {"name": "fsmn-vad", "description": "语音活动检测"}
        ],
        "punc_models": [
            {"name": "ct-punc", "description": "标点恢复"},
            {"name": "ct-punc-c", "description": "中文标点恢复"}
        ],
        "spk_models": [
            {"name": "cam++", "description": "说话人分离"}
        ]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, config: StreamingConfig = StreamingConfig()):
    """WebSocket 端点"""
    await service.handle_websocket(websocket, config)

@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    return {
        "active_sessions": len(service.streaming_sessions),
        "uptime": time.time() - service.start_time if service.is_running else 0,
        "total_requests": service.total_requests,
        "error_count": service.error_count
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)