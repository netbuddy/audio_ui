from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
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
from sse_starlette.sse import EventSourceResponse
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 添加控制台处理器
    ]
)
logger = logging.getLogger(__name__)

# 确保 FastAPI 的日志也被配置
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = []  # 清除现有的处理器
uvicorn_logger.addHandler(logging.StreamHandler())
uvicorn_logger.setLevel(logging.INFO)

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

# 在件开头的其他 BaseModel 类定义旁边添加
class VADConfig(BaseModel):
    """VAD 配置"""
    model: str = "fsmn-vad"
    model_revision: str = "v2.0.4"
    min_duration_on: float = 0.02  # 最小语音段长度(秒)
    min_duration_off: float = 0.8  # 最小静音段长度(秒)
    window_size: int = 200  # 窗口大小(帧)
    max_end_silence: float = 0.8  # 最大结尾静音长度(秒)

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
        self.vad_model = None
        self.streaming_model = None
        self.accept_ws_connections = False  # 添加标志控制是否接受 WebSocket 连接
        
    async def start(self, config: ModelConfig) -> Dict:
        """启动服务"""
        try:
            if self.is_running:
                return {"status": "success", "message": "Service is already running"}
                
            # ��建模型参数
            model_kwargs = {
                "disable_update": True
            }
            
            # 处理 ASR 模型
            if config.model:
                # 检查是否是本地微调模型路径
                if os.path.isdir(config.model):
                    model_kwargs["model"] = config.model
                    model_kwargs["model_revision"] = "local"
                else:
                    # 使用预训练模型
                    model_name = audio_model_name_maps.get(config.model)
                    if not model_name:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid model name: {config.model}"
                        )
                    model_kwargs["model"] = model_name
                    model_kwargs["model_revision"] = config.model_revision
            
            # 添加可选模型
            if config.vad_model:
                model_kwargs["vad_model"] = audio_model_name_maps.get(config.vad_model)
                model_kwargs["vad_model_revision"] = config.vad_model_revision
                
            if config.punc_model:
                model_kwargs["punc_model"] = audio_model_name_maps.get(config.punc_model)
                model_kwargs["punc_model_revision"] = config.punc_model_revision
                
            if config.spk_model:
                model_kwargs["spk_model"] = audio_model_name_maps.get(config.spk_model)
                model_kwargs["spk_model_revision"] = config.spk_model_revision
            
            # 初始化模型
            self.model = AutoModel(**model_kwargs)
            self.config = config
            self.is_running = True
            self.accept_ws_connections = True  # 允许接受 WebSocket 连接
            
            return {"status": "success", "message": "Service started successfully"}
            
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def stop(self) -> Dict:
        """停止服务"""
        try:
            if not self.is_running:
                return {"status": "success", "message": "Service is not running"}
            
            # 清理模型资源
            if self.model:
                del self.model
                self.model = None
            if self.streaming_model:
                del self.streaming_model
                self.streaming_model = None
                
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

    async def start_vad(self, config: VADConfig) -> Dict:
        """启动 VAD 服务"""
        try:
            if self.vad_model:
                return {"status": "success", "message": "VAD service is already running"}
                
            # 初始化 VAD 模型
            model_name = audio_model_name_maps.get(config.model)
            self.vad_model = AutoModel(
                model=model_name,
                model_revision=config.model_revision,
                disable_update=True
            )
            
            return {"status": "success", "message": "VAD service started successfully"}
            
        except Exception as e:
            logger.error(f"Error starting VAD service: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def stop_vad(self) -> Dict:
        """停止 VAD 服务"""
        try:
            if not self.vad_model:
                return {"status": "success", "message": "VAD service is not running"}
            
            del self.vad_model
            self.vad_model = None
            
            return {"status": "success", "message": "VAD service stopped successfully"}
            
        except Exception as e:
            logger.error(f"Error stopping VAD service: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def process_vad(self, audio_file: UploadFile, config: VADConfig) -> Dict:
        """处理 VAD 请求"""
        input_path = None
        converted_path = None
        
        try:
            if not self.vad_model:
                raise HTTPException(
                    status_code=400,
                    detail="VAD service is not running"
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
            result = self.vad_model.generate(
                input=converted_path,
                min_duration_on=config.min_duration_on,
                min_duration_off=config.min_duration_off,
                window_size=config.window_size,
                max_end_silence=config.max_end_silence
            )
            
            # 处理结果
            processed_result = self._process_vad_result(result)
            
            return {
                "status": "success",
                "result": processed_result
            }
            
        except Exception as e:
            logger.error(f"Error processing VAD: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # 清理临时文件
            for path in [input_path, converted_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Error removing temporary file {path}: {e}")
                        
    def _process_vad_result(self, result: List) -> Dict:
        """处理 VAD 结果"""
        if not result:
            return {"segments": []}
            
        return {
            "segments": result[0].get("value", [])
        }

    async def handle_stream(self, websocket: WebSocket):
        """处理流式音频识别"""
        try:
            await websocket.accept()
            logger.info("Client connected")
            
            # 初始化流式识别模型
            model = AutoModel(
                model="paraformer-zh-streaming", 
                model_revision="v2.0.4"
            )
            cache = {}
            
            # 流式识别参数
            chunk_size = [0, 10, 5]  # 600ms
            encoder_chunk_look_back = 4
            decoder_chunk_look_back = 1
            chunk_stride = chunk_size[1] * 960
            
            while True:
                try:
                    # 接收音频数据
                    audio_data = await websocket.receive_bytes()
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # 分块处理音频
                    total_chunk_num = int(len(audio_array - 1) / chunk_stride + 1)
                    for i in range(total_chunk_num):
                        speech_chunk = audio_array[i * chunk_stride:(i + 1) * chunk_stride]
                        is_final = i == total_chunk_num - 1
                        
                        # 识别当前块
                        result = model.generate(
                            input=speech_chunk,
                            cache=cache,
                            is_final=is_final,
                            chunk_size=chunk_size,
                            encoder_chunk_look_back=encoder_chunk_look_back,
                            decoder_chunk_look_back=decoder_chunk_look_back
                        )
                        
                        # 发送识别结果
                        if result:
                            await websocket.send_json({
                                "text": result[0]["text"],
                                "is_final": is_final
                            })
                            
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close()

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

# 添加请求体模型
class StreamRequest(BaseModel):
    """流式识别请求"""
    url: str

# 添加代理配置
class ProxyConfig(BaseModel):
    """代理配置"""
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    socks_proxy: Optional[str] = None

# 从环境变量加载代理配置
def get_proxy_config() -> ProxyConfig:
    """获取代理配置"""
    return ProxyConfig(
        http_proxy=os.getenv("HTTP_PROXY"),
        https_proxy=os.getenv("HTTPS_PROXY"),
        socks_proxy=os.getenv("SOCKS_PROXY")
    )

@app.post("/recognize_stream")
async def recognize_stream(request: StreamRequest):
    """从流媒体URL实时识别音频"""
    try:
        logger.info(f"Starting stream recognition for URL: {request.url}")
        
        # 获取代理配置
        proxy_config = get_proxy_config()
        logger.info(f"Proxy config: {proxy_config}")
        
        # 构建 ffmpeg 命令
        ffmpeg_cmd = [
            'ffmpeg',
            '-re',              # 实时模式
        ]
        
        # 添加代理参数
        if proxy_config.http_proxy:
            ffmpeg_cmd.extend([
                '-http_proxy', proxy_config.http_proxy
            ])
        if proxy_config.https_proxy:
            ffmpeg_cmd.extend([
                '-https_proxy', proxy_config.https_proxy
            ])
        if proxy_config.socks_proxy:
            ffmpeg_cmd.extend([
                '-socks_proxy', proxy_config.socks_proxy
            ])
            
        # 添加其他参数
        ffmpeg_cmd.extend([
            '-i', request.url,  # 输入URL
            '-vn',             # 不处理视频
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            'pipe:1',          # 输出到管道
        ])
        
        logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        try:
            # 启动 ffmpeg 进程
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.info(f"FFmpeg process started with PID: {process.pid}")
            
            # 检查进程是否成功启动
            if process.returncode is not None:
                error_msg = await process.stderr.read()
                logger.error(f"FFmpeg process failed to start: {error_msg.decode()}")
                raise Exception(f"FFmpeg failed to start: {error_msg.decode()}")
                
        except Exception as e:
            logger.error(f"Error starting FFmpeg process: {e}")
            raise

        # 创建 SSE 响应
        async def event_generator():
            chunk_size = 32000  # 2秒的音频数据
            cache = {}
            has_data = False
            total_bytes = 0
            
            try:
                logger.info("Starting to read audio data...")
                while True:
                    try:
                        chunk = await process.stdout.read(chunk_size)
                        if not chunk:
                            logger.info("End of audio stream reached")
                            break
                            
                        total_bytes += len(chunk)
                        has_data = True
                        logger.debug(f"Read {len(chunk)} bytes (total: {total_bytes})")
                        
                        # 使用模型处理音频块
                        result = service.model.generate(
                            input=chunk,
                            cache=cache,
                            is_final=False,
                            chunk_size=[0, 10, 5],
                            encoder_chunk_look_back=4,
                            decoder_chunk_look_back=1
                        )
                        
                        if result:
                            logger.debug(f"Generated result: {result}")
                            yield {
                                "event": "transcript",
                                "data": json.dumps({
                                    "status": "success",
                                    "result": result
                                })
                            }
                            
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        # 检查 ffmpeg 进程状态
                        if process.returncode is not None:
                            error_msg = await process.stderr.read()
                            logger.error(f"FFmpeg process error: {error_msg.decode()}")
                        break
                
                # 只在有数据时处理最后一块
                if has_data and cache:
                    logger.info("Processing final chunk...")
                    try:
                        final_result = service.model.generate(
                            input=b"",
                            cache=cache,
                            is_final=True
                        )
                        
                        if final_result:
                            logger.info(f"Final result: {final_result}")
                            yield {
                                "event": "transcript",
                                "data": json.dumps({
                                    "status": "success",
                                    "result": final_result,
                                    "is_final": True
                                })
                            }
                    except Exception as e:
                        logger.error(f"Error processing final chunk: {e}")
                        
            finally:
                logger.info("Cleaning up FFmpeg process...")
                process.terminate()
                await process.wait()
                logger.info(f"FFmpeg process terminated with return code: {process.returncode}")
                
            # 如果没有数据，返回错误信息
            if not has_data:
                logger.error("No audio data received from URL")
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "status": "error",
                        "message": "No audio data received from URL"
                    })
                }

        return EventSourceResponse(event_generator())
        
    except Exception as e:
        logger.error(f"Stream recognition error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    """处理流式音频识别"""
    try:
        await websocket.accept()
        logger.info("Client connected")
        
        # 初始化流式识别模型
        model = AutoModel(
            model="paraformer-zh-streaming", 
            model_revision="v2.0.4"
        )
        cache = {}
        
        # 流式识别参数
        chunk_size = [0, 10, 5]  # 600ms
        encoder_chunk_look_back = 4
        decoder_chunk_look_back = 1
        chunk_stride = chunk_size[1] * 960
        
        while True:
            try:
                # 接收音频数据
                audio_data = await websocket.receive_bytes()
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # 分块处理音频
                total_chunk_num = int(len(audio_array - 1) / chunk_stride + 1)
                for i in range(total_chunk_num):
                    speech_chunk = audio_array[i * chunk_stride:(i + 1) * chunk_stride]
                    is_final = i == total_chunk_num - 1
                    
                    # 识别当前块
                    result = model.generate(
                        input=speech_chunk,
                        cache=cache,
                        is_final=is_final,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=encoder_chunk_look_back,
                        decoder_chunk_look_back=decoder_chunk_look_back
                    )
                    
                    # 发送识别结果
                    if result:
                        await websocket.send_json({
                            "text": result[0]["text"],
                            "is_final": is_final
                        })
                        
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

@app.post("/vad/start")
async def start_vad_service(config: VADConfig = VADConfig()):
    """启动 VAD 服务"""
    return await service.start_vad(config)

@app.post("/vad/stop")
async def stop_vad_service():
    """停止 VAD 服务"""
    return await service.stop_vad()

@app.post("/vad/detect")
async def detect_vad(
    audio_file: UploadFile = File(...),
    config: VADConfig = VADConfig()
):
    """处理 VAD 请求"""
    return await service.process_vad(audio_file, config)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
