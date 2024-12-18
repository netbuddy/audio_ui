from fastapi import FastAPI, HTTPException
import docker
import time
import logging
from typing import Dict
import subprocess
import os
import asyncio

app = FastAPI()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRServiceManager:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.container_name = "funasr-offline-gpu"
        self.container = None
        self.service_running = False
    
    def get_container(self) -> docker.models.containers.Container:
        """获取或创建容器"""
        try:
            return self.docker_client.containers.get(self.container_name)
        except docker.errors.NotFound:
            logger.info(f"Container {self.container_name} not found, creating new container...")
            try:
                # 创建容器的配置
                container_config = {
                    'image': 'registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-gpu-0.2.0',
                    'name': self.container_name,
                    'ports': {'10095/tcp': 10098},
                    'volumes': {
                        '/sata/docker_data/funasr-runtime-resources/models': {
                            'bind': '/workspace/models',
                            'mode': 'rw'
                        }
                    },
                    'device_requests': [
                        docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                    ],
                    'privileged': True,
                    'detach': True,
                    'tty': True
                }
                
                # 创建并启动容器
                container = self.docker_client.containers.run(**container_config)
                logger.info(f"Container {self.container_name} created successfully")
                return container
                
            except Exception as e:
                error_msg = f"Failed to create container: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

    def is_service_running(self) -> bool:
        """检查服务是否运行"""
        try:
            container = self.get_container()
            # 修改进程检查命令
            exec_result = container.exec_run(
                ["/bin/bash", "-c", "ps -ef | grep run_server.sh | grep -v grep"]
            )
            # 检查命令输出是否包含进程信息
            output = exec_result.output.decode()
            logger.info(f"Process check result: {output}")
            return len(output.strip()) > 0 and "run_server.sh" in output
            
        except Exception as e:
            logger.error(f"Error checking service status: {e}")
            return False

    async def start_service(self) -> Dict[str, str]:
        """启动语音识别服务"""
        try:
            if self.is_service_running():
                return {"status": "success", "message": "Service is already running"}

            container = self.get_container()
            
            # 1. 安装 supervisor - 同步执行并检查结果
            install_cmd = ["/bin/bash", "-c", "apt-get update && apt-get install -y supervisor"]
            result = container.exec_run(
                cmd=install_cmd,
                user="root",
                detach=False  # 同步执行
            )
            if result.exit_code != 0:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to install supervisor: {result.output.decode()}"
                )
            
            # 2. 写入配置文件 - 同步执行并检查结果
            supervisor_conf = """
[program:asr_service]
command=bash run_server.sh --download-model-dir /workspace/models --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst --itn-dir thuduj12/fst_itn_zh --hotword /workspace/models/hotwords.txt
directory=/workspace/FunASR/runtime
autostart=true
autorestart=true
stderr_logfile=/var/log/asr_service.err.log
stdout_logfile=/var/log/asr_service.out.log
"""
            cmd = f'bash -c \'echo "{supervisor_conf}" > /etc/supervisor/conf.d/asr_service.conf\''
            result = container.exec_run(cmd, user="root", detach=False)
            if result.exit_code != 0:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to write config: {result.output.decode()}"
                )
            
            # 3. 启动 supervisor 服务 - 同步执行每个步骤
            commands = [
                # 清理所有相关进程
                "pkill -f supervisord || true",  # 停止 supervisord
                "pkill -f funasr-wss-server || true",  # 停止 ASR 服务
                "pkill -f funasr_progress.py || true",  # 停止进度监控
                "rm -f /var/run/supervisor.sock || true",  # 删除 supervisor socket 文件
                "rm -f /tmp/supervisor.sock || true",  # 删除可能存在的临时 socket 文件
                # 启动新的 supervisord
                "supervisord -c /etc/supervisor/supervisord.conf",
                "supervisorctl reread",
                "supervisorctl update",
                "supervisorctl start asr_service"
            ]
            
            for cmd in commands:
                result = container.exec_run(
                    ["/bin/bash", "-c", cmd],  # 使用 bash -c 执行命令
                    user="root",
                    detach=False
                )
                if result.exit_code != 0 and "pkill" not in cmd and "rm" not in cmd:  # 忽略清理命令的错误
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to execute {cmd}: {result.output.decode()}"
                    )
                # 在清理命令后添加短暂延时，确保进程完全终止
                if "pkill" in cmd:
                    await asyncio.sleep(1)
            
            # 等待服务启动，最多等待60秒
            max_wait_time = 60  # 最大等待时间
            start_time = time.time()
            
            while True:
                # 检查是否超时
                if time.time() - start_time > max_wait_time:
                    raise HTTPException(
                        status_code=500,
                        detail="Service startup timeout after 60 seconds"
                    )
                
                # 检查 supervisor 状态
                status_result = container.exec_run(
                    "supervisorctl status asr_service",
                    user="root",
                    detach=False
                )
                status_output = status_result.output.decode()
                logger.info(f"Current service status: {status_output}")
                
                if "RUNNING" in status_output:
                    return {"status": "success", "message": "Service started successfully"}
                elif "FATAL" in status_output or "ERROR" in status_output:
                    # 检查错误日志
                    log_result = container.exec_run("tail -n 20 /var/log/asr_service.err.log")
                    error_log = log_result.output.decode()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Service failed to start. Error log:\n{error_log}"
                    )
                
                # 如果状态是 STARTING，继续等待
                await asyncio.sleep(2)  # 每2秒检查一次
            
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def stop_service(self) -> Dict[str, str]:
        """停止语音识别服务"""
        try:
            if not self.is_service_running():
                return {"status": "success", "message": "Service is not running"}

            container = self.get_container()
            
            # 使用 supervisor 停止服务
            container.exec_run("supervisorctl stop asr_service", user="root")
            
            # 等待服务停止，最多等待30秒
            max_wait_time = 30  # 最大等待时间
            start_time = time.time()
            
            while True:
                # 检查是否超时
                if time.time() - start_time > max_wait_time:
                    raise HTTPException(
                        status_code=500,
                        detail="Service shutdown timeout after 30 seconds"
                    )
                
                # 检查 supervisor 状态
                status_result = container.exec_run(
                    "supervisorctl status asr_service",
                    user="root",
                    detach=False
                )
                status_output = status_result.output.decode()
                logger.info(f"Current service status: {status_output}")
                
                if "STOPPED" in status_output:
                    return {"status": "success", "message": "Service stopped successfully"}
                elif "FATAL" in status_output:
                    return {"status": "success", "message": "Service stopped (with errors)"}
                
                # 如果还在运行或停止过程中，继续等待
                await asyncio.sleep(1)  # 每秒检查一次
                
        except Exception as e:
            logger.error(f"Error stopping service: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# 创建服务管理器实例
service_manager = ASRServiceManager()

@app.post("/start")
async def start_service():
    """启动语音识别服务"""
    return await service_manager.start_service()

@app.post("/stop")
async def stop_service():
    """停止语音识别服务"""
    return await service_manager.stop_service()

@app.get("/status")
async def get_status():
    """获取服务状态"""
    is_running = service_manager.is_service_running()
    return {
        "status": "running" if is_running else "stopped",
        "message": "Service is running" if is_running else "Service is not running"
    }

@app.get("/status/{model_index}")
async def get_model_status(model_index: int):
    """获取指定模型的状态"""
    if model_index < 0 or model_index > 2:
        raise HTTPException(status_code=400, detail="Invalid model index")
        
    is_running = service_manager.is_service_running()
    return {
        "status": "running" if is_running else "stopped",
        "message": f"Model {model_index} is {'running' if is_running else 'not running'}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10097) 