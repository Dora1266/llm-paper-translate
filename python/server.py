import os
import re
import time
import json
import uuid
import queue
import threading
from flask import Flask, request, jsonify, Response
from lmdeploy import pipeline, TurbomindEngineConfig
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("main")

app = Flask(__name__)

# 全局配置
CONFIG = {
    "model_path": "D:\\tran\\models\\DeepSeek-R1-Distill-Qwen-1.5B",
    "max_workers": 10,
    "max_batch_size": 10,
    "session_len": 2048,
    "max_tokens": 2048,
    "temperature": 0.7,
    "port": 8000,
    "request_timeout": 60,
    "max_queue_size": 50
}

# 全局变量来存储模型管道
global_pipe = None

# 添加线程锁来保护模型访问
model_lock = threading.RLock()

# 请求队列
request_queue = queue.Queue(maxsize=CONFIG["max_queue_size"])

# 请求处理工作线程池
worker_threads = []


def initialize_model():
    """初始化并加载模型"""
    global global_pipe

    logger.info(f"正在加载模型: {CONFIG['model_path']}...")

    # 配置引擎 - 调整参数以适应更长的对话历史
    engine_config = TurbomindEngineConfig(
        max_batch_size=CONFIG["max_batch_size"],
        session_len=CONFIG["session_len"],
        quant_policy=0,
        cache_max_entry_count=0.2
    )

    try:
        # 初始化模型
        global_pipe = pipeline(
            CONFIG["model_path"],
            backend_config=engine_config
        )
        logger.info("模型加载完成！服务器准备就绪。")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def process_request(request_id, chat_history, generation_config=None):
    """处理单个请求并生成回复(非流式)"""
    global global_pipe

    logger.info(f"开始处理请求 {request_id}, 聊天历史长度: {len(chat_history)}")
    # 记录聊天历史用于调试
    for i, msg in enumerate(chat_history):
        logger.info(f"  消息 {i + 1}: role={msg.get('role', 'unknown')}, content={msg.get('content', '')[:5000]}...")

    start_time = time.time()

    try:
        # 设置默认生成参数
        gen_config = {
            "max_new_tokens": min(CONFIG["max_tokens"], 1024),  # 确保不超过1024
            "temperature": CONFIG["temperature"],
            "top_p": 0.8,  # 添加top_p采样
            "repetition_penalty": 1.1  # 添加重复惩罚
        }

        # 更新用户指定的生成参数
        if generation_config:
            gen_config.update(generation_config)

        # 使用锁保护模型访问
        with model_lock:
            # 调用模型生成回复
            try:
                # 尝试直接调用
                response = global_pipe([chat_history])
                if isinstance(response, str):
                    response_text = response
                elif hasattr(response, 'text'):
                    response_text = response.text
                else:
                    # 可能是迭代器
                    response_text = ""
                    for chunk in response:
                        if isinstance(chunk, str):
                            response_text += chunk
                        elif hasattr(chunk, 'text'):
                            response_text += chunk.text
                        elif isinstance(chunk, dict) and 'text' in chunk:
                            response_text += chunk['text']
            except Exception as model_error:
                logger.error(f"首次调用模型失败: {str(model_error)}, 尝试其他方法")
                # 尝试使用stream_infer
                try:
                    response_text = ""
                    for chunk in global_pipe.stream_infer([chat_history], **gen_config):
                        if hasattr(chunk, 'text'):
                            response_text += chunk.text
                except Exception as stream_error:
                    logger.error(f"流式调用也失败: {str(stream_error)}")
                    # 尝试使用最简单的调用
                    response_text = global_pipe.infer([chat_history], **gen_config)

        # 确保我们有文本响应
        if not isinstance(response_text, str):
            if hasattr(response_text, 'text'):
                response_text = response_text.text
            elif hasattr(response_text, '__str__'):
                response_text = str(response_text)
            else:
                response_text = "无法从模型获取有效响应。"

        # 记录处理时间
        elapsed = time.time() - start_time
        logger.info(f"请求 {request_id} 处理完成，用时: {elapsed:.2f}秒，生成了 {len(response_text)} 个字符")

        return {
            "success": True,
            "text": response_text,
            "request_time": elapsed
        }
    except Exception as e:
        logger.error(f"处理请求 {request_id} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "request_time": time.time() - start_time
        }


def worker_function():
    """工作线程函数，不断从队列中获取请求并处理"""
    while True:
        job = None
        try:
            # 从队列获取请求
            job = request_queue.get()
            if job is None:  # 结束信号
                break

            request_id, chat_history, generation_config, result_queue = job

            # 处理请求
            result = process_request(request_id, chat_history, generation_config)

            # 将结果放入结果队列
            result_queue.put(result)
        except Exception as e:
            logger.error(f"工作线程处理出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 如果有结果队列，返回错误结果
            if job and len(job) >= 4:
                result_queue = job[3]
                result_queue.put({
                    "success": False,
                    "error": f"工作线程错误: {str(e)}",
                    "request_time": 0
                })
        finally:
            if job is not None:
                request_queue.task_done()


def format_message_for_model(role, content):
    """根据角色格式化消息内容"""
    if role == "system":
        return {"role": "user", "content": f"<system>\n{content}\n</system>"}
    else:
        return {"role": role, "content": content}


def prepare_chat_history(messages):
    chat_history = []

    try:
        # 系统消息特殊处理
        has_system = False
        for msg in messages:
            if msg.get('role') == 'system':
                has_system = True
                break

        for i, msg in enumerate(messages):
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Invalid message format at index {i}")

            role = msg['role']
            content = msg['content']

            # 角色映射
            if role == "system":
                # 将系统提示添加为特殊格式的用户消息
                chat_history.append(format_message_for_model("system", content))
                # 如果系统消息不是第一条，则不添加助手响应
                if i == 0:
                    chat_history.append({"role": "assistant", "content": "我会按照您的指示行动。"})
            elif role in ["user", "assistant"]:
                # 直接添加用户和助手消息
                chat_history.append({"role": role, "content": content})
            else:
                # 不支持的角色，视为用户消息
                logger.warning(f"不支持的角色 '{role}'，将作为用户消息处理")
                chat_history.append({"role": "user", "content": content})
    except Exception as e:
        logger.error(f"处理聊天历史时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    return chat_history


def estimate_tokens(text):
    """简单估算文本的令牌数"""
    if not text:
        return 0
    return len(text.split())


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理类似OpenAI的聊天完成请求"""
    global global_pipe

    # 确保模型已加载
    if global_pipe is None:
        return jsonify({"error": "Model not initialized"}), 500

    try:
        # 获取请求数据
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.json
        logger.info(f"收到请求: {json.dumps(data, ensure_ascii=False)[:200]}...")

        # 提取关键参数
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        # 记录消息概况
        logger.info(f"收到 {len(messages)} 条消息")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:30]
            logger.info(f"  消息 {i + 1}: {role} - {content}...")

        # 提取生成参数
        generation_config = {
            "max_new_tokens": min(data.get('max_tokens', CONFIG["max_tokens"]), 1024),
            "temperature": data.get('temperature', CONFIG["temperature"]),
            "top_p": data.get('top_p', 0.8),
            "repetition_penalty": data.get('repetition_penalty', 1.1)
        }

        # 准备聊天历史
        try:
            chat_history = prepare_chat_history(messages)
            logger.info(f"处理后的聊天历史长度: {len(chat_history)}")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # 生成唯一请求ID
        response_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

        # 使用队列进行请求处理
        result_queue = queue.Queue()

        # 将请求放入请求队列
        try:
            request_queue.put(
                (response_id, chat_history, generation_config, result_queue),
                block=True,
                timeout=5  # 最多等待5秒
            )
        except queue.Full:
            logger.error("请求队列已满，无法处理更多请求")
            return jsonify({"error": "Server is too busy, please try again later"}), 503

        # 等待结果，设置超时
        try:
            result = result_queue.get(block=True, timeout=CONFIG["request_timeout"])
        except queue.Empty:
            logger.error(f"请求 {response_id} 处理超时")
            return jsonify({"error": "Request timed out"}), 504

        if not result["success"]:
            logger.error(f"请求 {response_id} 处理失败: {result['error']}")
            return jsonify({"error": result["error"]}), 500

        # 获取生成的文本
        generated_text = result["text"]
        logger.info(f"生成的文本 (前5000个字符): {generated_text[:5000]}...")

        # 构建OpenAI兼容的响应格式
        response_data = {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(CONFIG["model_path"]),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(estimate_tokens(m["content"]) for m in messages),
                "completion_tokens": estimate_tokens(generated_text),
                "total_tokens": sum(estimate_tokens(m["content"]) for m in messages) + estimate_tokens(generated_text)
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    global global_pipe
    if global_pipe is None:
        return jsonify({"status": "error", "message": "Model not initialized"}), 500

    # 添加队列状态信息
    queue_status = {
        "current_size": request_queue.qsize(),
        "max_size": CONFIG["max_queue_size"],
        "usage_percent": (request_queue.qsize() / CONFIG["max_queue_size"]) * 100
    }

    return jsonify({
        "status": "ok",
        "model": os.path.basename(CONFIG["model_path"]),
        "config": {k: v for k, v in CONFIG.items() if k != "model_path"},
        "queue": queue_status,
        "workers": len(worker_threads)
    })


# 添加模型信息端点
@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型，兼容OpenAI API"""
    model_name = os.path.basename(CONFIG["model_path"])
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ]
    })


def start_worker_threads():
    """启动工作线程"""
    global worker_threads

    for i in range(CONFIG["max_workers"]):
        thread = threading.Thread(target=worker_function, daemon=True)
        thread.start()
        worker_threads.append(thread)
        logger.info(f"启动工作线程 {i + 1}/{CONFIG['max_workers']}")


def stop_worker_threads():
    """停止工作线程"""
    global worker_threads

    # 发送结束信号给所有线程
    for _ in range(len(worker_threads)):
        request_queue.put(None)

    # 等待所有线程结束
    for i, thread in enumerate(worker_threads):
        thread.join(timeout=1)
        logger.info(f"工作线程 {i + 1} 已停止")

    worker_threads = []


if __name__ == "__main__":
    try:
        # 在启动Flask前初始化模型
        initialize_model()

        # 启动工作线程
        start_worker_threads()

        # 启动Flask应用
        logger.info(f"启动Flask服务器在端口 {CONFIG['port']}...")

        # 使用waitress作为生产服务器以处理高并发
        try:
            from waitress import serve

            serve(app, host='0.0.0.0', port=CONFIG["port"], threads=CONFIG["max_workers"] + 2)
        except ImportError:
            logger.warning("找不到waitress，使用Flask开发服务器。建议在生产环境中安装waitress: pip install waitress")
            app.run(host='0.0.0.0', port=CONFIG["port"], threaded=True)

    except Exception as e:
        logger.critical(f"服务器启动失败: {str(e)}")
        logger.critical(traceback.format_exc())

    finally:
        # 确保在退出时停止所有工作线程
        stop_worker_threads()
