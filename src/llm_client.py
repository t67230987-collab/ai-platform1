"""
LLM 客户端 - 支持 Ollama / OpenAI / Demo 三种后端
"""
import sys
import json
import requests
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
)


class LLMClient:
    """统一的 LLM 调用接口"""

    def __init__(self, backend="ollama", model=None):
        self.backend = backend
        self.model = model or (
            OLLAMA_MODEL if backend == "ollama" else OPENAI_MODEL
        )

    def chat(self, messages, temperature=0.7, top_p=0.9, max_tokens=1024):
        """发送对话请求"""
        if self.backend == "ollama":
            return self._chat_ollama(messages, temperature, top_p, max_tokens)
        elif self.backend == "openai":
            return self._chat_openai(messages, temperature, top_p, max_tokens)
        else:
            return self._chat_demo(messages)

    def _chat_ollama(self, messages, temperature, top_p, max_tokens):
        """Ollama API 调用"""
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["message"]["content"]
            return f"[Ollama 错误] HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.ConnectionError:
            return (
                "⚠️ 无法连接 Ollama。请确保:\n"
                "1. 已安装 Ollama (https://ollama.com)\n"
                "2. 已拉取模型: `ollama pull qwen2.5:7b`\n"
                "3. Ollama 服务正在运行"
            )
        except Exception as e:
            return f"[Ollama 异常] {str(e)}"

    def _chat_openai(self, messages, temperature, top_p, max_tokens):
        """OpenAI 兼容 API 调用"""
        if not OPENAI_API_KEY:
            return "⚠️ 请先在 config.py 中设置 OPENAI_API_KEY"
        try:
            resp = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return f"[API 错误] HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return f"[API 异常] {str(e)}"

    def _chat_demo(self, messages):
        """Demo 模式 - 返回预设回复 (无需联网)"""
        last_msg = messages[-1]["content"] if messages else ""
        return (
            f"🤖 [Demo 模式] 这是模拟回复。\n\n"
            f"你问的是: \"{last_msg[:100]}{'...' if len(last_msg) > 100 else ''}\"\n\n"
            f"✨ 本项目为「AI 智能问答平台」，支持:\n"
            f"  • Ollama 本地大模型 (推荐 qwen2.5:7b)\n"
            f"  • OpenAI 兼容 API\n"
            f"  • RAG 文档检索增强生成\n"
            f"  • 参数实时调节\n\n"
            f"📹 如需完整演示，请连接 Ollama 或配置 API Key。\n"
            f"🔗 Ollama 下载: https://ollama.com"
        )


# 全局单例
llm_client = None


def get_llm_client(backend="ollama", model=None):
    global llm_client
    if llm_client is None or llm_client.backend != backend:
        llm_client = LLMClient(backend, model)
    return llm_client
