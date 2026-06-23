"""工具函数"""


def format_response(text, max_lines=30):
    """格式化输出文本"""
    lines = text.strip().split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("... (内容过长，已截断)")
    return "\n".join(lines)


def check_ollama_status():
    """检测 Ollama 服务是否可用"""
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return True, [m["name"] for m in models]
        return False, []
    except Exception:
        return False, []


def get_demo_upload_path():
    """返回示例文档路径"""
    import os
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
