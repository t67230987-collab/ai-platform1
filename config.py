"""
AI 智能平台 - 全局配置
学生: 宋星轮 | 学号: 202335020633
"""

# ---- 学生信息 ----
STUDENT_NAME = "宋星轮"
STUDENT_ID = "202335020633"

# ---- LLM 配置 ----
# 默认使用 Ollama 本地模型 (免费, 需先安装 Ollama)
# 也支持 OpenAI 兼容 API
DEFAULT_LLM_BACKEND = "ollama"  # "ollama" | "openai" | "demo"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"  # 推荐阿里通义千问 7B

# OpenAI 兼容 API (可选)
OPENAI_API_KEY = ""  # 填入你的 API Key
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-3.5-turbo"

# ---- RAG 配置 ----
CHUNK_SIZE = 512       # 文档分块大小
CHUNK_OVERLAP = 50     # 分块重叠
TOP_K_RETRIEVAL = 4    # 检索最相关文档数
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers 模型

# ---- 参数默认值 ----
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1024
