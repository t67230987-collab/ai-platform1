# 🤖 AI 智能问答平台

> **学生:** 宋星轮 | **学号:** 202335020633  
> Multi-LLM · RAG 检索增强生成 · Multi-Agent 协作

---

## 🎯 项目简介

本项目是一个集 **AI智能对话**、**RAG文档检索增强生成**、**多Agent协作分析** 于一体的综合性AI平台。采用前沿的AI技术栈，支持多种LLM后端切换，提供实时参数调节，适用于学习、研究及实际应用场景。

### ✨ 核心功能

| 功能 | 说明 | 技术亮点 |
|------|------|----------|
| 💬 **AI 智能对话** | 多轮对话，支持 Ollama 本地模型 / OpenAI API | 前后端分离架构，灵活切换 |
| 📚 **RAG 文档问答** | 上传PDF/TXT，自动分块、向量化、语义检索 | ChromaDB向量库 + Sentence-Transformers |
| 🤝 **多Agent协作** | 4个AI角色同时分析，综合决策 | Multi-Agent + 角色分工 |

### 🎛️ 可调参数

- **Temperature** (0.0-2.0): 控制回复创造性
- **Top-P** (0.0-1.0): 核采样多样性
- **Max Tokens** (64-4096): 最大回复长度
- **Top-K Retrieval** (1-10): RAG 检索数量
- **Chunk Size** (可配置): 文档分块大小

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│    (学生信息卡片 | 参数面板 | 三Tab切换)           │
├─────────────────────────────────────────────────┤
│                    后端层                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │LLM Client│  │RAG Engine│  │Multi-Agent   │   │
│  │Ollama    │  │ChromaDB  │  │4角色并行分析 │   │
│  │OpenAI    │  │Embedding │  │综合决策      │   │
│  │Demo      │  │Retrieval │  │              │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
├─────────────────────────────────────────────────┤
│                 数据层                            │
│  ChromaDB (向量存储) | 文件系统 (文档)            │
└─────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd ai-platform

# 安装依赖
pip install -r requirements.txt
```

### 2. 选择 LLM 后端

**方式A: Ollama (推荐，免费本地)**
```bash
# 安装 Ollama: https://ollama.com
ollama pull qwen2.5:7b     # 阿里通义千问 7B
# 或
ollama pull llama3:8b       # Meta Llama3 8B
```

**方式B: OpenAI API**
```python
# 在 config.py 中设置
OPENAI_API_KEY = "sk-xxxxx"
```

**方式C: Demo 模式**
无需任何配置，使用预设回复演示

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`

---

## 📂 项目结构

```
ai-platform/
├── app.py                 # Streamlit 主程序
├── config.py              # 全局配置 & 学生信息
├── requirements.txt       # 依赖列表
├── README.md              # 项目文档
├── src/
│   ├── __init__.py
│   ├── llm_client.py      # LLM 客户端 (Ollama/OpenAI/Demo)
│   ├── rag_engine.py      # RAG 引擎 (加载/分块/向量化/检索)
│   ├── agents.py          # 多 Agent 协作系统
│   └── utils.py           # 工具函数
├── data/                  # 文档存储 & 向量数据库
│   └── chroma_db/         # ChromaDB 持久化
└── assets/                # 静态资源
```

---

## 🎥 演示视频

5分钟录屏解说视频请见: [B站链接](https://www.bilibili.com/video/BV1Q2iXBtEme?vd_source=8ae9d521b063cacc89e59cbdfe8f518e)

### 视频解说要点

1. **0:00-1:00** — 项目背景介绍、学生信息、技术栈概览
2. **1:00-2:00** — 演示 AI 智能对话功能，展示参数调节效果
3. **2:00-3:30** — 演示 RAG 文档问答，上传PDF并提问
4. **3:30-4:30** — 演示多 Agent 协作，展示前沿功能
5. **4:30-5:00** — 总结、项目优化方向、GitHub 仓库信息

---

## 🔧 可优化方向

- [ ] 支持更多文档格式 (Word, Markdown, HTML)
- [ ] 添加对话记忆 (Memory) 机制
- [ ] Agent 并行调用 (异步)
- [ ] 接入更多模型 (Claude, Gemini)
- [ ] Docker 容器化部署
- [ ] 添加用户认证系统
- [ ] API 接口化 (FastAPI)

---

## 📄 License

MIT License

---

**© 2025 宋星轮 (202335020633)** — AI 智能平台
