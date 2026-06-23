"""
AI 智能问答平台 - 主程序
====================================================
学生: 宋星轮 | 学号: 202335020633
====================================================

功能:
  1. AI 智能对话 (Ollama / OpenAI 多后端)
  2. RAG 文档检索增强生成
  3. 多 Agent 协作分析 (前沿功能)
  4. 实时参数调节

技术栈:
  - 前端: Streamlit
  - LLM: Ollama (本地) / OpenAI 兼容 API
  - RAG: ChromaDB + Sentence-Transformers
  - 多智能体: Multi-Agent Collaboration
"""

import streamlit as st
import os
import sys

# 确保 src 可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    STUDENT_NAME, STUDENT_ID,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
    OLLAMA_MODEL, OPENAI_MODEL,
)
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.agents import MultiAgentSystem, PRESET_AGENTS
from src.utils import check_ollama_status

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title=f"AI 智能平台 - {STUDENT_NAME}",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 侧边栏 - 学生信息 + 参数调节
# ============================================================
with st.sidebar:
    # ---- 学生信息卡片 ----
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin-bottom: 15px;
    ">
        <h3 style="margin: 0; color: white;">🎓 学生信息</h3>
        <p style="margin: 8px 0; font-size: 18px;"><b>姓名:</b> {name}</p>
        <p style="margin: 0; font-size: 18px;"><b>学号:</b> {id}</p>
    </div>
    """.format(name=STUDENT_NAME, id=STUDENT_ID), unsafe_allow_html=True)

    # ---- 后端选择 ----
    st.subheader("⚙️ 模型配置")

    backend = st.selectbox(
        "LLM 后端",
        options=["ollama", "openai", "demo"],
        format_func=lambda x: {"ollama": "🦙 Ollama (本地)", "openai": "🌐 OpenAI API", "demo": "📦 Demo 模式"}[x],
        help="Ollama=免费本地模型 | OpenAI=云端API | Demo=无需联网"
    )

    if backend == "ollama":
        model_name = st.text_input("模型名称", value=OLLAMA_MODEL)
        ok, models = check_ollama_status()
        if ok:
            st.success(f"✅ Ollama 已连接 ({len(models)} 个模型)")
        else:
            st.warning("⚠️ Ollama 未运行，请启动服务")
            st.caption("安装: https://ollama.com")
            st.caption(f"拉取模型: `ollama pull {OLLAMA_MODEL}`")
    elif backend == "openai":
        model_name = st.text_input("模型名称", value=OPENAI_MODEL)
        st.caption("请确保 config.py 中已配置 API Key")
    else:
        model_name = "demo"
        st.info("📦 Demo 模式 - 无需联网，使用预设回复演示")

    # ---- 参数调节 ----
    st.subheader("🎛️ 参数调节")

    temperature = st.slider(
        "Temperature (创造性)", 0.0, 2.0, DEFAULT_TEMPERATURE, 0.05,
        help="越高=回复越有创造性；越低=回复越保守"
    )
    top_p = st.slider(
        "Top-P (核采样)", 0.0, 1.0, DEFAULT_TOP_P, 0.05,
        help="控制词汇多样性，1.0=最大多样性"
    )
    max_tokens = st.slider(
        "Max Tokens (最大长度)", 64, 4096, DEFAULT_MAX_TOKENS, 64,
        help="回复的最大 token 数量"
    )

    st.divider()
    st.caption(f"© 2025 {STUDENT_NAME} | AI 智能平台 v1.0")


# ============================================================
# 主区域 - 标题
# ============================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🤖 AI 智能问答平台")
    st.caption("Multi-LLM · RAG 检索增强 · Multi-Agent 协作")
with col2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 10px;">
        <span style="background: #667eea; color: white; padding: 5px 12px; border-radius: 8px; font-size: 14px;">
            {STUDENT_NAME}
        </span>
        <br>
        <span style="font-size: 12px; color: gray;">{STUDENT_ID}</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================
# 初始化 Session State
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False

# ============================================================
# 三个 Tab
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "💬 AI 智能对话",
    "📚 RAG 文档问答",
    "🤝 多 Agent 协作",
])

# ============================================================
# Tab 1: AI 智能对话
# ============================================================
with tab1:
    st.subheader("💬 AI 智能对话")

    # 显示聊天历史
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 你好！我是 AI 助手，请在下方输入你的问题。")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 输入框
    if prompt := st.chat_input("输入你的问题..."):
        # 添加用户消息
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # 调用 LLM
        with st.spinner("🤔 AI 思考中..."):
            client = LLMClient(backend=backend, model=model_name)
            response = client.chat(
                messages=[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.chat_history],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # 清空按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

# ============================================================
# Tab 2: RAG 文档问答
# ============================================================
with tab2:
    st.subheader("📚 RAG 文档检索增强生成")

    col_upload, col_stats = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "上传文档 (PDF / TXT)",
            type=["pdf", "txt"],
            help="上传后自动分块、向量化，然后可以基于文档内容提问"
        )

        if uploaded_file:
            # 保存上传文件
            save_path = f"./data/{uploaded_file.name}"
            os.makedirs("./data", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 处理文档
            with st.spinner("📄 正在处理文档 (分块 → 向量化)..."):
                n_chunks = st.session_state.rag_engine.process_document(
                    save_path, uploaded_file.name
                )

            if n_chunks > 0:
                st.session_state.rag_loaded = True
                st.success(f"✅ 文档已加载: {uploaded_file.name} → {n_chunks} 个分块")

    with col_stats:
        st.markdown("**📊 文档状态**")
        if st.session_state.rag_loaded:
            stats = st.session_state.rag_engine.get_stats()
            for k, v in stats.items():
                st.metric(k, v)
        else:
            st.caption("尚未加载文档")

    # RAG 参数
    st.divider()
    st.caption("RAG 参数")
    rag_top_k = st.slider("检索数 (Top-K)", 1, 10, TOP_K_RETRIEVAL, 1, key="rag_k")

    # 问答
    if st.session_state.rag_loaded:
        rag_query = st.chat_input("基于文档提问...", key="rag_input")
        if rag_query:
            # 检索
            with st.spinner("🔍 检索相关文档块..."):
                contexts = st.session_state.rag_engine.retrieve(rag_query, top_k=rag_top_k)

            if contexts:
                # 构建 RAG prompt
                context_text = "\n\n---\n\n".join(contexts)
                rag_prompt = (
                    f"你是一个基于文档的问答助手。请根据以下文档内容回答问题。"
                    f"如果文档中没有相关信息，请如实告知。\n\n"
                    f"## 文档内容\n{context_text}\n\n"
                    f"## 用户问题\n{rag_query}\n\n"
                    f"## 回答"
                )

                # 显示检索结果
                with st.expander(f"📎 检索到 {len(contexts)} 个相关块", expanded=False):
                    for i, ctx in enumerate(contexts):
                        st.caption(f"块 #{i+1} (前200字):")
                        st.text(ctx[:200] + "...")

                # 调用 LLM 生成回答
                with st.spinner("🤔 生成回答..."):
                    client = LLMClient(backend=backend, model=model_name)
                    answer = client.chat(
                        messages=[{"role": "user", "content": rag_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                st.markdown("**🤖 AI 回答:**")
                st.markdown(answer)
            else:
                st.warning("未找到相关文档内容。")
    else:
        st.info("👆 请先上传一个文档 (PDF 或 TXT)")

# ============================================================
# Tab 3: 多 Agent 协作 (前沿功能)
# ============================================================
with tab3:
    st.subheader("🤝 多 Agent 协作分析")
    st.caption("多个 AI 角色从不同角度分析同一问题，最终综合得出全面答案")

    # Agent 选择
    st.markdown("**选择参与的 Agent:**")
    cols = st.columns(4)
    enabled = []
    for i, agent in enumerate(PRESET_AGENTS):
        with cols[i]:
            checked = st.checkbox(
                f"{agent['name']}",
                value=True,
                key=f"agent_{i}",
                help=agent['role'][:80]
            )
            if checked:
                enabled.append(i)

    if not enabled:
        st.warning("请至少选择一个 Agent")
    else:
        agent_question = st.text_area(
            "输入问题，多个 Agent 将同时分析:",
            placeholder="例如: '如何用AI技术优化校园选课系统?' 或 '前端框架 React vs Vue 选型分析'",
            height=80,
        )

        if st.button("🚀 启动多 Agent 分析", type="primary", disabled=not agent_question):
            # 进度容器
            progress_placeholder = st.empty()
            responses_container = st.container()

            agent_responses = []

            # 逐个调用 Agent (也可改为并行)
            for i in enabled:
                agent = PRESET_AGENTS[i]
                with st.status(f"{agent['name']} 分析中...", expanded=True) as status:
                    client = LLMClient(backend=backend, model=model_name)
                    response = client.chat(
                        messages=[{
                            "role": "user",
                            "content": f"{agent['role']}\n\n问题: {agent_question}\n\n请从你的专业角度分析，给出300字以内的核心观点。"
                        }],
                        temperature=temperature,
                        max_tokens=400,
                    )
                    agent_responses.append({"agent": agent["name"], "response": response})
                    st.markdown(f"**{agent['name']}:** {response[:200]}...")
                    status.update(label=f"✅ {agent['name']} 完成")

            # 综合
            if len(agent_responses) >= 2:
                st.divider()
                st.markdown("### 🧠 综合决策")
                with st.spinner("综合各方观点中..."):
                    mas = MultiAgentSystem(backend=backend, model=model_name)
                    final = mas.synthesize(
                        agent_question,
                        agent_responses,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                st.markdown(final)
            else:
                st.markdown("### 分析结果")
                st.markdown(agent_responses[0]["response"])

# ============================================================
# 底部
# ============================================================
st.divider()
st.caption(
    f"🎓 {STUDENT_NAME} | 学号: {STUDENT_ID} | "
    f"AI 智能平台 v1.0 | 后端: {backend} | "
    f"Temperature={temperature:.2f} Top-P={top_p:.2f}"
)
