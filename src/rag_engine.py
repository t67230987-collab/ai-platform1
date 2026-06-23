"""
RAG 引擎 - 文档检索增强生成
支持 PDF / TXT 上传, 向量化存储, 语义检索

向量化方案 (按优先级):
  1. Sentence-Transformers (HF 镜像)
  2. TF-IDF (sklearn, 无需网络)
  3. 关键词匹配 (纯本地回退)
"""
import os
import hashlib
import numpy as np
import chromadb
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, EMBEDDING_MODEL


def _get_embedder():
    """获取最优可用的 Embedder"""
    # 方案1: Sentence-Transformers (尝试国内镜像)
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        # 快速测试
        model.encode(["test"])
        return ("semantic", model)
    except Exception:
        pass

    # 方案2: TF-IDF
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        class TFIDFEmbedder:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(max_features=256)
                self.fitted = False
                self.doc_vectors = None

            def fit(self, texts):
                self.doc_vectors = self.vectorizer.fit_transform(texts)
                self.fitted = True

            def encode(self, texts):
                if not self.fitted:
                    self.fit(texts)
                    return self.doc_vectors.toarray()
                return self.vectorizer.transform(texts).toarray()

            def encode_query(self, query):
                return self.vectorizer.transform([query]).toarray()[0]

        return ("tfidf", TFIDFEmbedder())
    except ImportError:
        pass

    # 方案3: 简单关键词匹配
    class KeywordMatcher:
        def encode(self, texts):
            return np.array([[len(set(t) & set(texts[0].split()))]
                             for t in texts])

    return ("keyword", KeywordMatcher())


class RAGEngine:
    """文档检索增强生成引擎"""

    def __init__(self, persist_dir="./data/chroma_db"):
        os.makedirs(persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.embed_type, self.embedder = _get_embedder()
        print(f"[RAG] 使用向量化方案: {self.embed_type}")
        self.current_collection = None
        self.chunks = []
        self.doc_names = []

    def load_document(self, file_path, file_type="txt"):
        """加载并解析文档"""
        if file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_type == "pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                return "\n\n".join([p.page_content for p in pages])
            except ImportError:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                return "\n\n".join(
                    [page.extract_text() or "" for page in reader.pages]
                )
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")

    def chunk_text(self, text):
        """将文本分割为重叠的块"""
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _encode(self, texts, is_query=False):
        """统一编码接口"""
        if self.embed_type == "semantic":
            return self.embedder.encode(texts, show_progress_bar=False)
        elif self.embed_type == "tfidf":
            if is_query:
                return self.embedder.encode_query(texts[0])
            else:
                self.embedder.fit(texts)
                return self.embedder.encode(texts)
        else:
            return self.embedder.encode(texts)

    def process_document(self, file_path, file_name=None):
        """完整流程: 加载 → 分块 → 向量化 → 存入 ChromaDB"""
        ext = os.path.splitext(file_path)[1].lower()
        file_type = "pdf" if ext == ".pdf" else "txt"
        doc_name = file_name or os.path.basename(file_path)

        text = self.load_document(file_path, file_type)
        chunks = self.chunk_text(text)

        if not chunks:
            return 0

        doc_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        collection_name = f"rag_{doc_hash}"

        try:
            self.current_collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            self.current_collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"doc_name": doc_name, "chunk_count": len(chunks)},
            )

        # 向量化
        if self.embed_type == "tfidf":
            # TF-IDF: fit on all chunks
            self.embedder.fit(chunks)
            embeddings = self.embedder.encode(chunks)
        else:
            embeddings = self._encode(chunks)

        # 清空并写入
        existing = self.current_collection.get()
        if existing["ids"]:
            self.current_collection.delete(ids=existing["ids"])

        self.current_collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )

        self.chunks = chunks
        self.doc_names = [doc_name]
        return len(chunks)

    def retrieve(self, query, top_k=None):
        """检索最相关的文档块"""
        if top_k is None:
            top_k = TOP_K_RETRIEVAL

        if self.current_collection is None:
            return []

        query_vec = self._encode([query], is_query=True)
        # 确保形状为 (1, dim) → [[...values...]]
        if isinstance(query_vec, np.ndarray) and query_vec.ndim == 2:
            query_embeddings = query_vec.tolist()
        else:
            query_embeddings = [query_vec.tolist()] if hasattr(query_vec, 'tolist') else [[float(v) for v in query_vec]]

        results = self.current_collection.query(
            query_embeddings=query_embeddings,
            n_results=min(top_k, len(self.chunks)),
        )

        if results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []

    def get_stats(self):
        """获取当前文档统计信息"""
        if self.current_collection is None:
            return {"文档数": 0, "分块数": 0}
        return {
            "文档名": self.doc_names[0] if self.doc_names else "无",
            "分块数": len(self.chunks),
            "分块大小": CHUNK_SIZE,
            "重叠": CHUNK_OVERLAP,
            "向量化方案": self.embed_type,
        }
