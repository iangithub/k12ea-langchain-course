"""Day 2 共用知識庫工具：讀文件、切 chunk、建本機 Qdrant、開啟既有知識庫。"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
QDRANT_PATH = PROJECT_ROOT / ".qdrant"
COLLECTION_NAME = "school_documents"


class KnowledgeBaseError(RuntimeError):
    """Raised when the local knowledge base is missing or inconsistent."""


def load_school_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    # 把每一份 Markdown 讀成 LangChain Document，並先補好後面會用到的 metadata。
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        # 讀取 Markdown 內容，去掉前後空白；如果是空文件就跳過。
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        # 取 Markdown 第一行當作 title，如果沒有標題就用檔名。
        first_line = content.splitlines()[0]
        # 取第一行的標題，去掉開頭的 #，再去掉前後空白；如果沒有標題就用檔名。
        # path.stem 會去掉副檔名，保留檔名本身。並用檔名當作 category，方便後續篩選。
        title = first_line.removeprefix("# ").strip() or path.stem
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": f"data/{path.name}",
                    "title": title,
                    "category": path.stem,
                },
            )
        )
    if not documents:
        raise KnowledgeBaseError(f"在 {data_dir} 找不到 Markdown 文件")
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 280,
    chunk_overlap: int = 50,
) -> list[Document]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size 必須大於 0，chunk_overlap 必須介於 0 與 chunk_size 之間")

    # 優先沿著 Markdown 標題、段落與句子邊界切分；內容仍太長時，
    # 才依序改用較細的分隔符號，最後的空字串表示必要時可按字元切開。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 每個 chunk 的目標最大字元數，不是 Token 數。
        chunk_overlap=chunk_overlap,  # 相鄰 chunk 保留部分重複內容，避免語意被切斷。
        add_start_index=True,  # 在 metadata 記錄 chunk 位於原文件的起始位置。
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", ""],
    )
    chunks = splitter.split_documents(documents) # 把每份文件切成多個 chunk，並保留原始文件的 metadata。
    source_counts: defaultdict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = str(chunk.metadata["source"])
        source_counts[source] += 1
        chunk.metadata["chunk_id"] = f"{Path(source).stem}-{source_counts[source]:02d}"
    return chunks


def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "（沒有檢索到相關資料）"

    # 把檢索結果包成模型容易閱讀的文字區塊，同時保留 chunk_id 與 source 方便引用。
    blocks = []
    for document in documents:
        metadata = document.metadata
        blocks.append(
            f"[{metadata.get('chunk_id', 'unknown')}] "
            f"{metadata.get('title', '未命名')}（{metadata.get('source', 'unknown')}）\n"
            f"{document.page_content}"
        )
    return "\n\n".join(blocks)


def source_summaries(documents: list[Document]) -> list[str]:
    # 終端機輸出只需要精簡版來源清單，不需要把整段內容重印一次。
    seen: set[tuple[str, str]] = set()
    summaries: list[str] = []
    for document in documents:
        key = (
            str(document.metadata.get("source", "unknown")),
            str(document.metadata.get("chunk_id", "unknown")),
        )
        if key not in seen:
            seen.add(key)
            summaries.append(f"{key[1]} | {key[0]}")
    return summaries


def _point_ids(chunks: list[Document]) -> list[str]:
    # Qdrant 每個點都需要穩定 id；這裡用 source + chunk_id 產生可重建的 UUID。
    return [
        str(uuid5(NAMESPACE_URL, f"{chunk.metadata['source']}:{chunk.metadata['chunk_id']}"))
        for chunk in chunks
    ]


def build_local_vector_store(
    embeddings: Embeddings,
    db_path: Path = QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 280,
    chunk_overlap: int = 50,
) -> tuple[QdrantVectorStore, QdrantClient, list[Document]]:
    # 每次重建都先清空舊資料，確保 chunk 內容與 embedding model 完全一致。
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = split_documents(
        load_school_documents(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # 先用一筆測試文字探出 embedding 維度，再建立對應大小的 collection。
    vector_size = len(embeddings.embed_query("校務文件向量維度測試"))
    client = QdrantClient(path=str(db_path))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    # 真正寫入 Qdrant 的就是這一步：chunk 文字 + 對應 id。
    vector_store.add_documents(chunks, ids=_point_ids(chunks))
    return vector_store, client, chunks


def open_local_vector_store(
    embeddings: Embeddings,
    db_path: Path = QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
) -> tuple[QdrantVectorStore, QdrantClient]:
    # 這支函式只負責「開啟既有知識庫」，不自動重建，讓 Day 2 的建庫步驟保持清楚。
    if not db_path.exists():
        raise KnowledgeBaseError("尚未建立 Qdrant 知識庫，請先執行 02_build_qdrant.py")
    client = QdrantClient(path=str(db_path))
    if not client.collection_exists(collection_name):
        client.close()
        raise KnowledgeBaseError("Qdrant collection 不存在，請重新執行 02_build_qdrant.py")
    return (
        QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        ),
        client,
    )