"""可供命令列與 LINE Bot 共用的 PDF RAG 流程。"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from examples.schemas import RagAnswer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "教育人員任用條例.pdf"
PDF_QDRANT_PATH = PROJECT_ROOT / ".qdrant_pdf"
COLLECTION_NAME = "education_personnel_act"

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是教育法規問答助理。只能使用提供的檢索內容回答，不要自行補充規定。"
            "每個主張都要引用實際出現的 source、chunk_id 與原文短句。"
            "資料不足時，明確說明無法從本次檢索內容回答，"
            "並將 insufficient_evidence 設為 true。",
        ),
        (
            "human",
            "使用者問題：{question}\n\n"
            "檢索內容：\n{context}\n\n"
            "請依指定 Schema 回答。這是課程展示，不提供法律或人事決策建議。",
        ),
    ]
)


@dataclass(frozen=True)
class PdfRagResult:
    answer: RagAnswer
    documents: list[Document]
    citation_issues: list[str]


class PdfVectorStoreError(RuntimeError):
    """PDF 向量資料庫尚未建立或內容不完整。"""


def load_pdf_pages(pdf_path: Path = PDF_PATH) -> list[Document]:
    """把 PDF 每一頁的文字轉成一個 LangChain Document。"""
    reader = PdfReader(pdf_path)
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"data/{pdf_path.name}",
                    "title": "教育人員任用條例",
                    "page": page_number,
                },
            )
        )
    if not documents:
        raise ValueError("PDF 沒有可擷取的文字；掃描型 PDF 需要先進行 OCR")
    return documents


def split_pdf_pages(
    pages: list[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[Document]:
    """切分 PDF 文字，並加入包含頁碼的可讀 chunk_id。"""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size 必須大於 0，chunk_overlap 必須介於 0 與 chunk_size 之間")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n第 ", "\n", "。", "；", "，", ""],
    )
    chunks = splitter.split_documents(pages)
    page_counts: defaultdict[int, int] = defaultdict(int)
    for chunk in chunks:
        page_number = int(chunk.metadata["page"])
        page_counts[page_number] += 1
        chunk.metadata["chunk_id"] = (
            f"教育人員任用條例-p{page_number:03d}-c{page_counts[page_number]:02d}"
        )
    return chunks


def build_pdf_vector_store(
    embeddings: Embeddings,
    db_path: Path = PDF_QDRANT_PATH,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> tuple[QdrantVectorStore, QdrantClient, list[Document], int]:
    """重建 PDF 專用 Qdrant，回傳向量資料庫、client、chunks 與頁數。"""
    pages = load_pdf_pages()
    chunks = split_pdf_pages(pages, chunk_size, chunk_overlap)

    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    vector_size = len(embeddings.embed_query("教育人員任用條例向量維度測試"))
    client = QdrantClient(path=str(db_path))
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    return vector_store, client, chunks, len(pages)


def open_pdf_vector_store(
    embeddings: Embeddings,
    db_path: Path = PDF_QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
) -> tuple[QdrantVectorStore, QdrantClient]:
    """開啟既有 PDF 向量資料庫，不重新切分文件或寫入向量。"""
    if not db_path.exists():
        raise PdfVectorStoreError(
            "尚未建立 PDF 向量資料庫，請先執行 02_build_pdf_qdrant.py"
        )

    client = QdrantClient(path=str(db_path))
    if not client.collection_exists(collection_name):
        client.close()
        raise PdfVectorStoreError(
            "PDF Qdrant collection 不存在，請重新執行 02_build_pdf_qdrant.py"
        )

    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
    except Exception:
        client.close()
        raise
    return vector_store, client


def format_pdf_chunks(documents: list[Document]) -> str:
    blocks = []
    for document in documents:
        metadata = document.metadata
        blocks.append(
            f"[{metadata['chunk_id']}] source={metadata['source']} page={metadata['page']}\n"
            f"{document.page_content}"
        )
    return "\n\n".join(blocks)


def validate_citations(answer: RagAnswer, documents: list[Document]) -> list[str]:
    """確認模型引用的來源、chunk_id 與原文短句都在本次檢索結果中。"""
    by_chunk_id = {
        str(document.metadata.get("chunk_id")): document for document in documents
    }
    issues: list[str] = []
    for citation in answer.citations:
        document = by_chunk_id.get(citation.chunk_id)
        if document is None:
            issues.append(f"引用 {citation.chunk_id} 不在本次檢索結果中")
            continue
        if citation.source != document.metadata.get("source"):
            issues.append(f"引用 {citation.chunk_id} 的 source 與檢索 Metadata 不一致")
        normalized_quote = "".join(citation.quote.split())
        normalized_content = "".join(document.page_content.split())
        if normalized_quote not in normalized_content:
            issues.append(f"引用 {citation.chunk_id} 的 quote 不在原文中")
    if not answer.citations and not answer.insufficient_evidence:
        issues.append("回答宣稱證據足夠，但沒有提供引用")
    return issues


def answer_pdf_question(
    model: BaseChatModel,
    vector_store: Any,
    question: str,
    k: int = 4,
) -> PdfRagResult:
    """檢索 PDF、產生結構化回答，並驗證引用。"""
    documents = vector_store.similarity_search(question, k=k)
    structured_model = model.with_structured_output(RagAnswer)
    prompt_value = ANSWER_PROMPT.invoke(
        {"question": question, "context": format_pdf_chunks(documents)}
    )
    answer = structured_model.invoke(prompt_value)
    if not isinstance(answer, RagAnswer):
        raise TypeError("模型未回傳 RagAnswer")
    return PdfRagResult(
        answer=answer,
        documents=documents,
        citation_issues=validate_citations(answer, documents),
    )


def format_line_answer(result: PdfRagResult) -> str:
    """把結構化 RAG 結果整理成適合 LINE 顯示的文字。"""
    if result.citation_issues:
        return "引用驗證未通過，暫不提供答案。請查閱法規原文或洽詢承辦單位。"

    lines = [result.answer.answer.strip()]
    if result.answer.citations:
        lines.append("\n來源：")
        seen: set[tuple[str, str]] = set()
        documents = {
            str(document.metadata.get("chunk_id")): document
            for document in result.documents
        }
        for citation in result.answer.citations:
            key = (citation.source, citation.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            page = documents[citation.chunk_id].metadata.get("page")
            lines.append(f"- {citation.source}，第 {page} 頁（{citation.chunk_id}）")
    if result.answer.insufficient_evidence:
        lines.append("\n本次檢索證據不足，請查閱法規原文或洽詢承辦單位。")
    lines.append("\n本結果僅供 RAG 技術課程展示，不作為法律或人事決策依據。")
    return "\n".join(lines)
