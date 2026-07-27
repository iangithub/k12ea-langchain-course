"""完整 PDF RAG：讀取法規、切分、寫入 Qdrant，再依檢索內容回答。"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from examples.model_factory import create_chat_model, create_embeddings
from examples.rag_pipeline import validate_citations
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
            "資料不足時，明確說明無法從本次檢索內容回答，並將 insufficient_evidence 設為 true。",
        ),
        (
            "human",
            "使用者問題：{question}\n\n"
            "檢索內容：\n{context}\n\n"
            "請依指定 Schema 回答。這是課程示範，不提供法律或人事決策建議。",
        ),
    ]
)


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


def format_pdf_chunks(documents: list[Document]) -> str:
    blocks = []
    for document in documents:
        metadata = document.metadata
        blocks.append(
            f"[{metadata['chunk_id']}] source={metadata['source']} page={metadata['page']}\n"
            f"{document.page_content}"
        )
    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="從教育人員任用條例 PDF 建庫並執行 RAG 問答")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--question", default="教育人員的任用應注意哪些條件？")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = create_embeddings(args.provider)
    vector_store, client, chunks, page_count = build_pdf_vector_store(
        embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    try:
        documents = vector_store.similarity_search(args.question, k=args.k)
        model = create_chat_model(args.provider)
        structured_model = model.with_structured_output(RagAnswer)
        prompt_value = ANSWER_PROMPT.invoke(
            {"question": args.question, "context": format_pdf_chunks(documents)}
        )
        answer = structured_model.invoke(prompt_value)
        if not isinstance(answer, RagAnswer):
            raise TypeError("模型未回傳 RagAnswer")
        citation_issues = validate_citations(answer, documents)

        print(f"PDF：{PDF_PATH.name}")
        print(f"載入結果：{page_count} 頁，切成 {len(chunks)} 個 chunk")
        print(f"向量資料庫：{PDF_QDRANT_PATH}")
        print(f"問題：{args.question}")
        print("檢索來源：")
        for document in documents:
            print(
                f"- {document.metadata['chunk_id']} | "
                f"第 {document.metadata['page']} 頁"
            )
        print("\n結構化回答：")
        print(answer.model_dump_json(indent=2))
        if citation_issues:
            print("\n引用驗證未通過：")
            for issue in citation_issues:
                print(f"- {issue}")
        else:
            print("\n引用驗證：通過")
        print("\n提醒：本結果僅供 RAG 技術課程示範，不作為法律或人事決策依據。")
    finally:
        client.close()


if __name__ == "__main__":
    main()