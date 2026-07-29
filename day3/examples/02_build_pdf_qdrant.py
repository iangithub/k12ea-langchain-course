"""將 PDF 切分並寫入本機 Qdrant，供後續 RAG 問答使用。

執行：uv run examples/02_build_pdf_qdrant.py
"""

from __future__ import annotations

import argparse

from examples.model_factory import create_embeddings
from examples.pdf_rag import PDF_PATH, PDF_QDRANT_PATH, build_pdf_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立教育人員任用條例 PDF 向量資料庫")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = create_embeddings(args.provider)
    vector_store, client, chunks, page_count = build_pdf_vector_store(
        embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    del vector_store
    client.close()

    print(f"PDF：{PDF_PATH.name}")
    print(f"載入結果：{page_count} 頁，切成 {len(chunks)} 個 chunk")
    print(f"已建立向量資料庫：{PDF_QDRANT_PATH}")
    print("後續只有 PDF、Chunk 設定或 Embedding Model 改變時才需要重新建庫。")


if __name__ == "__main__":
    main()