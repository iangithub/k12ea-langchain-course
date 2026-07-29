"""開啟既有 PDF 向量資料庫，在命令列執行 RAG 問答。

執行：uv run examples/02a_pdf_rag.py
"""

from __future__ import annotations

import argparse

from examples.model_factory import create_chat_model, create_embeddings
from examples.pdf_rag import (
    PDF_QDRANT_PATH,
    answer_pdf_question,
    format_line_answer,
    open_pdf_vector_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用既有 PDF 向量資料庫執行 RAG 問答")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--question", default="教育人員的任用應注意哪些條件？")
    parser.add_argument("-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = create_embeddings(args.provider)
    vector_store, client = open_pdf_vector_store(embeddings)
    try:
        model = create_chat_model(args.provider)
        result = answer_pdf_question(model, vector_store, args.question, args.k)
        print(f"向量資料庫：{PDF_QDRANT_PATH}")
        print(f"問題：{args.question}\n")
        print(format_line_answer(result))
        if result.citation_issues:
            print("\n引用驗證問題：")
            for issue in result.citation_issues:
                print(f"- {issue}")
    finally:
        client.close()


if __name__ == "__main__":
    main()