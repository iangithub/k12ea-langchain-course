import argparse

"""直接觀察向量檢索回來的 chunk，先不讓模型整理答案。"""

from qdrant_client import models

from examples.knowledge_base import open_local_vector_store
from examples.model_factory import create_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="搜尋本機 Qdrant 校務知識庫")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--query", default="借書證不見了要如何處理？")
    parser.add_argument("--category", choices=["校規", "圖書館規則", "活動辦法"])
    parser.add_argument("-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = create_embeddings(args.provider)
    vector_store, client = open_local_vector_store(embeddings)

    # category filter 是選做條件：讓學員觀察 metadata 也能影響檢索範圍。
    query_filter = None
    if args.category:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.category",
                    match=models.MatchValue(value=args.category),
                )
            ]
        )

    try:
        # similarity_search_with_score 會同時回傳文件與分數，方便比較排序結果。
        results = vector_store.similarity_search_with_score(
            args.query,
            k=args.k,
            filter=query_filter,
        )
        print(f"查詢：{args.query}")
        for rank, (document, score) in enumerate(results, start=1):
            # 分數越接近前面，不代表一定最好，但通常表示和問題更相似。
            print("\n" + "=" * 72)
            print(
                f"#{rank} score={score:.4f} | {document.metadata['chunk_id']} | "
                f"{document.metadata['source']}"
            )
            print(document.page_content)
    finally:
        client.close()


if __name__ == "__main__":
    main()