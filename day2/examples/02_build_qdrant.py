import argparse

"""把切好的 chunk 寫進本機 Qdrant，建立 Day 2 之後要查詢的知識庫。"""

from examples.knowledge_base import QDRANT_PATH, build_local_vector_store
from examples.model_factory import create_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立本機 Qdrant 校務知識庫")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--chunk-size", type=int, default=280)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 建庫時用哪一個 embedding model，之後查詢時也要用同一個。
    embeddings = create_embeddings(args.provider)
    vector_store, client, chunks = build_local_vector_store(
        embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # 這支範例的目的只是建立資料庫，所以寫入完成後就把資源關掉。
    del vector_store
    client.close()
    print(f"已建立 {QDRANT_PATH}")
    print(f"已寫入 {len(chunks)} 個 chunk")


if __name__ == "__main__":
    main()