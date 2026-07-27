import argparse

"""先看原始文件，再看它們如何被切成適合檢索的 chunk。"""

from examples.knowledge_base import load_school_documents, split_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="觀察校務 Document、Chunk 與 Metadata")
    parser.add_argument("--chunk-size", type=int, default=280)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # documents 是三份完整校務文件。
    documents = load_school_documents()
    # chunks 是切給向量檢索用的小段落。
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)

    print(f"原始文件：{len(documents)} 份")
    print(f"切分結果：{len(chunks)} 個 chunk")
    for chunk in chunks:
        # 這裡把每個 chunk 的來源與位置一起印出，方便觀察切分是否合理。
        print("\n" + "=" * 72)
        print(
            f"{chunk.metadata['chunk_id']} | {chunk.metadata['source']} | "
            f"start_index={chunk.metadata['start_index']}"
        )
        print(chunk.page_content)


if __name__ == "__main__":
    main()