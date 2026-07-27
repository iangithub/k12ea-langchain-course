"""最小 RAG 範例：讀取整份短校規，直接放進 Prompt 再回答。"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from examples.model_factory import create_chat_model

# 讀取整份短校規，直接放進 Prompt 再回答
RULES_PATH = Path(__file__).resolve().parents[1] / "data" / "校規.md"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是校規問答助理。只能根據下方校規回答，不要使用自己的知識補充規定。"
            "如果校規沒有答案，請明確回答「校規中查無相關資訊」。\n\n"
            "校規：\n{rules}",
        ),
        ("human", "{question}"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把整份短校規放進 Prompt 後回答問題")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--question", default="學生證遺失後要先到哪裡登記？")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 這裡的「取得」很單純：不做向量搜尋，直接讀取整份短文件。
    rules = RULES_PATH.read_text(encoding="utf-8")
    prompt_value = PROMPT.invoke({"rules": rules, "question": args.question})

    # 建立模型client
    model = create_chat_model(args.provider)
    # 呼叫模型並取得回答
    response = model.invoke(prompt_value)

    print(f"參考資料：{RULES_PATH.name}（整份讀取）")
    print(f"問題：{args.question}")
    print(f"回答：{response.text}")


if __name__ == "__main__":
    main()