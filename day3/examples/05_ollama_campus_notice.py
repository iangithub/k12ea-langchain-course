"""使用 Ollama 本地模型把校務草稿整理成通知。

準備模型：ollama pull gpt-oss:20b
執行：uv run examples/05_ollama_campus_notice.py
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

DEFAULT_DRAFT = (
    "五年級閱讀活動，星期五下午兩點在視聽教室，參加學生請帶一本喜歡的書，"
    "家長回條星期三前交給導師，有問題請洽教務處閱讀推動教師。"
)

NOTICE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是校務通知整理助手。只能使用草稿中的資訊，不得自行增加日期、"
            "地點、費用、聯絡人或規定。缺少的重要欄位請標示「草稿未提供」。"
            "使用繁體中文，依序輸出標題、活動資訊、應辦事項、聯絡方式四段。",
        ),
        ("human", "請把以下草稿整理成給家長閱讀的校務通知：\n\n{draft}"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama 本地校務通知助手")
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    model = ChatOllama(
        model=args.model,
        base_url=args.base_url,
        temperature=0,
        validate_model_on_init=True,
    )
    chain = NOTICE_PROMPT | model | StrOutputParser()

    print(f"本地模型：{args.model}")
    print(f"Ollama：{args.base_url}")
    print(f"\n原始草稿：\n{args.draft}\n")
    print("=== 整理後通知 ===")
    print(chain.invoke({"draft": args.draft}))


if __name__ == "__main__":
    main()