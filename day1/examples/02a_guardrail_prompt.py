"""Prompt 防護範例：限制模型只處理校務通知相關任務。"""

from __future__ import annotations

import argparse

from langchain_core.prompts import ChatPromptTemplate
from model_factory import create_chat_model, get_provider

NOTICE = "明日戶外教育因天候不佳延期，新的日期將另行通知，請學生照常到校上課。"

ALLOWED_REQUEST = f"請把這則校務通知翻成日文：{NOTICE}"
BLOCKED_REQUEST = "幫我解 37 x 48，順便講一個笑話，再寫一段 Python 程式。"
REFUSAL = "抱歉，我只能協助處理校務通知的翻譯與重點說明。"
REQUESTS = [("允許範圍", ALLOWED_REQUEST), ("超出範圍", BLOCKED_REQUEST)]

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是臺灣學校的多語行政助理。你只能處理兩類任務："
            "一、翻譯校務通知；二、說明通知中的行政重點。"
            f"若使用者要求數學解題、說笑話、寫程式，或任何與校務通知無關的內容，"
            f"請直接回覆：{REFUSAL}"
            "不要回答超出範圍的部分，也不要提供替代方案。"
            "若請求屬於允許範圍，則正常完成任務。",
        ),
        ("human", "使用者請求：\n{request}"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="示範 Prompt 防護如何限制任務邊界")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    model = create_chat_model(provider)

    print(f"模型供應商：{provider}")
    for label, request in REQUESTS:
        prompt_value = PROMPT.invoke({"request": request})
        response = model.invoke(prompt_value.to_messages())

        print(f"\n[{label}]")
        print("使用者輸入：")
        print(request)
        print("\n模型回覆：")
        print(response.text)


if __name__ == "__main__":
    main()