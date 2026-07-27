from __future__ import annotations

"""Tool Calling 暖身範例：先看模型提出工具需求，再看程式真正執行工具。"""

import argparse

from langchain.tools import tool

from model_factory import create_chat_model, get_provider

SCHOOL_DATES = {
    "親師座談": "2026-09-12 14:00",
    "校慶": "2026-11-21 08:30",
    "寒假開始日": "2027-01-21",
}


@tool
def get_school_date(event: str) -> str:
    """查詢虛構校園活動的日期與時間。當使用者詢問活動日期時使用。"""
    return SCHOOL_DATES.get(event, f"查無「{event}」的日期資料")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="觀察模型提出工具呼叫並由程式執行")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--event", default="親師座談")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    model = create_chat_model(provider)

    # tool_choice="any" 是教學上的刻意安排，保證第一輪一定看得到 tool call。
    forced_tool_model = model.bind_tools([get_school_date], tool_choice="any")

    # 先準備對話訊息，讓模型知道：回答日期前，必須先查工具。
    messages: list[object] = [
        (
            "system",
            "回答校園活動日期前，必須先使用 get_school_date 查詢，不可根據記憶猜測。",
        ),
        ("human", f"{args.event}是什麼時候？"),
    ]

    ai_message = forced_tool_model.invoke(messages)
    messages.append(ai_message)
    print(f"模型提出的工具呼叫：{ai_message.tool_calls}")

    # 這個 for 迴圈就是 Tool Calling 的重點：模型只提出要求，真正執行的是 Python。
    for tool_call in ai_message.tool_calls:
        tool_result = get_school_date.invoke(tool_call)
        messages.append(tool_result)
        print(f"工具執行結果：{tool_result.content}")

    # 把工具結果放回對話後，再呼叫一次原始模型整理給使用者看的最終答案。
    final_response = model.invoke(messages)
    print(f"最終回答：{final_response.text}")


if __name__ == "__main__":
    main()
