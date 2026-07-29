"""使用一個 Agent 與多個工具處理校園行政問題。

執行：uv run examples/03_single_agent.py
指定問題：uv run examples/03_single_agent.py --question "請查王老師的課表"
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from examples.campus_tools import CAMPUS_TOOLS
from examples.message_utils import content_to_text
from examples.model_factory import create_chat_model

SYSTEM_PROMPT = """
你是校園行政助理，只能根據工具回傳的虛構課程資料回答。

工作原則：
- 先判斷問題需要哪些工具；複合問題可以依序呼叫多個工具。
- 涉及教師課表、活動流程、場地或費用時，必須先呼叫對應工具。
- 不得虛構工具沒有提供的師生資料、時段、規定或金額。
- 資料不足時，說明缺少哪些資訊，不要自行補值。
- 回答以繁體中文呈現，清楚區分查詢結果與提醒。

所有姓名、課表、場地與流程均為課程虛構資料，不可作為真實校務決策依據。
""".strip()

DEFAULT_QUESTION = (
    "王怡婷老師星期五下午想帶 60 人辦閱讀活動，"
    "請確認她的課表、建議可用場地，並列出申請流程。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-Agent 校園行政助理")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    return parser.parse_args()


def print_tool_trace(messages: list[Any]) -> None:
    """顯示可觀測的工具呼叫，不輸出模型內部推理。"""
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_names = "、".join(call["name"] for call in message.tool_calls)
            print(f"[Agent 選擇工具] {tool_names}")
        elif isinstance(message, ToolMessage):
            print(f"[工具結果：{message.name}]\n{content_to_text(message.content)}\n")


def main() -> None:
    args = parse_args()
    model = create_chat_model(args.provider)
    agent = create_agent(
        model=model,
        tools=CAMPUS_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        name="campus_single_agent",
    )

    print(f"問題：{args.question}\n")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.question}]},
        config={"recursion_limit": 12},
    )
    messages = result["messages"]
    print_tool_trace(messages)
    print("=== Agent 回答 ===")
    print(content_to_text(messages[-1].content))


if __name__ == "__main__":
    main()