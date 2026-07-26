"""連續對話範例：比較有記憶與無記憶時的差異。"""

from __future__ import annotations

import argparse

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from model_factory import create_chat_model, get_provider

SETUP_REQUEST = (
    "接下來請都幫我把通知翻成日文，讀者是學生家長，"
    "語氣清楚、親切且正式。先記住，不用翻譯，回答『已記住』即可。"
)

FOLLOW_UP_REQUEST = (
    "那這則呢？\n"
    "通知：各位家長您好，明日戶外教育因天候不佳延期，新的日期將另行通知，"
    "請學生照常到校上課。"
)

RECALL_REQUEST = "我先前問過哪些問題？請只整理使用者問過的重點，用條列式回答。"

SYSTEM_MESSAGE = SystemMessage(
    content=(
        "你是臺灣學校的多語行政助理。"
        "請根據目前這一輪對話中已提供的條件翻譯通知。"
        "若目前對話沒有明確提供目標語言、讀者或語氣，"
        "請直接指出缺少哪些資訊，不要自行假設。"
        "若使用者追問先前問過哪些問題，你只能根據這次 API 呼叫實際附帶的對話歷史回答。"
        "如果這次呼叫沒有附帶更早的提問紀錄，就直接說看不到先前對話。"
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比較連續對話有記憶與無記憶的差異")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    model = create_chat_model(provider)

    # HumanMessage:是使用者的訊息
    # SystemMessage:是系統訊息
    # BaseMessage:是訊息的基底類別
    first_turn = HumanMessage(content=SETUP_REQUEST)
    first_reply = model.invoke([SYSTEM_MESSAGE, first_turn])

    second_turn = HumanMessage(content=FOLLOW_UP_REQUEST)
    without_memory_translation = model.invoke([SYSTEM_MESSAGE, second_turn])

    # 有記憶的情境：把第一輪使用者訊息與模型回覆，一起附回第二輪 Prompt
    with_memory_translation_history: list[BaseMessage] = [
        SYSTEM_MESSAGE,
        first_turn,
        first_reply,
        second_turn,
    ]
    with_memory_translation = model.invoke(with_memory_translation_history)

    recall_turn = HumanMessage(content=RECALL_REQUEST)
    without_memory_recall = model.invoke([SYSTEM_MESSAGE, recall_turn])
    with_memory_recall_history: list[BaseMessage] = [
        SYSTEM_MESSAGE,
        first_turn,
        first_reply,
        second_turn,
        with_memory_translation,
        recall_turn,
    ]
    with_memory_recall = model.invoke(with_memory_recall_history)

    print(f"模型供應商：{provider}")
    print("\n觀察重點：第三輪直接問模型『我先前問過哪些問題？』。")
    print("沒有附回歷史時，它應該只能承認看不到前文；附回歷史時，才列得出前兩輪提問。")

    print("\n[第一輪：先設定條件]")
    print("使用者：")
    print(SETUP_REQUEST)
    print("\n模型：")
    print(first_reply.text)

    print("\n[第二輪：只丟『那這則呢？』加通知]")
    print("無記憶：")
    print("使用者：")
    print(FOLLOW_UP_REQUEST)
    print("\n模型：")
    print(without_memory_translation.text)

    print("\n有記憶：")
    print("使用者：")
    print(FOLLOW_UP_REQUEST)
    print("\n模型：")
    print(with_memory_translation.text)

    print("\n[第三輪：直接追問是否記得前文]")
    print("無記憶：")
    print("使用者：")
    print(RECALL_REQUEST)
    print("\n模型：")
    print(without_memory_recall.text)

    print("\n有記憶：")
    print("使用者：")
    print(RECALL_REQUEST)
    print("\n模型：")
    print(with_memory_recall.text)


if __name__ == "__main__":
    main()