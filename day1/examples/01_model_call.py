from __future__ import annotations

"""最小模型呼叫範例：只示範訊息怎麼送進模型。"""

import argparse

from model_factory import create_chat_model, get_provider, print_usage_metadata

NOTICE = """各位家長您好，明天下午兩點將舉行親師座談，請於一點五十分前到校。"""



def parse_args() -> argparse.Namespace:
    # 解析命令列參數，決定要使用哪一家模型。
    parser = argparse.ArgumentParser(description="使用 Gemini 或 Azure OpenAI 改寫親師通知")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 先根據命令列參數或 .env 決定要使用哪一家模型。
    provider = get_provider(args.provider)
    model = create_chat_model(provider)

    # 這裡故意只保留最小的兩則訊息：
    # system 負責定義規則，human 放入真正要處理的通知內容。
    messages = [
        (
            "system",
            "你是臺灣學校的行政助理。請保留原意，將通知改寫得清楚、禮貌，"
            "不要加入原文沒有的日期、地點或規定。",
        ),
        ("human", NOTICE),
    ]

    response = model.invoke(messages)
    print(f"模型供應商：{provider}")
    print("\n改寫結果：")
    print(response.text)
    print()
    print_usage_metadata(response)


if __name__ == "__main__":
    main()
