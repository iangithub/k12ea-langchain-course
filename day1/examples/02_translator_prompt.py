"""Prompt Template 範例：把語言、讀者與語氣改成可替換的欄位。"""

from __future__ import annotations

import argparse

from langchain_core.prompts import ChatPromptTemplate
from model_factory import create_chat_model, get_provider

NOTICE = "各位家長您好，明天下午兩點將舉行親師座談，請於一點五十分前到校。"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是臺灣學校的多語行政助理。請將通知翻譯成{target_language}，"
            "讀者是{audience}，語氣為{tone}。保留日期、時間、地點與專有名詞；"
            "不要補充原文沒有的資訊。只輸出翻譯結果。",
        ),
        ("human", "待處理通知：\n{notice}"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修改 Prompt 參數以翻譯親師通知")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--language", default="日文")
    parser.add_argument("--audience", default="學生家長")
    parser.add_argument("--tone", default="清楚、親切且正式")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    model = create_chat_model(provider)

    # 先把模板中的變數填入，取得這一次真正要送給模型的訊息內容。
    prompt_value = PROMPT.invoke(
        {
            "target_language": args.language,
            "audience": args.audience,
            "tone": args.tone,
            "notice": NOTICE,
        }
    )

    # to_messages() 會把 PromptTemplate 轉成模型可直接吃的訊息串列。
    response = model.invoke(prompt_value.to_messages())
    print(f"模型供應商：{provider}")
    print(response.text)


if __name__ == "__main__":
    main()
