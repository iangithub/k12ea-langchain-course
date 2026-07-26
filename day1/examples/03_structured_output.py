"""Structured Output 範例：讓模型回傳固定欄位，方便後續程式接手。"""

from __future__ import annotations

import argparse

from langchain_core.prompts import ChatPromptTemplate
from model_factory import create_chat_model, get_provider
from schemas import TranslationResult

NOTICE = "各位家長您好，明天下午兩點將舉行親師座談，請於一點五十分前到校。"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是臺灣學校的多語行政助理。翻譯時不可自行補充原文沒有的資訊。"
            "不明確的日期、地點或稱謂必須列入人工確認項目。",
        ),
        (
            "human",
            "請將下列通知翻譯成{target_language}，讀者是{audience}：\n{notice}",
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Pydantic Schema 取得結構化翻譯")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--language", default="日文")
    parser.add_argument("--audience", default="學生家長")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    model = create_chat_model(provider)

    # with_structured_output 會要求模型依照 TranslationResult 的欄位回傳資料，
    # 這樣學員就能直接看到「譯文、術語、待確認事項」被拆成不同欄位。
    structured_model = model.with_structured_output(TranslationResult)
    prompt_value = PROMPT.invoke(
        {
            "target_language": args.language,
            "audience": args.audience,
            "notice": NOTICE,
        }
    )

    # 這裡拿到的不是純文字，而是 Pydantic 物件，可直接轉成 JSON 檢查各欄位。
    result = structured_model.invoke(prompt_value.to_messages())
    print(f"模型供應商：{provider}")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
