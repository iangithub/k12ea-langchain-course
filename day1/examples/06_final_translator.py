"""Day 1 最終成果：把可調參數與結構化輸出組回完整翻譯器。"""

from __future__ import annotations

import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from model_factory import create_chat_model, get_provider
from schemas import TranslationResult

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是臺灣學校的多語行政助理。請依指定讀者與語氣翻譯通知，"
            "完整保留原文事實，不可自行增加日期、地點、規定或聯絡方式。只輸出譯文。",
        ),
        (
            "human",
            "目標語言：{target_language}\n讀者：{audience}\n語氣：{tone}\n\n通知：\n{notice}",
        ),
    ]
)

REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是校務通知校對人員。比對原文與翻譯初稿，修正遺漏、錯譯及語氣問題。"
            "將校對後譯文、重要校園用語，以及原文中需要人工確認的資訊整理成指定欄位。",
        ),
        (
            "human",
            "目標語言：{target_language}\n讀者：{audience}\n語氣：{tone}\n\n"
            "原文：\n{notice}\n\n翻譯初稿：\n{draft}",
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 1 最終成果：可切換模型的校園翻譯器")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--language", default="日文")
    parser.add_argument("--audience", default="學生家長")
    parser.add_argument("--tone", default="清楚、親切且正式")
    parser.add_argument(
        "--text",
        default="各位家長您好，明天下午兩點將舉行親師座談，請於一點五十分前到校。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    model = create_chat_model(provider)

    # 這裡使用 LCEL 把 Prompt、模型與 Output Parser 串成負責翻譯的 Chain。
    translation_chain = TRANSLATE_PROMPT | model | StrOutputParser()
    structured_model = model.with_structured_output(TranslationResult)

    # 第一階段產生 draft；第二階段取得原始輸入與 draft，再回傳結構化結果。
    full_chain = (
        RunnablePassthrough.assign(draft=translation_chain)
        | REVIEW_PROMPT
        | structured_model
    )
    result = full_chain.invoke(
        {
            "target_language": args.language,
            "audience": args.audience,
            "tone": args.tone,
            "notice": args.text,
        }
    )
    print(f"模型供應商：{provider}")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
