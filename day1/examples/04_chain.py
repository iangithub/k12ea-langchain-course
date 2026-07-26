"""Chain 範例：先翻譯，再把第一次結果交給第二步校對。"""

from __future__ import annotations

import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from model_factory import create_chat_model, get_provider

NOTICE = "各位家長您好，明天下午兩點將舉行親師座談，請於一點五十分前到校。"

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是臺灣學校的翻譯人員。將內容翻譯成{target_language}，"
            "保留所有日期、時間與專有名詞，不補充原文沒有的資訊。",
        ),
        ("human", "{notice}"),
    ]
)

REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是校務通知校對人員。比對原文與譯文，修正遺漏、錯譯及不適合家長閱讀的語氣。"
            "只輸出校對後的譯文。",
        ),
        ("human", "原文：\n{notice}\n\n待校對譯文：\n{draft}"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="串接翻譯與校對兩條 Chain")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--language", default="日文")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    model = create_chat_model(provider)

    # StrOutputParser 會把模型回覆訊息轉成純文字，方便後續程式使用。
    parser = StrOutputParser()

    # 這裡使用 LCEL 把 Prompt、模型與 Output Parser 串成兩條 Chain，分別負責翻譯與校對。
    translation_chain = TRANSLATE_PROMPT | model | parser
    review_chain = REVIEW_PROMPT | model | parser

    # assign 會保留原始輸入，並把chain做的結果加入 draft 與 final 兩個欄位。
    full_chain = (
        RunnablePassthrough.assign(draft=translation_chain)
        | RunnablePassthrough.assign(final=review_chain)
    )
    result = full_chain.invoke({"target_language": args.language, "notice": NOTICE})

    print(f"模型供應商：{provider}")
    print("\n第一次翻譯：")
    print(result["draft"])
    print("\n校對後結果：")
    print(result["final"])


if __name__ == "__main__":
    main()
