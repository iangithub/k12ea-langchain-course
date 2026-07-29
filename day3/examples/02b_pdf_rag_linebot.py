"""使用既有 PDF 向量資料庫回答 LINE 文字訊息。

LINE Platform 經 ngrok 對 /callback 送出 Webhook。程式先驗證簽章，
再檢索既有 PDF 向量資料庫，最後使用 Reply Message API 回覆。

執行：uv run examples/02b_pdf_rag_linebot.py
公開本機服務：ngrok http 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

from examples.line_utils import send_text_reply, verify_signature
from examples.model_factory import create_chat_model, create_embeddings, get_provider
from examples.pdf_rag import (
    answer_pdf_question,
    format_line_answer,
    open_pdf_vector_store,
)

ANSWER_TIMEOUT_SECONDS = 25

MODEL: Any = None
VECTOR_STORE: Any = None
QDRANT_CLIENT: Any = None
PROVIDER_OVERRIDE: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, QDRANT_CLIENT, VECTOR_STORE

    load_dotenv()
    provider = get_provider(PROVIDER_OVERRIDE)
    MODEL = create_chat_model(provider)
    embeddings = create_embeddings(provider)
    VECTOR_STORE, QDRANT_CLIENT = open_pdf_vector_store(embeddings)
    print(f"PDF RAG LINE Bot 已啟動（provider={provider}，使用既有向量資料庫）")
    try:
        yield
    finally:
        if QDRANT_CLIENT is not None:
            QDRANT_CLIENT.close()
        QDRANT_CLIENT = None
        VECTOR_STORE = None
        MODEL = None


app = FastAPI(title="教育人員任用條例 PDF RAG LINE Bot", lifespan=lifespan)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="啟動教育人員任用條例 PDF RAG LINE Bot")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    return parser.parse_args()


async def answer_question(text: str) -> str:
    """執行 PDF RAG，並把逾時與例外轉成適合 LINE 顯示的訊息。"""
    try:
        if MODEL is None or VECTOR_STORE is None:
            raise RuntimeError("PDF RAG 尚未初始化")

        result = await asyncio.wait_for(
            asyncio.to_thread(answer_pdf_question, MODEL, VECTOR_STORE, text),
            timeout=ANSWER_TIMEOUT_SECONDS,
        )
        if result.citation_issues:
            print("[引用驗證未通過]")
            for issue in result.citation_issues:
                print(f"- {issue}")
        return format_line_answer(result)
    except TimeoutError:
        return "抱歉，這個問題處理時間過長，請稍後再試。"
    except Exception as error:  # noqa: BLE001
        print(f"[錯誤] {type(error).__name__}: {error}")
        return "系統暫時無法查詢法規，請稍後再試或直接查閱法規原文。"


@app.post("/callback")
async def callback(request: Request, x_line_signature: str = Header(default="")) -> dict:
    body = await request.body()
    channel_secret = os.getenv("LINE_CHANNEL_SECRET", "")
    if not channel_secret or channel_secret.startswith("replace-with-"):
        raise HTTPException(status_code=500, detail="請先在 .env 設定 LINE_CHANNEL_SECRET")
    if not verify_signature(channel_secret, body, x_line_signature):
        raise HTTPException(status_code=403, detail="簽章驗證失敗")

    channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    payload = json.loads(body)
    for event in payload.get("events", []):
        if event.get("type") != "message":
            continue
        message = event.get("message", {})
        if message.get("type") != "text":
            send_text_reply(
                channel_access_token,
                event["replyToken"],
                "目前只支援文字訊息。",
            )
            continue

        print(f"[收到訊息] {message['text']}")
        answer = await answer_question(message["text"])
        send_text_reply(channel_access_token, event["replyToken"], answer)
    return {"status": "ok"}


def main() -> None:
    global PROVIDER_OVERRIDE

    args = parse_args()
    PROVIDER_OVERRIDE = args.provider
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()