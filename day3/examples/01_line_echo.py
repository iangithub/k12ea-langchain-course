"""LINE Webhook 基礎版：透過 ngrok 接收真實 LINE 事件並回覆原文。

流程：LINE Platform 經 ngrok 對 /callback 送出 POST，
程式先以 Channel Secret 驗證 X-Line-Signature 簽章，
再逐一處理事件；文字訊息以「你說的是：…」回覆。

執行：uv run examples/01_line_echo.py
公開本機服務：ngrok http 8000
"""

from __future__ import annotations

import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

from examples.line_utils import send_text_reply, verify_signature

load_dotenv()
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

app = FastAPI(title="校園 LINE Bot：Echo 版")


@app.post("/callback")
async def callback(request: Request, x_line_signature: str = Header(default="")) -> dict:
    body = await request.body()
    if not CHANNEL_SECRET or CHANNEL_SECRET.startswith("replace-with-"):
        raise HTTPException(status_code=500, detail="請先在 .env 設定 LINE_CHANNEL_SECRET")
    if not verify_signature(CHANNEL_SECRET, body, x_line_signature):
        raise HTTPException(status_code=403, detail="簽章驗證失敗")

    payload = json.loads(body)
    for event in payload.get("events", []):
        if event.get("type") != "message":
            continue
        message = event.get("message", {})
        if message.get("type") != "text":
            send_text_reply(CHANNEL_ACCESS_TOKEN, event["replyToken"], "目前只支援文字訊息。")
            continue

        print(f"[收到訊息] {message['text']}")
        send_text_reply(CHANNEL_ACCESS_TOKEN, event["replyToken"], f"你說的是：{message['text']}")
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)