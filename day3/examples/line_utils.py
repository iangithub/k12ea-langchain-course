"""LINE Webhook 簽章與回覆的共用函式。

簽章演算法依 LINE 官方文件：以 Channel Secret 為金鑰，
對 request body 計算 HMAC-SHA256，再以 Base64 編碼，
與 X-Line-Signature 標頭比對。
"""

from __future__ import annotations

import base64
import hashlib
import hmac


def compute_signature(channel_secret: str, body: bytes) -> str:
    # LINE 的簽章就是：Secret + 原始 request body 做 HMAC-SHA256，再轉 Base64。
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    # 收到請求後重算一次簽章，和標頭裡送來的值做安全比對。
    expected = compute_signature(channel_secret, body)
    return hmac.compare_digest(expected, signature)


def send_text_reply(channel_access_token: str, reply_token: str, text: str) -> None:
    """透過 LINE Messaging API 回覆文字。"""
    if not channel_access_token or channel_access_token.startswith("replace-with-"):
        raise ValueError("請先在 .env 設定 LINE_CHANNEL_ACCESS_TOKEN")

    from linebot.v3.messaging import (
        ApiClient,
        Configuration,
        MessagingApi,
        ReplyMessageRequest,
        TextMessage,
    )

    configuration = Configuration(access_token=channel_access_token)
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text[:4900])],
            )
        )
