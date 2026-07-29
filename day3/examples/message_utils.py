"""將不同模型供應商的訊息內容整理成終端機文字。"""

from __future__ import annotations

from typing import Any


def content_to_text(content: Any) -> str:
    """將字串或 content blocks 轉成可讀文字。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def last_message_text(result: dict[str, Any]) -> str:
    """取出 Agent 執行結果中的最後一則訊息文字。"""
    return content_to_text(result["messages"][-1].content)