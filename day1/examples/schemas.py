from __future__ import annotations

from pydantic import BaseModel, Field

from typing import Literal


class TranslationResult(BaseModel):
    """Structured result for a translated school notice."""

    translation: str = Field(description="完成翻譯與語氣調整後的通知")
    key_terms: list[str] = Field(description="需要維持一致翻譯的重要校園用語")
    items_to_verify: list[str] = Field(description="原文不明確、需要人工確認的資訊；若無則為空陣列")

# class TranslationResult(BaseModel):
#     translation: str = Field(description="完成翻譯與語氣調整後的通知")
#     key_terms: list[str] = Field(description="需要維持一致翻譯的重要校園用語")
#     items_to_verify: list[str] = Field(
#         description="原文不明確、需要人工確認的資訊；若無則為空陣列"
#     )
#     risk_level: Literal["low", "medium", "high"] = Field(
#         description="依日期、金額、資格條件與原文歧義判斷人工複核優先度"
#     )