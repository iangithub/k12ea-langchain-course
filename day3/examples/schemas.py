"""PDF RAG 的結構化輸出 Schema。"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = Field(description="引用資料的來源路徑")
    chunk_id: str = Field(description="引用段落的 chunk_id")
    quote: str = Field(description="支持答案的原文短句")


class RagAnswer(BaseModel):
    answer: str = Field(description="只依據檢索內容整理的回答")
    citations: list[Citation] = Field(description="支持答案的來源；無證據時為空陣列")
    insufficient_evidence: bool = Field(description="現有檢索內容是否不足以回答")