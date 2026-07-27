from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = Field(description="引用資料的來源路徑")
    chunk_id: str = Field(description="引用段落的 chunk_id")
    quote: str = Field(description="支持答案的原文短句")


class RagAnswer(BaseModel):
    answer: str = Field(description="只依據檢索內容整理的回答")
    citations: list[Citation] = Field(description="支持答案的來源；無證據時為空陣列")
    insufficient_evidence: bool = Field(description="現有檢索內容是否不足以回答")


class EvidenceReview(BaseModel):
    grounded: bool = Field(description="草稿是否完全受到證據支持")
    issues: list[str] = Field(description="未被來源支持或引用不完整之處")
    revised_answer: str = Field(description="依證據修正後的答案；不需修正時保留原答案")