from __future__ import annotations

"""Day 2 共用模型設定：集中處理 provider、API 金鑰與 embeddings 建立。"""

import os
from typing import Literal, cast

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

Provider = Literal["gemini", "azure"]


class ConfigurationError(ValueError):
    """Raised when a required model setting is missing or invalid."""


def _required(name: str) -> str:
    # 課堂故意在這裡先攔掉假值，讓錯誤集中發生在設定層，不要散落到每支範例。
    value = os.getenv(name, "").strip()
    if not value or value.lower().startswith("replace-with-"):
        raise ConfigurationError(f"請在 .env 設定 {name}")
    return value


def get_provider(override: str | None = None) -> Provider:
    # 優先使用命令列 --provider；沒指定時再回退到 .env 的 MODEL_PROVIDER。
    load_dotenv()
    value = (override or os.getenv("MODEL_PROVIDER", "gemini")).strip().lower()
    if value not in {"gemini", "azure"}:
        raise ConfigurationError("MODEL_PROVIDER 只能是 gemini 或 azure")
    return cast(Provider, value)


def _azure_v1_endpoint() -> str:
    # 課程統一使用 Azure OpenAI v1 API，所以這裡先檢查 endpoint 格式。
    endpoint = _required("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if not endpoint.lower().endswith("/openai/v1"):
        raise ConfigurationError("AZURE_OPENAI_ENDPOINT 必須以 /openai/v1/ 結尾")
    return endpoint + "/"


def create_chat_model(provider: str | None = None) -> BaseChatModel:
    selected = get_provider(provider)
    if selected == "gemini":
        # Gemini 路線只建立 chat model，給 Day 2 各支範例直接拿去問答或做 Agent。
        return ChatGoogleGenerativeAI(
            model=_required("GEMINI_MODEL"),
            google_api_key=_required("GOOGLE_API_KEY"),
            temperature=0,
            timeout=60,
            max_retries=2,
        )

    # Azure 路線沿用同一個 helper，避免每支範例都重複寫 endpoint 與 key 設定。
    return ChatOpenAI(
        model=_required("AZURE_OPENAI_MODEL"),
        base_url=_azure_v1_endpoint(),
        api_key=_required("AZURE_OPENAI_API_KEY"),
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def create_embeddings(provider: str | None = None) -> Embeddings:
    selected = get_provider(provider)
    if selected == "gemini":
        # 向量模型和 chat model 分開建立，因為 Day 2 會同時用到兩種模型。
        return GoogleGenerativeAIEmbeddings(
            model=_required("GEMINI_EMBEDDING_MODEL"),
            google_api_key=_required("GOOGLE_API_KEY"),
        )

    return OpenAIEmbeddings(
        model=_required("AZURE_OPENAI_EMBEDDING_MODEL"),
        base_url=_azure_v1_endpoint(),
        api_key=_required("AZURE_OPENAI_API_KEY"),
        timeout=60,
        max_retries=2,
    )