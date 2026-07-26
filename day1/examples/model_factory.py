from __future__ import annotations

"""Day 1 共用模型設定：把 provider 判斷與 .env 讀取集中起來。"""

import os
from typing import Literal, cast

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI # 使用 Google Gemini API
from langchain_openai import ChatOpenAI # Azure OpenAI，使用 Azure OpenAI API

Provider = Literal["gemini", "azure"]


class ConfigurationError(RuntimeError):
    """Raised when a required course setting is missing or invalid."""


def _required(name: str) -> str:
    # 範例檔不想各自處理設定錯誤，所以先在 helper 集中檢查。
    value = os.getenv(name, "").strip()
    if not value or value.startswith("replace-with-"):
        raise ConfigurationError(f"請在 .env 設定 {name}")
    return value


def get_provider(override: str | None = None) -> Provider:
    # 優先使用命令列傳入的 provider，沒有才回退到 .env。
    load_dotenv()
    value = (override or os.getenv("MODEL_PROVIDER", "gemini")).strip().lower()
    if value not in {"gemini", "azure"}:
        raise ConfigurationError("MODEL_PROVIDER 只能是 gemini 或 azure")
    return cast(Provider, value)


def create_chat_model(provider: str | None = None) -> BaseChatModel:
    """Create a Gemini or Azure OpenAI chat model from environment settings."""
    # 先讀取 .env，並根據命令列參數或 .env 決定要使用哪一家模型。
    load_dotenv()
    selected = get_provider(provider)

    if selected == "gemini":
        # Day 1 大多數範例都從這裡直接拿 Gemini chat model。
        _required("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=_required("GEMINI_MODEL"),
            timeout=60,
            max_retries=2, # 避免網路不穩時，程式直接失敗
        )

    # Azure 走 v1 API，所以 endpoint 需要先標準化成 /openai/v1/。
    endpoint = _required("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if not endpoint.lower().endswith("/openai/v1"):
        raise ConfigurationError("AZURE_OPENAI_ENDPOINT 必須以 /openai/v1/ 結尾")

    return ChatOpenAI(
        model=_required("AZURE_OPENAI_MODEL"),
        base_url=endpoint + "/",
        api_key=_required("AZURE_OPENAI_API_KEY"),
        timeout=60,
        max_retries=2,
    )


def print_usage_metadata(response: object) -> None:
    # 某些供應商會回傳 token 用量，某些不會；這裡統一成一致的教學輸出。
    usage = getattr(response, "usage_metadata", None)
    if usage:
        print(f"Token 用量：{usage}")
    else:
        print("Token 用量：供應商未回傳用量資訊")
