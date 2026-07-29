from __future__ import annotations

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


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.lower().startswith("replace-with-"):
        raise ConfigurationError(f"請在 .env 設定 {name}")
    return value


def get_provider(override: str | None = None) -> Provider:
    load_dotenv()
    value = (override or os.getenv("MODEL_PROVIDER", "gemini")).strip().lower()
    if value not in {"gemini", "azure"}:
        raise ConfigurationError("MODEL_PROVIDER 只能是 gemini 或 azure")
    return cast(Provider, value)


def azure_v1_endpoint() -> str:
    endpoint = required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if not endpoint.lower().endswith("/openai/v1"):
        raise ConfigurationError("AZURE_OPENAI_ENDPOINT 必須以 /openai/v1/ 結尾")
    return endpoint + "/"


def create_chat_model(provider: str | None = None) -> BaseChatModel:
    selected = get_provider(provider)
    if selected == "gemini":
        return ChatGoogleGenerativeAI(
            model=required_env("GEMINI_MODEL"),
            google_api_key=required_env("GOOGLE_API_KEY"),
            temperature=0,
            timeout=60,
            max_retries=2,
        )

    return ChatOpenAI(
        model=required_env("AZURE_OPENAI_MODEL"),
        base_url=azure_v1_endpoint(),
        api_key=required_env("AZURE_OPENAI_API_KEY"),
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def create_embeddings(provider: str | None = None) -> Embeddings:
    selected = get_provider(provider)
    if selected == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=required_env("GEMINI_EMBEDDING_MODEL"),
            google_api_key=required_env("GOOGLE_API_KEY"),
        )

    return OpenAIEmbeddings(
        model=required_env("AZURE_OPENAI_EMBEDDING_MODEL"),
        base_url=azure_v1_endpoint(),
        api_key=required_env("AZURE_OPENAI_API_KEY"),
        timeout=60,
        max_retries=2,
    )
