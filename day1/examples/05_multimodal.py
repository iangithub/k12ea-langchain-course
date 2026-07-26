"""多模態範例：同一支程式切換語音、圖片、影片等不同任務。"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from model_factory import ConfigurationError, create_chat_model, get_provider
from openai import APIStatusError, AzureOpenAI, OpenAI

# 定義輸出資料夾，存放 TTS 與生圖的結果檔案。
# Path(__file__).resolve().parent 代表目前這支程式所在的目錄。
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# MIME_TYPES 是程式內建的副檔名對應表，若要支援更多格式，可自行擴充。
MIME_TYPES = {
    ".mp3": "audio/mp3",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
}


def required_env(name: str) -> str:
    load_dotenv()
    value = os.getenv(name, "").strip()
    if not value or value.startswith("replace-with-"):
        raise ConfigurationError(f"請在 .env 設定 {name}")
    return value


## 讀檔並轉成 base64，回傳 (base64, mime_type)。
def read_file_base64(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise SystemExit(f"找不到檔案：{path}")
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        supported = ", ".join(sorted(MIME_TYPES))
        raise SystemExit(f"不支援的副檔名：{path.suffix}（支援 {supported}）")
    return base64.b64encode(path.read_bytes()).decode("utf-8"), mime


def azure_openai_client() -> OpenAI:
    return OpenAI(
        base_url=required_env("AZURE_OPENAI_ENDPOINT"),
        api_key=required_env("AZURE_OPENAI_API_KEY"),
    )


def azure_audio_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=required_env("AZURE_OPENAI_API_KEY"),
        api_version="2024-10-21",
        azure_endpoint=required_env("AZURE_OPENAI_RESOURCE_ENDPOINT"),
    )


# Gemini 的多模態訊息格式與一般文字訊息不同，這裡用一個小函式包裝。
def gemini_media_message(prompt: str, data: str, mime: str) -> HumanMessage:
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "media", "data": data, "mime_type": mime},
        ]
    )

# 語音轉文字（Speech-to-Text）
def run_stt(provider: str, path: Path) -> None:
    if provider == "gemini":
        # Gemini 走原生多模態路線：把音訊直接當成訊息內容送進 chat model。
        data, mime = read_file_base64(path)
        model = create_chat_model("gemini")
        response = model.invoke(
            [gemini_media_message("請將這段語音逐字轉寫成繁體中文文字，不要加入評論。", data, mime)]
        )
        print(response.text)
        return

    # Azure 的 STT 走 deployment-based audio/transcriptions 端點。
    client = azure_audio_client()
    with path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=required_env("AZURE_OPENAI_STT_MODEL"),
            file=audio_file,
        )
    print(result.text)

# 語音合成（Text-to-Speech）
def run_tts(provider: str, text: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    if provider == "gemini":
        output_path = OUTPUT_DIR / "tts_output_gemini.wav"
        model = ChatGoogleGenerativeAI(
            model=required_env("GEMINI_TTS_MODEL"),
            timeout=60,
            max_retries=2,
        )
        response = model.invoke(text)
        audio_data = response.additional_kwargs.get("audio")
        if not isinstance(audio_data, bytes):
            raise SystemExit("Gemini TTS 未回傳音訊資料，請確認模型名稱與帳號權限。")
        output_path.write_bytes(audio_data)
        print(f"已產生語音檔：{output_path}")
        print("提醒：免費層有頻率限制，且內容可能用於改善 Google 產品。")
        return

    output_path = OUTPUT_DIR / "tts_output.mp3"
    client = azure_openai_client()
    with client.audio.speech.with_streaming_response.create(
        model=required_env("AZURE_OPENAI_TTS_MODEL"),
        voice="alloy",
        input=text,
    ) as response:
        response.stream_to_file(output_path)
    print(f"已產生語音檔：{output_path}")
    print("提醒：對外使用時，必須向聽眾揭露這是 AI 合成語音。")

# 圖片理解（Image Understanding）
def run_image(provider: str, path: Path, question: str) -> None:
    data, mime = read_file_base64(path)
    if provider == "gemini":
        # Gemini 可以直接把圖片當成多模態訊息的一部分。
        model = create_chat_model("gemini")
        response = model.invoke([gemini_media_message(question, data, mime)])
    else:
        # Azure 的圖片理解仍走 chat model，但圖片要包成 image_url block。
        model = create_chat_model("azure")
        response = model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"},
                        },
                    ]
                )
            ]
        )
    print(response.text)

# 圖片生成（Image Generation）
def run_imagegen(provider: str, prompt: str) -> None:
    if provider == "gemini":
        print("本範例的生圖以 Azure 路徑示範（--provider azure）。")
        print("Gemini 生圖需使用具生圖能力的模型與 SDK，課堂以概念與對照體驗為主。")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "imagegen_output.png"
    client = azure_openai_client()
    result = client.images.generate(
        model=required_env("AZURE_OPENAI_IMAGE_MODEL"),
        prompt=prompt,
        size="1024x1024",
    )
    output_path.write_bytes(base64.b64decode(result.data[0].b64_json))
    print(f"已產生圖片：{output_path}")
    print("提醒：用於教材或文宣時，仍須確認內容合宜與標示方式。")

# 影片理解（Video Understanding）
def run_video(provider: str, path: Path, question: str) -> None:
    if provider == "azure":
        print("Azure OpenAI 沒有直接的影片輸入介面。課堂在這裡只觀察組合思路：")
        print("1. 抽影格交給圖片理解")
        print("2. 聲音軌交給 STT")
        print("3. 最後交給 Chat Model 彙整")
        print("若想直接體驗影片輸入，請改用 --provider gemini。")
        return

    data, mime = read_file_base64(path)
    model = create_chat_model("gemini")
    response = model.invoke([gemini_media_message(question, data, mime)])
    print(response.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="體驗模型處理語音、圖片與影片的能力")
    parser.add_argument(
        "--task", required=True, choices=["stt", "tts", "image", "imagegen", "video"]
    )
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--file", help="stt、image、video 任務的輸入檔案路徑")
    parser.add_argument("--text", help="tts 與 imagegen 任務的輸入文字")
    parser.add_argument(
        "--question",
        default="請以繁體中文說明這份內容的重點，若有文字請一併整理。",
        help="image 與 video 任務要問模型的問題",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)

    # 只有 stt、image、video 這三類任務需要實際讀檔，先在進入主流程前檢查。
    if args.task in {"stt", "image", "video"}:
        if not args.file:
            raise SystemExit(f"--task {args.task} 需要 --file 指定輸入檔案")
        path = Path(args.file)

    try:
        # 同一支程式用 task 分流，方便學員從同一份程式碼比較不同模態的處理方式。
        if args.task == "stt":
            run_stt(provider, path)
        elif args.task == "tts":
            run_tts(provider, args.text or "各位家長好，這是一則由 AI 合成語音朗讀的測試通知。")
        elif args.task == "image":
            run_image(provider, path, args.question)
        elif args.task == "imagegen":
            run_imagegen(provider, args.text or "溫馨的校園閱讀週宣傳插圖，水彩風格，不含文字")
        else:
            run_video(provider, path, args.question)
    except APIStatusError as error:
        if error.status_code == 404:
            raise SystemExit(
                "404 DeploymentNotFound：.env 指定的 Azure Deployment 在此帳號不存在。\n"
                "請對照 .env 的 AZURE_OPENAI_*_MODEL 欄位與 Azure 入口網站的部署清單；\n"
                "課程帳號未必開通所有多模態部署，開通範圍以當日公布為準。"
            ) from error
        raise


if __name__ == "__main__":
    main()
