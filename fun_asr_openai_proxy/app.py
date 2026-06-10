from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os
from typing import Protocol

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dashscope import DashScopeError, DashScopeFunASRClient, VocabularyWord
from .doubao import DoubaoASRClient, DoubaoASRError
from .providers import ModelRoutingTranscriber


class Transcriber(Protocol):
    async def transcribe(
        self,
        *,
        audio_bytes: bytes,
        mime_type: str,
        filename: str,
        model: str,
        language: str | None,
        language_hints: list[str] | None,
        vocabulary_id: str | None,
    ) -> str: ...

    async def create_vocabulary(
        self,
        *,
        prefix: str,
        target_model: str,
        vocabulary: list[VocabularyWord],
    ) -> dict[str, object]: ...


class VocabularyWordRequest(BaseModel):
    text: str
    weight: int = 4
    lang: str = "zh"


class CreateVocabularyRequest(BaseModel):
    prefix: str
    target_model: str = "fun-asr"
    vocabulary: list[VocabularyWordRequest] = Field(..., min_length=1)


class RuntimeSettings(BaseModel):
    language_hints: list[str] | None = None
    vocabulary_id: str | None = None


MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100 MB


def create_app(
    transcriber: Transcriber | None = None,
    *,
    initial_language_hints: list[str] | None = None,
    initial_vocabulary_id: str | None = None,
) -> FastAPI:
    shared_http_client = httpx.AsyncClient(timeout=60)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await shared_http_client.aclose()

    app = FastAPI(
        title="ASR Bridge",
        description="OpenAI-compatible audio transcription proxy for Bailian Fun-ASR and Doubao ASR.",
        version="0.1.0",
        lifespan=lifespan,
    )
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    configured_transcriber = transcriber
    log_events: deque[dict[str, object]] = deque(maxlen=300)
    log_subscribers: set[asyncio.Queue[dict[str, object]]] = set()
    log_sequence = 0
    runtime_settings = RuntimeSettings(
        language_hints=initial_language_hints
        if initial_language_hints is not None
        else _parse_language_hints(os.environ.get("DASHSCOPE_LANGUAGE_HINTS")),
        vocabulary_id=initial_vocabulary_id
        if initial_vocabulary_id is not None
        else _clean_optional_value(os.environ.get("DASHSCOPE_VOCABULARY_ID")),
    )

    def add_log(
        level: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        nonlocal log_sequence
        log_sequence += 1
        event: dict[str, object] = {
            "id": log_sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            "details": details or {},
        }
        log_events.append(event)
        for queue in list(log_subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except asyncio.QueueEmpty:
                    pass

    add_log("info", "服务已启动")

    def get_transcriber() -> Transcriber:
        if configured_transcriber is not None:
            return configured_transcriber

        dashscope_api_key = _clean_optional_value(os.environ.get("DASHSCOPE_API_KEY"))
        dashscope = (
            DashScopeFunASRClient(
                api_key=dashscope_api_key,
                http_client=shared_http_client,
                default_language_hints=runtime_settings.language_hints,
                default_vocabulary_id=runtime_settings.vocabulary_id,
            )
            if dashscope_api_key
            else None
        )

        doubao_api_key = _clean_optional_value(os.environ.get("DOUBAO_API_KEY"))
        doubao_app_id = _clean_optional_value(os.environ.get("DOUBAO_APP_ID"))
        doubao_access_token = _clean_optional_value(
            os.environ.get("DOUBAO_ACCESS_TOKEN")
        )
        doubao = None
        if doubao_api_key or (doubao_app_id and doubao_access_token):
            doubao = DoubaoASRClient(
                api_key=doubao_api_key,
                app_id=doubao_app_id,
                access_token=doubao_access_token,
                resource_id=_clean_optional_value(
                    os.environ.get("DOUBAO_RESOURCE_ID")
                )
                or "volc.seedasr.auc",
                http_client=shared_http_client,
            )

        return ModelRoutingTranscriber(dashscope=dashscope, doubao=doubao)

    @app.get("/", include_in_schema=False)
    async def admin_ui() -> FileResponse:
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(
            os.path.join(static_dir, "favicon.png"),
            media_type="image/png",
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/settings")
    async def get_settings() -> RuntimeSettings:
        return runtime_settings

    @app.put("/api/settings")
    async def update_settings(settings: RuntimeSettings) -> RuntimeSettings:
        runtime_settings.language_hints = settings.language_hints
        runtime_settings.vocabulary_id = settings.vocabulary_id
        add_log(
            "info",
            "运行配置已更新",
            {
                "language_hints": settings.language_hints,
                "vocabulary_id_set": bool(settings.vocabulary_id),
            },
        )
        return runtime_settings

    @app.get("/api/logs")
    async def get_logs(limit: int = 200) -> dict[str, object]:
        normalized_limit = max(1, min(limit, 300))
        return {"data": list(log_events)[-normalized_limit:]}

    @app.get("/api/logs/stream")
    async def stream_logs() -> StreamingResponse:
        async def event_stream():
            queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=100)
            for event in list(log_events):
                yield _sse_event(event)
            log_subscribers.add(queue)
            try:
                while True:
                    event = await queue.get()
                    yield _sse_event(event)
            finally:
                log_subscribers.discard(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/models")
    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "fun-asr",
                    "object": "model",
                    "created": 0,
                    "owned_by": "alibaba-bailian",
                },
                {
                    "id": "doubao-asr",
                    "object": "model",
                    "created": 0,
                    "owned_by": "volcengine-doubao",
                },
            ],
        }

    @app.post("/audio/transcriptions")
    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: str = Form("fun-asr"),
        language: str | None = Form(None),
        language_hints: str | None = Form(None),
        vocabulary_id: str | None = Form(None),
        response_format: str | None = Form(None),
        service: Transcriber = Depends(get_transcriber),
    ):
        audio_bytes = await file.read()
        if not audio_bytes:
            add_log(
                "warn",
                "转写失败",
                {"model": model, "filename": file.filename or "audio", "reason": "empty_file"},
            )
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        if len(audio_bytes) > MAX_AUDIO_SIZE:
            max_mb = MAX_AUDIO_SIZE // (1024 * 1024)
            add_log(
                "warn",
                "转写失败",
                {
                    "model": model,
                    "filename": file.filename or "audio",
                    "reason": "file_too_large",
                    "bytes": len(audio_bytes),
                },
            )
            raise HTTPException(
                status_code=413,
                detail=f"Audio file is too large ({len(audio_bytes)} bytes). Maximum allowed size is {max_mb} MB.",
            )

        log_details = {
            "model": model,
            "filename": file.filename or "audio",
            "mime_type": file.content_type or "application/octet-stream",
            "bytes": len(audio_bytes),
        }
        add_log("info", "转写开始", log_details)
        try:
            text = await service.transcribe(
                audio_bytes=audio_bytes,
                mime_type=file.content_type or "application/octet-stream",
                filename=file.filename or "audio",
                model=model,
                language=language,
                language_hints=_parse_language_hints(language_hints)
                or runtime_settings.language_hints,
                vocabulary_id=_clean_optional_value(vocabulary_id)
                or runtime_settings.vocabulary_id,
            )
        except (DashScopeError, DoubaoASRError) as exc:
            add_log("error", "转写失败", {**log_details, "error": str(exc)[:300]})
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        add_log("info", "转写成功", {**log_details, "text_length": len(text)})
        normalized_format = (response_format or "json").lower()
        if normalized_format == "text":
            return PlainTextResponse(text)
        if normalized_format in {"json", "verbose_json"}:
            return JSONResponse({"text": text})
        raise HTTPException(
            status_code=400,
            detail="Unsupported response_format. Use json, text, or verbose_json.",
        )

    @app.post("/audio/vocabularies")
    @app.post("/v1/audio/vocabularies")
    async def create_vocabulary(
        request: CreateVocabularyRequest,
        service: Transcriber = Depends(get_transcriber),
    ) -> JSONResponse:
        add_log(
            "info",
            "创建热词表开始",
            {
                "prefix": request.prefix,
                "target_model": request.target_model,
                "word_count": len(request.vocabulary),
            },
        )
        try:
            result = await service.create_vocabulary(
                prefix=request.prefix,
                target_model=request.target_model,
                vocabulary=[
                    VocabularyWord(
                        text=word.text,
                        weight=word.weight,
                        lang=word.lang,
                    )
                    for word in request.vocabulary
                ],
            )
        except (DashScopeError, DoubaoASRError) as exc:
            add_log(
                "error",
                "创建热词表失败",
                {
                    "prefix": request.prefix,
                    "target_model": request.target_model,
                    "error": str(exc)[:300],
                },
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        add_log(
            "info",
            "创建热词表成功",
            {
                "prefix": request.prefix,
                "target_model": request.target_model,
                "vocabulary_id": result.get("vocabulary_id"),
            },
        )
        return JSONResponse(result)

    return app


def _parse_language_hints(value: str | None) -> list[str] | None:
    if not value:
        return None
    raw_value = value.strip()
    if raw_value.startswith("["):
        try:
            decoded = json.loads(raw_value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            hints = [str(item).strip() for item in decoded if str(item).strip()]
            return hints or None

    hints = [part.strip() for part in raw_value.split(",") if part.strip()]
    return hints or None


def _clean_optional_value(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    return cleaned or None


def _sse_event(event: dict[str, object]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


app = create_app()
