from __future__ import annotations

import os
from typing import Protocol

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .dashscope import DashScopeError, DashScopeFunASRClient


class Transcriber(Protocol):
    async def transcribe(
        self,
        *,
        audio_bytes: bytes,
        mime_type: str,
        filename: str,
        model: str,
        language: str | None,
    ) -> str: ...


def create_app(transcriber: Transcriber | None = None) -> FastAPI:
    app = FastAPI(
        title="Fun-ASR OpenAI Proxy",
        description="OpenAI-compatible audio transcription proxy for Bailian Fun-ASR.",
        version="0.1.0",
    )

    configured_transcriber = transcriber

    def get_transcriber() -> Transcriber:
        if configured_transcriber is not None:
            return configured_transcriber
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="DASHSCOPE_API_KEY is not set on the proxy server.",
            )
        return DashScopeFunASRClient(api_key=api_key)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
                }
            ],
        }

    @app.post("/audio/transcriptions")
    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: str = Form("fun-asr"),
        language: str | None = Form(None),
        response_format: str | None = Form(None),
        service: Transcriber = Depends(get_transcriber),
    ):
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        try:
            text = await service.transcribe(
                audio_bytes=audio_bytes,
                mime_type=file.content_type or "application/octet-stream",
                filename=file.filename or "audio",
                model=model,
                language=language,
            )
        except DashScopeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        normalized_format = (response_format or "json").lower()
        if normalized_format == "text":
            return PlainTextResponse(text)
        if normalized_format in {"json", "verbose_json"}:
            return JSONResponse({"text": text})
        raise HTTPException(
            status_code=400,
            detail="Unsupported response_format. Use json, text, or verbose_json.",
        )

    return app


app = create_app()
