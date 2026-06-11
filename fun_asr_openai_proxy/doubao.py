from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class DoubaoASRError(RuntimeError):
    """Raised when Doubao ASR returns an error or no usable transcript."""


@dataclass
class DoubaoASRClient:
    api_key: str | None = None
    app_id: str | None = None
    access_token: str | None = None
    resource_id: str = "volc.seedasr.auc"
    base_url: str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel"
    http_client: httpx.AsyncClient | None = None
    poll_interval_seconds: float = 1.0
    timeout_seconds: float = 300.0

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
    ) -> str:
        del model, vocabulary_id
        self._validate_credentials()

        client = self.http_client or httpx.AsyncClient(timeout=60)
        close_client = self.http_client is None
        request_id = str(uuid.uuid4())
        try:
            await self._submit_task(
                client=client,
                request_id=request_id,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                filename=filename,
                language=self._effective_language(
                    language=language,
                    language_hints=language_hints,
                ),
            )
            return await self._wait_for_result(client=client, request_id=request_id)
        except DoubaoASRError:
            raise
        except httpx.HTTPError as exc:
            raise DoubaoASRError(
                f"Doubao ASR network error ({type(exc).__name__}): {exc or 'no detail'}"
            ) from exc
        finally:
            if close_client:
                await client.aclose()

    async def _submit_task(
        self,
        *,
        client: httpx.AsyncClient,
        request_id: str,
        audio_bytes: bytes,
        mime_type: str,
        filename: str,
        language: str | None,
    ) -> None:
        audio: dict[str, object] = {
            "format": self._audio_format(mime_type=mime_type, filename=filename),
            "data": base64.b64encode(audio_bytes).decode("ascii"),
        }
        mapped_language = self._doubao_language(language)
        if mapped_language:
            audio["language"] = mapped_language

        response = await client.post(
            f"{self.base_url}/submit",
            headers=self._headers(request_id=request_id),
            json={
                "user": {"uid": self.app_id or "openai-proxy"},
                "audio": audio,
                "request": {
                    "model_name": "bigmodel",
                    "enable_itn": True,
                    "enable_punc": True,
                    "show_utterances": True,
                },
            },
        )
        self._ensure_ok(response=response, phase="submit")

    async def _wait_for_result(
        self, *, client: httpx.AsyncClient, request_id: str
    ) -> str:
        deadline = time.monotonic() + self.timeout_seconds
        last_response: httpx.Response | None = None
        consecutive_network_errors = 0

        while time.monotonic() < deadline:
            try:
                response = await client.post(
                    f"{self.base_url}/query",
                    headers=self._headers(request_id=request_id),
                    json={},
                )
                consecutive_network_errors = 0
            except httpx.HTTPError as exc:
                consecutive_network_errors += 1
                logger.warning(
                    "Doubao poll network error (#%d): %s: %s",
                    consecutive_network_errors, type(exc).__name__, exc or "no detail",
                )
                if consecutive_network_errors >= 5:
                    raise DoubaoASRError(
                        f"Doubao ASR network error after {consecutive_network_errors} "
                        f"retries ({type(exc).__name__}): {exc or 'no detail'}"
                    ) from exc
                await asyncio.sleep(self.poll_interval_seconds)
                continue

            last_response = response
            status_code = response.headers.get("X-Api-Status-Code", "")

            if status_code == "20000000":
                data = self._json_or_empty(response)
                return self._extract_text(data)
            if status_code in {"20000001", "20000002"}:
                await asyncio.sleep(self.poll_interval_seconds)
                continue

            self._ensure_ok(response=response, phase="query")

        detail = self._response_detail(last_response) if last_response else "no response"
        raise DoubaoASRError(f"Doubao ASR query timed out: {detail}")

    def _headers(self, *, request_id: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": request_id,
            "X-Api-Sequence": "-1",
        }
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        else:
            headers["X-Api-App-Key"] = self.app_id or ""
            headers["X-Api-Access-Key"] = self.access_token or ""
        return headers

    def _validate_credentials(self) -> None:
        if self.api_key:
            return
        if self.app_id and self.access_token:
            return
        raise DoubaoASRError(
            "Doubao ASR credentials are not set. Configure DOUBAO_API_KEY, "
            "or DOUBAO_APP_ID and DOUBAO_ACCESS_TOKEN."
        )

    def _ensure_ok(self, *, response: httpx.Response, phase: str) -> None:
        status_code = response.headers.get("X-Api-Status-Code", "")
        if response.is_error or status_code != "20000000":
            raise DoubaoASRError(
                f"Doubao ASR {phase} error {status_code or response.status_code}: "
                f"{self._response_detail(response)}"
            )

    def _json_or_empty(self, response: httpx.Response) -> dict[str, Any]:
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise DoubaoASRError(
                f"Doubao ASR returned non-JSON response: {response.text[:500]}"
            ) from exc
        return data if isinstance(data, dict) else {}

    def _extract_text(self, data: dict[str, Any]) -> str:
        result = data.get("result")
        if isinstance(result, dict):
            text = result.get("text")
            if text is not None:
                return str(text).strip()

        text = data.get("text")
        if text is not None:
            return str(text).strip()

        raise DoubaoASRError(f"Doubao ASR result did not contain transcript text: {data}")

    def _response_detail(self, response: httpx.Response | None) -> str:
        if response is None:
            return "no response"
        message = response.headers.get("X-Api-Message")
        log_id = response.headers.get("X-Tt-Logid")
        body = response.text[:500]
        parts = [part for part in [message, f"log_id={log_id}" if log_id else None, body] if part]
        return "; ".join(parts) or f"HTTP {response.status_code}"

    def _audio_format(self, *, mime_type: str, filename: str) -> str:
        normalized = mime_type.lower().split(";", 1)[0].strip()
        mime_formats = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/wave": "wav",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/mp4": "m4a",
            "audio/x-m4a": "m4a",
            "audio/aac": "aac",
            "audio/flac": "flac",
            "audio/x-flac": "flac",
            "audio/ogg": "ogg",
        }
        if normalized in mime_formats:
            return mime_formats[normalized]

        suffix = Path(filename).suffix.lower().lstrip(".")
        return suffix or "wav"

    def _effective_language(
        self, *, language: str | None, language_hints: list[str] | None
    ) -> str | None:
        if language_hints:
            return language_hints[0]
        return language

    def _doubao_language(self, language: str | None) -> str | None:
        if not language:
            return None
        normalized = language.replace("_", "-").lower()
        language_map = {
            "zh": "zh-CN",
            "zh-cn": "zh-CN",
            "cn": "zh-CN",
            "en": "en-US",
            "en-us": "en-US",
            "ja": "ja-JP",
            "ja-jp": "ja-JP",
            "ko": "ko-KR",
            "ko-kr": "ko-KR",
        }
        return language_map.get(normalized, language)
