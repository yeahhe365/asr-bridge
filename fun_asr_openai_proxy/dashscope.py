from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any

import httpx


class DashScopeError(RuntimeError):
    """Raised when DashScope returns an error or no usable transcript."""


@dataclass
class DashScopeFunASRClient:
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
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
    ) -> str:
        del filename
        client = self.http_client or httpx.AsyncClient(timeout=60)
        close_client = self.http_client is None
        try:
            task_id = await self._submit_task(
                client=client,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                model=self._dashscope_model(model),
                language=language,
            )
            result_url = await self._wait_for_result_url(client=client, task_id=task_id)
            if result_url is None:
                return ""
            return await self._download_transcript_text(client=client, result_url=result_url)
        finally:
            if close_client:
                await client.aclose()

    async def _submit_task(
        self,
        *,
        client: httpx.AsyncClient,
        audio_bytes: bytes,
        mime_type: str,
        model: str,
        language: str | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "input": {
                "file_urls": [self._data_uri(audio_bytes=audio_bytes, mime_type=mime_type)]
            },
            "parameters": {"channel_id": [0]},
        }
        if language:
            payload["parameters"]["language_hints"] = [language]

        response = await client.post(
            f"{self.base_url}/services/audio/asr/transcription",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            json=payload,
        )
        data = self._json_or_error(response)
        output = data.get("output") or {}
        task_id = output.get("task_id")
        if not task_id:
            raise DashScopeError(f"DashScope did not return a task_id: {data}")
        return str(task_id)

    async def _wait_for_result_url(
        self, *, client: httpx.AsyncClient, task_id: str
    ) -> str | None:
        deadline = time.monotonic() + self.timeout_seconds
        last_payload: dict[str, Any] | None = None

        while time.monotonic() < deadline:
            response = await client.get(
                f"{self.base_url}/tasks/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            data = self._json_or_error(response)
            last_payload = data
            output = data.get("output") or {}
            status = output.get("task_status")

            if status == "SUCCEEDED":
                return self._extract_transcription_url(output)
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                if self._is_no_words_response(data):
                    return None
                raise DashScopeError(f"DashScope task {task_id} ended with {status}: {data}")

            await asyncio.sleep(self.poll_interval_seconds)

        raise DashScopeError(f"DashScope task {task_id} timed out: {last_payload}")

    async def _download_transcript_text(
        self, *, client: httpx.AsyncClient, result_url: str
    ) -> str:
        response = await client.get(result_url)
        data = self._json_or_error(response)
        transcripts = data.get("transcripts") or []
        texts = [
            str(transcript.get("text", "")).strip()
            for transcript in transcripts
            if transcript.get("text")
        ]
        text = "\n".join(texts).strip()
        if not text:
            raise DashScopeError(f"DashScope result did not contain transcript text: {data}")
        return text

    def _json_or_error(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise DashScopeError(
                f"DashScope returned non-JSON response {response.status_code}: "
                f"{response.text[:500]}"
            ) from exc

        if response.is_error or data.get("code"):
            message = data.get("message") or response.text[:500]
            code = data.get("code") or response.status_code
            raise DashScopeError(f"DashScope error {code}: {message}")
        return data

    def _extract_transcription_url(self, output: dict[str, Any]) -> str:
        results = output.get("results") or []
        for result in results:
            result_status = result.get("subtask_status") or result.get("status")
            if result_status and result_status != "SUCCEEDED":
                raise DashScopeError(f"DashScope subtask failed: {result}")
            url = result.get("transcription_url")
            if url:
                return str(url)
        raise DashScopeError(f"DashScope succeeded without transcription_url: {output}")

    def _is_no_words_response(self, data: dict[str, Any]) -> bool:
        output = data.get("output") or {}
        if output.get("code") == "ASR_RESPONSE_HAVE_NO_WORDS":
            return True
        if output.get("message") == "ASR_RESPONSE_HAVE_NO_WORDS":
            return True

        for result in output.get("results") or []:
            result_output = result.get("output") or {}
            if result_output.get("code") == "ASR_RESPONSE_HAVE_NO_WORDS":
                return True
            if result_output.get("message") == "ASR_RESPONSE_HAVE_NO_WORDS":
                return True
        return False

    def _data_uri(self, *, audio_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        return f"{mime_type};base64,{encoded}" if mime_type.startswith("data:") else (
            f"data:{mime_type};base64,{encoded}"
        )

    def _dashscope_model(self, requested_model: str) -> str:
        if requested_model in {"fun-asr", "fun-asr-2025-08-25"}:
            return requested_model
        return "fun-asr"
