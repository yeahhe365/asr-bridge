from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class DashScopeError(RuntimeError):
    """Raised when DashScope returns an error or no usable transcript."""


@dataclass
class VocabularyWord:
    text: str
    weight: int = 4
    lang: str = "zh"


@dataclass
class DashScopeFunASRClient:
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    http_client: httpx.AsyncClient | None = None
    poll_interval_seconds: float = 1.0
    timeout_seconds: float = 300.0
    default_language_hints: list[str] | None = None
    default_vocabulary_id: str | None = None

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
                language_hints=language_hints,
                vocabulary_id=vocabulary_id,
            )
            result_url = await self._wait_for_result_url(client=client, task_id=task_id)
            if result_url is None:
                return ""
            return await self._download_transcript_text(client=client, result_url=result_url)
        except DashScopeError:
            raise
        except httpx.HTTPError as exc:
            raise DashScopeError(
                f"DashScope network error ({type(exc).__name__}): {exc or 'no detail'}"
            ) from exc
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
        language_hints: list[str] | None,
        vocabulary_id: str | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "input": {
                "file_urls": [self._data_uri(audio_bytes=audio_bytes, mime_type=mime_type)]
            },
            "parameters": {"channel_id": [0]},
        }
        effective_language_hints = self._effective_language_hints(
            language=language,
            language_hints=language_hints,
        )
        if effective_language_hints:
            payload["parameters"]["language_hints"] = effective_language_hints
        effective_vocabulary_id = vocabulary_id or self.default_vocabulary_id
        if effective_vocabulary_id:
            payload["parameters"]["vocabulary_id"] = effective_vocabulary_id

        logger.info(
            "DashScope submit: model=%s vocabulary_id=%s hints=%s",
            model, effective_vocabulary_id or "-", effective_language_hints or "-",
        )

        for attempt in range(3):
            try:
                response = await client.post(
                    f"{self.base_url}/services/audio/asr/transcription",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-DashScope-Async": "enable",
                    },
                    json=payload,
                )
                break
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise DashScopeError(
                        f"DashScope submit failed after 3 attempts "
                        f"({type(exc).__name__}): {exc or 'no detail'}"
                    ) from exc
                logger.warning("DashScope submit attempt %d failed: %s", attempt + 1, exc)
                await asyncio.sleep(1.0 * (attempt + 1))

        data = self._json_or_error(response)
        output = data.get("output") or {}
        task_id = output.get("task_id")
        if not task_id:
            raise DashScopeError(f"DashScope did not return a task_id: {data}")
        return str(task_id)

    async def create_vocabulary(
        self,
        *,
        prefix: str,
        target_model: str,
        vocabulary: list[VocabularyWord],
    ) -> dict[str, object]:
        client = self.http_client or httpx.AsyncClient(timeout=60)
        close_client = self.http_client is None
        try:
            response = await client.post(
                f"{self.base_url}/services/audio/asr/customization",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "speech-biasing",
                    "input": {
                        "action": "create_vocabulary",
                        "target_model": self._dashscope_model(target_model),
                        "prefix": prefix,
                        "vocabulary": [
                            {
                                "text": word.text,
                                "weight": word.weight,
                                "lang": word.lang,
                            }
                            for word in vocabulary
                        ],
                    },
                },
            )
            data = self._json_or_error(response)
            output = data.get("output")
            if not isinstance(output, dict) or not output.get("vocabulary_id"):
                raise DashScopeError(f"DashScope did not return a vocabulary_id: {data}")
            return output
        except DashScopeError:
            raise
        except httpx.HTTPError as exc:
            raise DashScopeError(
                f"DashScope network error ({type(exc).__name__}): {exc or 'no detail'}"
            ) from exc
        finally:
            if close_client:
                await client.aclose()

    async def query_vocabulary(self, *, vocabulary_id: str) -> dict[str, object]:
        """Query a vocabulary's status and details.

        Returns the output dict which includes ``status`` (``"OK"`` or
        ``"UNDEPLOYED"``) and the vocabulary entries.
        """
        return await self._vocabulary_action(
            action="query_vocabulary",
            extra_input={"vocabulary_id": vocabulary_id},
        )

    async def list_vocabularies(
        self,
        *,
        prefix: str | None = None,
        page_index: int = 0,
        page_size: int = 50,
    ) -> list[dict[str, object]]:
        """List vocabularies owned by the current account.

        Supports optional ``prefix`` filter and pagination.
        """
        extra: dict[str, object] = {
            "page_index": page_index,
            "page_size": page_size,
        }
        if prefix:
            extra["prefix"] = prefix
        result = await self._vocabulary_action(
            action="list_vocabulary",
            extra_input=extra,
        )
        vocabularies = result.get("vocabulary_list") or result.get("vocabularies")
        if isinstance(vocabularies, list):
            return vocabularies
        return []

    async def delete_vocabulary(self, *, vocabulary_id: str) -> dict[str, object]:
        """Delete a vocabulary by its ID."""
        return await self._vocabulary_action(
            action="delete_vocabulary",
            extra_input={"vocabulary_id": vocabulary_id},
        )

    async def _vocabulary_action(
        self,
        *,
        action: str,
        extra_input: dict[str, object] | None = None,
    ) -> dict[str, object]:
        client = self.http_client or httpx.AsyncClient(timeout=60)
        close_client = self.http_client is None
        input_payload: dict[str, object] = {"action": action}
        if extra_input:
            input_payload.update(extra_input)
        try:
            response = await client.post(
                f"{self.base_url}/services/audio/asr/customization",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "speech-biasing",
                    "input": input_payload,
                },
            )
            data = self._json_or_error(response)
            output = data.get("output")
            if not isinstance(output, dict):
                raise DashScopeError(
                    f"DashScope vocabulary {action} returned unexpected response: {data}"
                )
            return output
        except DashScopeError:
            raise
        except httpx.HTTPError as exc:
            raise DashScopeError(
                f"DashScope network error ({type(exc).__name__}): {exc or 'no detail'}"
            ) from exc
        finally:
            if close_client:
                await client.aclose()

    async def _wait_for_result_url(
        self, *, client: httpx.AsyncClient, task_id: str
    ) -> str | None:
        deadline = time.monotonic() + self.timeout_seconds
        last_payload: dict[str, Any] | None = None
        consecutive_network_errors = 0

        while time.monotonic() < deadline:
            try:
                response = await client.get(
                    f"{self.base_url}/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                consecutive_network_errors = 0
            except httpx.HTTPError as exc:
                consecutive_network_errors += 1
                logger.warning(
                    "DashScope poll network error (#%d): %s: %s",
                    consecutive_network_errors, type(exc).__name__, exc or "no detail",
                )
                if consecutive_network_errors >= 5:
                    raise DashScopeError(
                        f"DashScope network error after {consecutive_network_errors} "
                        f"retries ({type(exc).__name__}): {exc or 'no detail'}"
                    ) from exc
                await asyncio.sleep(self.poll_interval_seconds)
                continue

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
        for attempt in range(3):
            try:
                response = await client.get(result_url)
                break
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise DashScopeError(
                        f"DashScope download failed after 3 attempts "
                        f"({type(exc).__name__}): {exc or 'no detail'}"
                    ) from exc
                logger.warning("DashScope download attempt %d failed: %s", attempt + 1, exc)
                await asyncio.sleep(1.0 * (attempt + 1))

        try:
            data = response.json()
        except ValueError as exc:
            raise DashScopeError(
                f"DashScope transcript URL returned non-JSON {response.status_code}: "
                f"{response.text[:500]}"
            ) from exc
        if not isinstance(data, dict):
            raise DashScopeError(f"DashScope transcript result is not a JSON object: {data}")
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
        # Strip any existing "data:" prefix to normalize
        raw_mime = mime_type
        if raw_mime.startswith("data:"):
            raw_mime = raw_mime[len("data:"):]
        # Strip any existing ";base64,..." suffix to get clean mime type
        if ";base64" in raw_mime:
            raw_mime = raw_mime.split(";base64")[0]
        return f"data:{raw_mime};base64,{encoded}"

    def _dashscope_model(self, requested_model: str) -> str:
        if requested_model in {
            "fun-asr",
            "fun-asr-mtl",
            "fun-asr-mtl-2025-08-25",
        }:
            return requested_model
        return "fun-asr"

    def _effective_language_hints(
        self, *, language: str | None, language_hints: list[str] | None
    ) -> list[str] | None:
        if language_hints:
            return language_hints
        if language:
            return [language]
        return self.default_language_hints
