from __future__ import annotations

import base64
import json

import httpx
import pytest

from fun_asr_openai_proxy.dashscope import DashScopeFunASRClient


@pytest.mark.anyio("asyncio")
async def test_transcribe_sends_data_uri_and_returns_transcript_text() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        if request.url.path == "/api/v1/services/audio/asr/transcription":
            payload = json.loads(request.content.decode("utf-8"))
            audio_url = payload["input"]["file_urls"][0]
            assert payload["model"] == "fun-asr"
            assert payload["parameters"]["language_hints"] == ["zh"]
            assert audio_url.startswith("data:audio/wav;base64,")
            assert base64.b64decode(audio_url.split(",", 1)[1]) == b"RIFFdemo-audio"
            assert request.headers["X-DashScope-Async"] == "enable"
            return httpx.Response(
                200,
                json={"output": {"task_id": "task-123", "task_status": "PENDING"}},
            )

        if request.url.path == "/api/v1/tasks/task-123":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_status": "SUCCEEDED",
                        "results": [
                            {
                                "subtask_status": "SUCCEEDED",
                                "transcription_url": "https://example.test/result.json",
                            }
                        ],
                    }
                },
            )

        if str(request.url) == "https://example.test/result.json":
            return httpx.Response(
                200,
                json={
                    "transcripts": [
                        {
                            "channel_id": 0,
                            "text": "欢迎使用阿里云语音识别。",
                            "sentences": [],
                        }
                    ]
                },
            )

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DashScopeFunASRClient(
        api_key="test-key",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    text = await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="fun-asr",
        language="zh",
    )

    assert text == "欢迎使用阿里云语音识别。"
    assert [request.url.path for request in requests[:2]] == [
        "/api/v1/services/audio/asr/transcription",
        "/api/v1/tasks/task-123",
    ]


@pytest.mark.anyio("asyncio")
async def test_transcribe_maps_whisper_style_model_names_to_fun_asr() -> None:
    submitted_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/services/audio/asr/transcription":
            payload = json.loads(request.content.decode("utf-8"))
            submitted_models.append(payload["model"])
            return httpx.Response(
                200,
                json={"output": {"task_id": "task-123", "task_status": "PENDING"}},
            )

        if request.url.path == "/api/v1/tasks/task-123":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_status": "SUCCEEDED",
                        "results": [
                            {
                                "subtask_status": "SUCCEEDED",
                                "transcription_url": "https://example.test/result.json",
                            }
                        ],
                    }
                },
            )

        if str(request.url) == "https://example.test/result.json":
            return httpx.Response(200, json={"transcripts": [{"text": "ok"}]})

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DashScopeFunASRClient(
        api_key="test-key",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="whisper-1",
        language=None,
    )

    assert submitted_models == ["fun-asr"]


@pytest.mark.anyio("asyncio")
async def test_transcribe_returns_empty_text_when_dashscope_detects_no_words() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/services/audio/asr/transcription":
            return httpx.Response(
                200,
                json={"output": {"task_id": "task-123", "task_status": "PENDING"}},
            )

        if request.url.path == "/api/v1/tasks/task-123":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-123",
                    "output": {
                        "task_id": "task-123",
                        "task_status": "FAILED",
                        "code": "ASR_RESPONSE_HAVE_NO_WORDS",
                        "message": "ASR_RESPONSE_HAVE_NO_WORDS",
                        "results": [],
                    },
                },
            )

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DashScopeFunASRClient(
        api_key="test-key",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    text = await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="fun-asr",
        language=None,
    )

    assert text == ""
