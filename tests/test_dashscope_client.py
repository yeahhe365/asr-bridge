from __future__ import annotations

import base64
import json

import httpx
import pytest

from fun_asr_openai_proxy.dashscope import DashScopeFunASRClient, VocabularyWord


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
        language_hints=None,
        vocabulary_id=None,
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
        language_hints=None,
        vocabulary_id=None,
    )

    assert submitted_models == ["fun-asr"]


@pytest.mark.anyio("asyncio")
async def test_transcribe_maps_dated_fun_asr_model_names_to_fun_asr() -> None:
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

    for model in ["fun-asr-2025-11-07", "fun-asr-2025-08-25"]:
        await client.transcribe(
            audio_bytes=b"RIFFdemo-audio",
            mime_type="audio/wav",
            filename="speech.wav",
            model=model,
            language=None,
            language_hints=None,
            vocabulary_id=None,
        )

    assert submitted_models == ["fun-asr", "fun-asr"]


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
        language_hints=None,
        vocabulary_id=None,
    )

    assert text == ""


@pytest.mark.anyio("asyncio")
async def test_transcribe_prefers_language_hints_over_openai_language() -> None:
    submitted_language_hints: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/services/audio/asr/transcription":
            payload = json.loads(request.content.decode("utf-8"))
            submitted_language_hints.append(payload["parameters"]["language_hints"])
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
        model="fun-asr",
        language="en",
        language_hints=["zh", "yue"],
        vocabulary_id=None,
    )

    assert submitted_language_hints == [["zh", "yue"]]


@pytest.mark.anyio("asyncio")
async def test_transcribe_uses_default_language_hints_when_request_has_none() -> None:
    submitted_language_hints: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/services/audio/asr/transcription":
            payload = json.loads(request.content.decode("utf-8"))
            submitted_language_hints.append(payload["parameters"]["language_hints"])
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
        default_language_hints=["zh"],
        http_client=http_client,
        poll_interval_seconds=0,
    )

    await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="fun-asr",
        language=None,
        language_hints=None,
        vocabulary_id=None,
    )

    assert submitted_language_hints == [["zh"]]


@pytest.mark.anyio("asyncio")
async def test_transcribe_sends_request_vocabulary_id() -> None:
    submitted_vocabulary_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/services/audio/asr/transcription":
            payload = json.loads(request.content.decode("utf-8"))
            submitted_vocabulary_ids.append(payload["parameters"]["vocabulary_id"])
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
        default_vocabulary_id="vocab-default",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="fun-asr",
        language=None,
        language_hints=None,
        vocabulary_id="vocab-request",
    )

    assert submitted_vocabulary_ids == ["vocab-request"]


@pytest.mark.anyio("asyncio")
async def test_transcribe_uses_default_vocabulary_id_when_request_has_none() -> None:
    submitted_vocabulary_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/services/audio/asr/transcription":
            payload = json.loads(request.content.decode("utf-8"))
            submitted_vocabulary_ids.append(payload["parameters"]["vocabulary_id"])
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
        default_vocabulary_id="vocab-default",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="fun-asr",
        language=None,
        language_hints=None,
        vocabulary_id=None,
    )

    assert submitted_vocabulary_ids == ["vocab-default"]


@pytest.mark.anyio("asyncio")
async def test_create_vocabulary_posts_customization_payload() -> None:
    submitted_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/services/audio/asr/customization":
            payload = json.loads(request.content.decode("utf-8"))
            submitted_payloads.append(payload)
            assert request.headers["Authorization"] == "Bearer test-key"
            return httpx.Response(
                200,
                json={
                    "output": {
                        "vocabulary_id": "vocab-stock-123",
                        "status": "PENDING",
                    }
                },
            )

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DashScopeFunASRClient(api_key="test-key", http_client=http_client)

    result = await client.create_vocabulary(
        prefix="stock",
        target_model="fun-asr",
        vocabulary=[
            VocabularyWord(text="英伟达", weight=4, lang="zh"),
            VocabularyWord(text="纳斯达克", weight=4, lang="zh"),
        ],
    )

    assert result == {
        "vocabulary_id": "vocab-stock-123",
        "status": "PENDING",
    }
    assert submitted_payloads == [
        {
            "model": "speech-biasing",
            "input": {
                "action": "create_vocabulary",
                "target_model": "fun-asr",
                "prefix": "stock",
                "vocabulary": [
                    {"text": "英伟达", "weight": 4, "lang": "zh"},
                    {"text": "纳斯达克", "weight": 4, "lang": "zh"},
                ],
            },
        }
    ]
