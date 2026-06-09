from __future__ import annotations

import base64
import json

import httpx
import pytest

from fun_asr_openai_proxy.doubao import DoubaoASRClient, DoubaoASRError


@pytest.mark.anyio("asyncio")
async def test_transcribe_submits_base64_audio_and_queries_result() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        if request.url.path == "/api/v3/auc/bigmodel/submit":
            payload = json.loads(request.content.decode("utf-8"))
            assert request.headers["X-Api-App-Key"] == "test-app-id"
            assert request.headers["X-Api-Access-Key"] == "test-access-token"
            assert request.headers["X-Api-Resource-Id"] == "volc.seedasr.auc"
            assert request.headers["X-Api-Request-Id"]
            assert request.headers["X-Api-Sequence"] == "-1"
            assert payload["user"] == {"uid": "test-app-id"}
            assert payload["audio"]["format"] == "wav"
            assert base64.b64decode(payload["audio"]["data"]) == b"RIFFdemo-audio"
            assert payload["request"] == {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
            }
            return httpx.Response(
                200,
                headers={
                    "X-Api-Status-Code": "20000000",
                    "X-Api-Message": "OK",
                    "X-Tt-Logid": "log-submit",
                },
            )

        if request.url.path == "/api/v3/auc/bigmodel/query":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload == {}
            assert request.headers["X-Api-Request-Id"] == requests[0].headers[
                "X-Api-Request-Id"
            ]
            return httpx.Response(
                200,
                headers={
                    "X-Api-Status-Code": "20000000",
                    "X-Api-Message": "OK",
                    "X-Tt-Logid": "log-query",
                },
                json={"result": {"text": "你好，佬友。"}},
            )

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DoubaoASRClient(
        app_id="test-app-id",
        access_token="test-access-token",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    text = await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="doubao-asr",
        language=None,
        language_hints=None,
        vocabulary_id=None,
    )

    assert text == "你好，佬友。"
    assert [request.url.path for request in requests] == [
        "/api/v3/auc/bigmodel/submit",
        "/api/v3/auc/bigmodel/query",
    ]


@pytest.mark.anyio("asyncio")
async def test_transcribe_polls_until_doubao_task_completes() -> None:
    query_statuses = ["20000001", "20000002", "20000000"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/auc/bigmodel/submit":
            return httpx.Response(200, headers={"X-Api-Status-Code": "20000000"})

        if request.url.path == "/api/v3/auc/bigmodel/query":
            status = query_statuses.pop(0)
            return httpx.Response(
                200,
                headers={"X-Api-Status-Code": status, "X-Api-Message": "OK"},
                json={"result": {"text": "轮询完成。"}} if status == "20000000" else {},
            )

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DoubaoASRClient(
        app_id="test-app-id",
        access_token="test-access-token",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    text = await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="doubao-asr",
        language=None,
        language_hints=None,
        vocabulary_id=None,
    )

    assert text == "轮询完成。"
    assert query_statuses == []


@pytest.mark.anyio("asyncio")
async def test_transcribe_maps_openai_language_to_doubao_audio_language() -> None:
    submitted_audio_languages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/auc/bigmodel/submit":
            payload = json.loads(request.content.decode("utf-8"))
            submitted_audio_languages.append(payload["audio"]["language"])
            return httpx.Response(200, headers={"X-Api-Status-Code": "20000000"})

        if request.url.path == "/api/v3/auc/bigmodel/query":
            return httpx.Response(
                200,
                headers={"X-Api-Status-Code": "20000000"},
                json={"result": {"text": "ok"}},
            )

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DoubaoASRClient(
        app_id="test-app-id",
        access_token="test-access-token",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="doubao-asr",
        language="zh",
        language_hints=None,
        vocabulary_id=None,
    )

    assert submitted_audio_languages == ["zh-CN"]


@pytest.mark.anyio("asyncio")
async def test_transcribe_uses_new_console_api_key_header_when_configured() -> None:
    auth_headers: list[dict[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/auc/bigmodel/submit":
            auth_headers.append(
                {
                    "api_key": request.headers.get("X-Api-Key"),
                    "app_key": request.headers.get("X-Api-App-Key"),
                    "access_key": request.headers.get("X-Api-Access-Key"),
                }
            )
            return httpx.Response(200, headers={"X-Api-Status-Code": "20000000"})

        if request.url.path == "/api/v3/auc/bigmodel/query":
            return httpx.Response(
                200,
                headers={"X-Api-Status-Code": "20000000"},
                json={"result": {"text": "ok"}},
            )

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DoubaoASRClient(
        api_key="test-api-key",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    await client.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="doubao-asr",
        language=None,
        language_hints=None,
        vocabulary_id=None,
    )

    assert auth_headers == [
        {"api_key": "test-api-key", "app_key": None, "access_key": None}
    ]


@pytest.mark.anyio("asyncio")
async def test_transcribe_raises_on_failed_doubao_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/auc/bigmodel/submit":
            return httpx.Response(
                200,
                headers={
                    "X-Api-Status-Code": "40000001",
                    "X-Api-Message": "auth failed",
                    "X-Tt-Logid": "log-failed",
                },
                text="bad credentials",
            )

        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DoubaoASRClient(
        app_id="test-app-id",
        access_token="test-access-token",
        http_client=http_client,
        poll_interval_seconds=0,
    )

    with pytest.raises(DoubaoASRError, match="40000001"):
        await client.transcribe(
            audio_bytes=b"RIFFdemo-audio",
            mime_type="audio/wav",
            filename="speech.wav",
            model="doubao-asr",
            language=None,
            language_hints=None,
            vocabulary_id=None,
        )
