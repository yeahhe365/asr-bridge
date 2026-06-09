from __future__ import annotations

import io

from fastapi.testclient import TestClient

from fun_asr_openai_proxy.app import create_app


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def transcribe(
        self,
        *,
        audio_bytes: bytes,
        mime_type: str,
        filename: str,
        model: str,
        language: str | None,
    ) -> str:
        self.calls.append(
            {
                "audio_bytes": audio_bytes,
                "mime_type": mime_type,
                "filename": filename,
                "model": model,
                "language": language,
            }
        )
        return "欢迎使用阿里云语音识别。"


def test_models_endpoint_lists_fun_asr() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
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


def test_models_endpoint_has_non_v1_alias() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "fun-asr"


def test_audio_transcriptions_accepts_openai_multipart_form() -> None:
    fake = FakeTranscriber()
    app = create_app(transcriber=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "fun-asr", "language": "zh"},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "欢迎使用阿里云语音识别。"}
    assert fake.calls == [
        {
            "audio_bytes": b"RIFFdemo-audio",
            "mime_type": "audio/wav",
            "filename": "speech.wav",
            "model": "fun-asr",
            "language": "zh",
        }
    ]


def test_audio_transcriptions_has_non_v1_alias() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.post(
        "/audio/transcriptions",
        data={"model": "fun-asr"},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "欢迎使用阿里云语音识别。"}


def test_audio_transcriptions_can_return_plain_text() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "fun-asr", "response_format": "text"},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "欢迎使用阿里云语音识别。"


def test_audio_transcriptions_rejects_missing_file() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "fun-asr"},
    )

    assert response.status_code == 422
