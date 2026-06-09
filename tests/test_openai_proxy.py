from __future__ import annotations

import io

from fastapi.testclient import TestClient

from fun_asr_openai_proxy.app import create_app
from fun_asr_openai_proxy.dashscope import VocabularyWord


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
        language_hints: list[str] | None,
        vocabulary_id: str | None,
    ) -> str:
        self.calls.append(
            {
                "audio_bytes": audio_bytes,
                "mime_type": mime_type,
                "filename": filename,
                "model": model,
                "language": language,
                "language_hints": language_hints,
                "vocabulary_id": vocabulary_id,
            }
        )
        return "欢迎使用阿里云语音识别。"

    async def create_vocabulary(
        self,
        *,
        prefix: str,
        target_model: str,
        vocabulary: list[VocabularyWord],
    ) -> dict[str, object]:
        self.calls.append(
            {
                "prefix": prefix,
                "target_model": target_model,
                "vocabulary": vocabulary,
            }
        )
        return {"vocabulary_id": "vocab-stock-123", "status": "PENDING"}


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
            },
            {
                "id": "doubao-asr",
                "object": "model",
                "created": 0,
                "owned_by": "volcengine-doubao",
            },
        ],
    }


def test_models_endpoint_has_non_v1_alias() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "fun-asr"


def test_admin_ui_is_served_at_root() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ASR Bridge" in response.text
    assert "Fun-ASR OpenAI Proxy" not in response.text
    assert "/static/admin.js" in response.text


def test_admin_ui_uses_selects_for_language_hints() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert '<select id="language-hints"' in response.text
    assert '<select id="transcription-language-hints"' in response.text
    assert '<input\n              id="language-hints"' not in response.text
    assert '<input id="transcription-language-hints"' not in response.text


def test_admin_ui_uses_beginner_friendly_settings_copy() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "默认设置" in response.text
    assert "识别语言" in response.text
    assert "自动判断" in response.text
    assert "热词表" in response.text
    assert "保存默认设置" in response.text
    assert "运行配置" not in response.text
    assert "语言提示" not in response.text
    assert "热词表 ID" not in response.text
    assert "保存运行配置" not in response.text


def test_admin_ui_links_funasr_favicon() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert '<link rel="icon" type="image/png" href="/static/favicon.png" />' in response.text


def test_admin_ui_includes_recording_controls() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="start-recording"' in response.text
    assert 'id="stop-recording"' in response.text
    assert 'id="transcribe-recording"' in response.text
    assert 'id="recorded-audio"' in response.text


def test_admin_ui_includes_doubao_transcription_models() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert '<option value="doubao-asr">doubao-asr</option>' in response.text
    assert '<option value="volc.seedasr.auc">volc.seedasr.auc</option>' not in response.text


def test_admin_ui_hides_whisper_compatibility_alias() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert '<option value="whisper-1">whisper-1</option>' not in response.text


def test_admin_ui_hides_dated_fun_asr_models() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "fun-asr-2025-11-07" not in response.text
    assert "fun-asr-2025-08-25" not in response.text


def test_admin_ui_includes_realtime_logs_panel() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="logs"' in response.text
    assert 'id="log-stream-status"' in response.text
    assert 'id="clear-logs"' in response.text


def test_favicon_is_served_from_root() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/favicon.ico")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG")


def test_static_admin_javascript_is_served() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/static/admin.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert "loadSettings" in response.text


def test_static_admin_javascript_prefills_laoyou_hotword_only() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/static/admin.js")

    assert response.status_code == 200
    assert 'addWordRow({ text: "佬友", weight: 4, lang: "zh" });' in response.text
    assert 'addWordRow({ text: "英伟达", weight: 4, lang: "zh" });' not in response.text
    assert 'addWordRow({ text: "纳斯达克", weight: 4, lang: "zh" });' not in response.text


def test_static_admin_javascript_supports_browser_recording() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/static/admin.js")

    assert response.status_code == 200
    assert "AudioContext" in response.text
    assert "audio/wav" in response.text
    assert "encodeWavBlob" in response.text
    assert "transcribeRecording" in response.text


def test_static_admin_javascript_connects_log_stream() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.get("/static/admin.js")

    assert response.status_code == 200
    assert "EventSource" in response.text
    assert "/api/logs/stream" in response.text
    assert "loadLogs" in response.text


def test_settings_endpoint_reads_runtime_configuration() -> None:
    app = create_app(
        transcriber=FakeTranscriber(),
        initial_language_hints=["zh"],
        initial_vocabulary_id="vocab-stock-123",
    )
    client = TestClient(app)

    response = client.get("/api/settings")

    assert response.status_code == 200
    assert response.json() == {
        "language_hints": ["zh"],
        "vocabulary_id": "vocab-stock-123",
    }


def test_settings_endpoint_updates_runtime_configuration() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.put(
        "/api/settings",
        json={"language_hints": ["zh", "en"], "vocabulary_id": "vocab-stock-456"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "language_hints": ["zh", "en"],
        "vocabulary_id": "vocab-stock-456",
    }
    assert client.get("/api/settings").json() == {
        "language_hints": ["zh", "en"],
        "vocabulary_id": "vocab-stock-456",
    }
    logs = client.get("/api/logs").json()["data"]
    assert any(event["message"] == "运行配置已更新" for event in logs)


def test_logs_endpoint_records_transcription_events() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "fun-asr"},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    logs = client.get("/api/logs").json()["data"]
    messages = [event["message"] for event in logs]
    assert "转写开始" in messages
    assert "转写成功" in messages
    success_event = next(event for event in logs if event["message"] == "转写成功")
    assert success_event["level"] == "info"
    assert success_event["details"]["model"] == "fun-asr"


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
            "language_hints": None,
            "vocabulary_id": None,
        }
    ]


def test_audio_transcriptions_accepts_dashscope_language_hints() -> None:
    fake = FakeTranscriber()
    app = create_app(transcriber=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "fun-asr", "language_hints": "zh,en"},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    assert fake.calls[0]["language_hints"] == ["zh", "en"]


def test_audio_transcriptions_accepts_dashscope_language_hints_json_array() -> None:
    fake = FakeTranscriber()
    app = create_app(transcriber=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "fun-asr", "language_hints": '["zh"]'},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    assert fake.calls[0]["language_hints"] == ["zh"]


def test_audio_transcriptions_accepts_vocabulary_id() -> None:
    fake = FakeTranscriber()
    app = create_app(transcriber=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "fun-asr", "vocabulary_id": "vocab-stock-123"},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    assert fake.calls[0]["vocabulary_id"] == "vocab-stock-123"


def test_audio_transcriptions_uses_runtime_settings_as_request_defaults() -> None:
    fake = FakeTranscriber()
    app = create_app(
        transcriber=fake,
        initial_language_hints=["zh"],
        initial_vocabulary_id="vocab-stock-123",
    )
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "fun-asr"},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    assert fake.calls[0]["language_hints"] == ["zh"]
    assert fake.calls[0]["vocabulary_id"] == "vocab-stock-123"


def test_audio_transcriptions_prefers_request_values_over_runtime_settings() -> None:
    fake = FakeTranscriber()
    app = create_app(
        transcriber=fake,
        initial_language_hints=["zh"],
        initial_vocabulary_id="vocab-stock-123",
    )
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "fun-asr",
            "language_hints": "en",
            "vocabulary_id": "vocab-request",
        },
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    assert fake.calls[0]["language_hints"] == ["en"]
    assert fake.calls[0]["vocabulary_id"] == "vocab-request"


def test_default_transcriber_routes_doubao_models_from_environment(monkeypatch) -> None:
    created_clients: list[dict[str, object]] = []

    class FakeDoubaoClient:
        def __init__(self, **kwargs) -> None:
            created_clients.append(kwargs)

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
            assert audio_bytes == b"RIFFdemo-audio"
            assert mime_type == "audio/wav"
            assert filename == "speech.wav"
            assert model == "doubao-asr"
            assert language_hints is None
            assert vocabulary_id is None
            return "豆包识别结果。"

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DOUBAO_APP_ID", "test-app-id")
    monkeypatch.setenv("DOUBAO_ACCESS_TOKEN", "test-access-token")
    monkeypatch.setenv("DOUBAO_RESOURCE_ID", "volc.seedasr.auc")
    monkeypatch.setattr(
        "fun_asr_openai_proxy.app.DoubaoASRClient",
        FakeDoubaoClient,
    )

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "doubao-asr"},
        files={"file": ("speech.wav", io.BytesIO(b"RIFFdemo-audio"), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "豆包识别结果。"}
    assert created_clients == [
        {
            "api_key": None,
            "app_id": "test-app-id",
            "access_token": "test-access-token",
            "resource_id": "volc.seedasr.auc",
        }
    ]


def test_create_vocabulary_accepts_json_payload() -> None:
    fake = FakeTranscriber()
    app = create_app(transcriber=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/vocabularies",
        json={
            "prefix": "stock",
            "target_model": "fun-asr",
            "vocabulary": [
                {"text": "英伟达", "weight": 4, "lang": "zh"},
                {"text": "纳斯达克", "weight": 4, "lang": "zh"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"vocabulary_id": "vocab-stock-123", "status": "PENDING"}
    assert fake.calls[0] == {
        "prefix": "stock",
        "target_model": "fun-asr",
        "vocabulary": [
            VocabularyWord(text="英伟达", weight=4, lang="zh"),
            VocabularyWord(text="纳斯达克", weight=4, lang="zh"),
        ],
    }


def test_create_vocabulary_rejects_empty_vocabulary() -> None:
    app = create_app(transcriber=FakeTranscriber())
    client = TestClient(app)

    response = client.post(
        "/v1/audio/vocabularies",
        json={"prefix": "stock", "vocabulary": []},
    )

    assert response.status_code == 422


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
