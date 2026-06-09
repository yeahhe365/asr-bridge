from __future__ import annotations

import pytest

from fun_asr_openai_proxy.dashscope import VocabularyWord
from fun_asr_openai_proxy.doubao import DoubaoASRError
from fun_asr_openai_proxy.providers import ModelRoutingTranscriber


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text
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
        return self.text

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
        return {"vocabulary_id": "vocab-test", "status": "PENDING"}


@pytest.mark.anyio("asyncio")
async def test_router_sends_doubao_models_to_doubao_transcriber() -> None:
    dashscope = FakeTranscriber("dashscope")
    doubao = FakeTranscriber("doubao")
    router = ModelRoutingTranscriber(dashscope=dashscope, doubao=doubao)

    text = await router.transcribe(
        audio_bytes=b"RIFFdemo-audio",
        mime_type="audio/wav",
        filename="speech.wav",
        model="doubao-asr",
        language=None,
        language_hints=None,
        vocabulary_id=None,
    )

    assert text == "doubao"
    assert len(doubao.calls) == 1
    assert dashscope.calls == []


@pytest.mark.anyio("asyncio")
async def test_router_sends_fun_asr_and_whisper_alias_to_dashscope() -> None:
    dashscope = FakeTranscriber("dashscope")
    doubao = FakeTranscriber("doubao")
    router = ModelRoutingTranscriber(dashscope=dashscope, doubao=doubao)

    for model in ["fun-asr", "whisper-1"]:
        text = await router.transcribe(
            audio_bytes=b"RIFFdemo-audio",
            mime_type="audio/wav",
            filename="speech.wav",
            model=model,
            language=None,
            language_hints=None,
            vocabulary_id=None,
        )
        assert text == "dashscope"

    assert [call["model"] for call in dashscope.calls] == ["fun-asr", "whisper-1"]
    assert doubao.calls == []


@pytest.mark.anyio("asyncio")
async def test_router_requires_doubao_transcriber_for_doubao_models() -> None:
    router = ModelRoutingTranscriber(
        dashscope=FakeTranscriber("dashscope"),
        doubao=None,
    )

    with pytest.raises(DoubaoASRError, match="DOUBAO"):
        await router.transcribe(
            audio_bytes=b"RIFFdemo-audio",
            mime_type="audio/wav",
            filename="speech.wav",
            model="doubao-asr",
            language=None,
            language_hints=None,
            vocabulary_id=None,
        )


@pytest.mark.anyio("asyncio")
async def test_router_delegates_vocabulary_creation_to_dashscope() -> None:
    dashscope = FakeTranscriber("dashscope")
    router = ModelRoutingTranscriber(dashscope=dashscope, doubao=FakeTranscriber("doubao"))

    result = await router.create_vocabulary(
        prefix="linuxdo",
        target_model="fun-asr",
        vocabulary=[VocabularyWord(text="佬友", weight=4, lang="zh")],
    )

    assert result == {"vocabulary_id": "vocab-test", "status": "PENDING"}
    assert dashscope.calls == [
        {
            "prefix": "linuxdo",
            "target_model": "fun-asr",
            "vocabulary": [VocabularyWord(text="佬友", weight=4, lang="zh")],
        }
    ]
