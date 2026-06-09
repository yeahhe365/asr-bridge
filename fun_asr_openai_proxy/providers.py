from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .dashscope import DashScopeError, VocabularyWord
from .doubao import DoubaoASRError


DOUBAO_MODEL_NAMES = {
    "doubao-asr",
    "doubao-seed-asr",
}


class Transcriber(Protocol):
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
    ) -> str: ...

    async def create_vocabulary(
        self,
        *,
        prefix: str,
        target_model: str,
        vocabulary: list[VocabularyWord],
    ) -> dict[str, object]: ...


@dataclass
class ModelRoutingTranscriber:
    dashscope: Transcriber | None
    doubao: Transcriber | None = None

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
        target = self._transcriber_for_model(model)
        return await target.transcribe(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            filename=filename,
            model=model,
            language=language,
            language_hints=language_hints,
            vocabulary_id=vocabulary_id,
        )

    async def create_vocabulary(
        self,
        *,
        prefix: str,
        target_model: str,
        vocabulary: list[VocabularyWord],
    ) -> dict[str, object]:
        if self.dashscope is None:
            raise DashScopeError(
                "DASHSCOPE_API_KEY must be set to create DashScope vocabularies."
            )
        return await self.dashscope.create_vocabulary(
            prefix=prefix,
            target_model=target_model,
            vocabulary=vocabulary,
        )

    def _transcriber_for_model(self, model: str) -> Transcriber:
        if model in DOUBAO_MODEL_NAMES:
            if self.doubao is None:
                raise DoubaoASRError(
                    "DOUBAO_API_KEY, or DOUBAO_APP_ID and DOUBAO_ACCESS_TOKEN, "
                    "must be set to use Doubao ASR models."
                )
            return self.doubao
        if self.dashscope is None:
            raise DashScopeError(
                "DASHSCOPE_API_KEY must be set to use DashScope Fun-ASR models."
            )
        return self.dashscope
