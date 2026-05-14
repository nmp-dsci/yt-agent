from __future__ import annotations

import json
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.context import RawTranscriptContextProvider, TranscriptContext, TranscriptContextProvider
from src.agents.models import (
    QuestionRequest,
    SummaryRequest,
    TranscriptAnswer,
    TranscriptSummary,
)
from src.agents.prompts import (
    SYSTEM_PROMPT,
    build_question_prompt,
    build_summary_prompt,
    build_transcript_context_prompt,
)
from src.config import Settings
from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.storage import ChromaTranscriptStore


class TranscriptTooLongError(RuntimeError):
    pass


class ChatModel(Protocol):
    def invoke(self, messages: list[SystemMessage | HumanMessage]) -> object:
        ...


class TranscriptAgent:
    def __init__(
        self,
        llm: ChatModel,
        context_provider: TranscriptContextProvider,
        max_transcript_chars: int = 120_000,
    ) -> None:
        self.llm = llm
        self.context_provider = context_provider
        self.max_transcript_chars = max_transcript_chars
        self.last_context: TranscriptContext | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        context_provider: TranscriptContextProvider | None = None,
    ) -> "TranscriptAgent":
        kwargs: dict[str, object] = {
            "api_key": settings.deepseek_api_key,
            "model": settings.deepseek_model,
        }
        if settings.deepseek_base_url:
            kwargs["base_url"] = settings.deepseek_base_url
        if context_provider is None:
            context_provider = RawTranscriptContextProvider(
                store=ChromaTranscriptStore(settings.chroma_path),
                fetcher=SuperdataTranscriptFetcher(settings.superdata_api_key),
            )
        return cls(ChatOpenAI(**kwargs), context_provider)

    def summarize(self, request: SummaryRequest) -> TranscriptSummary:
        context = self._get_context(request.video_id, request.source_url)
        self._ensure_context_size(context.transcript.raw_text)
        content = self._invoke(
            context_text=context.transcript.raw_text,
            user_prompt=build_summary_prompt(request.message),
        )
        data = _json_object(content)
        findings = data.get("top_findings")
        if isinstance(findings, list) and len(findings) > 3:
            data["top_findings"] = findings[:3]
        return TranscriptSummary.model_validate(data)

    def answer(self, request: QuestionRequest) -> TranscriptAnswer:
        context = self._get_context(request.video_id, request.source_url)
        self._ensure_context_size(context.transcript.raw_text)
        content = self._invoke(
            context_text=context.transcript.raw_text,
            user_prompt=build_question_prompt(
                question=request.question,
                video_id=context.transcript.video_id,
            ),
        )
        data = _json_object(content)
        data.setdefault("question", request.question)
        data.setdefault("source_video_id", context.transcript.video_id)
        return TranscriptAnswer.model_validate(data)

    def _get_context(self, video_id: str, source_url: str) -> TranscriptContext:
        context = self.context_provider.get_transcript(video_id, source_url)
        self.last_context = context
        return context

    def _invoke(self, context_text: str, user_prompt: str) -> str:
        response = self.llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                SystemMessage(content=build_transcript_context_prompt(context_text)),
                HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    def _ensure_context_size(self, transcript: str) -> None:
        if len(transcript) > self.max_transcript_chars:
            raise TranscriptTooLongError(
                "Transcript is too long for V1 raw full-transcript prompting. "
                "RAG/chunking is future work."
            )


def _json_object(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {content}") from exc
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    return value
