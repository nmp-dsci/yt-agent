from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from src.agents.context import TranscriptContext
from src.agents.models import QuestionRequest, SummaryRequest
from src.agents.transcript_agent import TranscriptAgent, TranscriptTooLongError
from src.transcripts.models import Transcript


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.response)


class FakeContextProvider:
    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.calls = []

    def get_transcript(self, video_id: str, source_url: str) -> TranscriptContext:
        self.calls.append((video_id, source_url))
        return TranscriptContext(transcript=self.transcript, cache_status="hit")


def test_summarize_returns_pydantic_output(sample_transcript: Transcript) -> None:
    llm = FakeLlm(
        '{"summary": "A useful summary", "top_findings": ["one", "two", "three"]}'
    )
    provider = FakeContextProvider(sample_transcript)
    agent = TranscriptAgent(llm, provider)

    summary = agent.summarize(
        SummaryRequest(
            video_id=sample_transcript.video_id,
            source_url=str(sample_transcript.url),
        )
    )

    assert summary.summary == "A useful summary"
    assert summary.top_findings == ["one", "two", "three"]
    assert "This transcript explains" in llm.messages[1].content
    assert "This transcript explains" not in llm.messages[2].content
    assert provider.calls == [(sample_transcript.video_id, str(sample_transcript.url))]


def test_summarize_trims_extra_findings(sample_transcript: Transcript) -> None:
    llm = FakeLlm(
        '{"summary": "A useful summary", "top_findings": '
        '["one", "two", "three", "four"]}'
    )
    agent = TranscriptAgent(llm, FakeContextProvider(sample_transcript))

    summary = agent.summarize(
        SummaryRequest(
            video_id=sample_transcript.video_id,
            source_url=str(sample_transcript.url),
        )
    )

    assert summary.top_findings == ["one", "two", "three"]


def test_answer_returns_pydantic_output(sample_transcript: Transcript) -> None:
    llm = FakeLlm(
        '{"question": "What?", "answer": "It explains agents.", '
        '"source_video_id": "3hk7nO_q0a8"}'
    )
    agent = TranscriptAgent(llm, FakeContextProvider(sample_transcript))

    answer = agent.answer(
        QuestionRequest(
            video_id=sample_transcript.video_id,
            source_url=str(sample_transcript.url),
            question="What?",
        )
    )

    assert answer.answer == "It explains agents."


def test_rejects_transcript_that_is_too_long(sample_transcript: Transcript) -> None:
    agent = TranscriptAgent(
        FakeLlm("{}"),
        FakeContextProvider(sample_transcript),
        max_transcript_chars=5,
    )

    with pytest.raises(TranscriptTooLongError):
        agent.summarize(
            SummaryRequest(
                video_id=sample_transcript.video_id,
                source_url=str(sample_transcript.url),
            )
        )
