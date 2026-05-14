from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.models import (
    QuestionRequest,
    SummaryRequest,
    TranscriptAnswer,
    TranscriptSummary,
)
from src.transcripts.models import Transcript


def test_agent_request_models_use_transcript(sample_transcript: Transcript) -> None:
    summary_request = SummaryRequest(
        video_id=sample_transcript.video_id,
        source_url=str(sample_transcript.url),
    )
    assert summary_request.video_id == "3hk7nO_q0a8"
    assert (
        QuestionRequest(
            video_id=sample_transcript.video_id,
            source_url=str(sample_transcript.url),
            question="What is it about?",
        ).question
        == "What is it about?"
    )


def test_agent_output_models_validate_top_three_findings() -> None:
    summary = TranscriptSummary(summary="Short", top_findings=["a", "b", "c"])
    answer = TranscriptAnswer(question="Q", answer="A", source_video_id="3hk7nO_q0a8")

    assert len(summary.top_findings) == 3
    assert answer.source_video_id == "3hk7nO_q0a8"


def test_summary_requires_exactly_three_findings() -> None:
    with pytest.raises(ValidationError):
        TranscriptSummary(summary="Short", top_findings=["a", "b"])
