from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptSummary(BaseModel):
    summary: str
    top_findings: list[str] = Field(min_length=3, max_length=3)


class TranscriptAnswer(BaseModel):
    question: str
    answer: str
    source_video_id: str


class SummaryRequest(BaseModel):
    video_id: str
    source_url: str
    message: str = "Summarize this transcript."


class QuestionRequest(BaseModel):
    video_id: str
    source_url: str
    question: str
