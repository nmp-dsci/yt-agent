from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class TranscriptSegment(BaseModel):
    text: str
    offset_ms: int | None = None
    duration_ms: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    language: str | None = None


class Transcript(BaseModel):
    video_id: str
    url: HttpUrl
    title: str | None = None
    language: str | None = None
    provider: str = "supadata"
    raw_text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    fetched_at: datetime
