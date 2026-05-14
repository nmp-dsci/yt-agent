from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.models import Transcript
from src.transcripts.storage import ChromaTranscriptStore


@dataclass(frozen=True)
class TranscriptContext:
    transcript: Transcript
    cache_status: str


class TranscriptContextProvider(Protocol):
    def get_transcript(self, video_id: str, source_url: str) -> TranscriptContext:
        ...


class RawTranscriptContextProvider:
    """Provides full raw transcript context today; replace with RAG later."""

    def __init__(
        self,
        store: ChromaTranscriptStore,
        fetcher: SuperdataTranscriptFetcher,
    ) -> None:
        self.store = store
        self.fetcher = fetcher

    def get_transcript(self, video_id: str, source_url: str) -> TranscriptContext:
        cached = self.store.get(video_id)
        if cached is not None:
            return TranscriptContext(transcript=cached, cache_status="hit")

        transcript = self.fetcher.fetch(source_url)
        self.store.upsert(transcript)
        return TranscriptContext(transcript=transcript, cache_status="miss")

    def refresh_transcript(self, source_url: str) -> TranscriptContext:
        transcript = self.fetcher.fetch(source_url)
        self.store.upsert(transcript)
        return TranscriptContext(transcript=transcript, cache_status="refresh")

    def get_or_refresh_transcript(
        self, video_id: str, source_url: str, no_refresh: bool
    ) -> TranscriptContext:
        if no_refresh:
            cached = self.store.get(video_id)
            if cached is not None:
                return TranscriptContext(transcript=cached, cache_status="hit")
        return self.refresh_transcript(source_url)
