from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import chromadb
from pydantic import HttpUrl

from src.transcripts.models import Transcript


class ChromaTranscriptStore:
    collection_name = "transcripts"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def get(self, video_id: str) -> Transcript | None:
        result = self.collection.get(ids=[self._id(video_id)], include=["documents", "metadatas"])
        ids = result.get("ids", [])
        if not ids:
            return None
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not documents or not metadatas:
            return None
        metadata = metadatas[0] or {}
        return Transcript(
            video_id=str(metadata.get("video_id", video_id)),
            url=HttpUrl(str(metadata["url"])),
            title=_none_if_empty(metadata.get("title")),
            language=_none_if_empty(metadata.get("language")),
            provider=str(metadata.get("provider", "supadata")),
            raw_text=documents[0],
            fetched_at=_parse_datetime(str(metadata.get("fetched_at"))),
        )

    def upsert(self, transcript: Transcript) -> None:
        self.collection.upsert(
            ids=[self._id(transcript.video_id)],
            documents=[transcript.raw_text],
            metadatas=[self._metadata(transcript)],
        )

    def _metadata(self, transcript: Transcript) -> dict[str, str | int]:
        return {
            "video_id": transcript.video_id,
            "url": str(transcript.url),
            "title": transcript.title or "",
            "language": transcript.language or "",
            "provider": transcript.provider,
            "fetched_at": transcript.fetched_at.isoformat(),
        }

    def _id(self, video_id: str) -> str:
        return f"transcript:{video_id}"


def _parse_datetime(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _none_if_empty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
