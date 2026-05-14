from __future__ import annotations

from src.transcripts.models import Transcript
from src.transcripts.storage import ChromaTranscriptStore


def test_stores_and_loads_transcript(tmp_path, sample_transcript: Transcript) -> None:
    store = ChromaTranscriptStore(tmp_path / "chroma")
    store.upsert(sample_transcript)

    loaded = store.get(sample_transcript.video_id)

    assert loaded is not None
    assert loaded.video_id == sample_transcript.video_id
    assert loaded.raw_text == sample_transcript.raw_text


def test_upsert_is_idempotent(tmp_path, sample_transcript: Transcript) -> None:
    store = ChromaTranscriptStore(tmp_path / "chroma")
    store.upsert(sample_transcript)
    updated = sample_transcript.model_copy(update={"raw_text": "updated transcript"})
    store.upsert(updated)

    result = store.collection.get(ids=["transcript:3hk7nO_q0a8"])

    assert result["ids"] == ["transcript:3hk7nO_q0a8"]
    assert result["documents"] == ["updated transcript"]
