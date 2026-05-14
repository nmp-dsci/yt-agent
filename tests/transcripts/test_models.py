from __future__ import annotations

from datetime import datetime, timezone

from src.transcripts.models import Transcript, TranscriptSegment


def test_transcript_model_validates_segments() -> None:
    transcript = Transcript(
        video_id="3hk7nO_q0a8",
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        raw_text="hello world",
        segments=[TranscriptSegment(text="hello", start_seconds=0, end_seconds=1)],
        fetched_at=datetime.now(timezone.utc),
    )

    assert transcript.video_id == "3hk7nO_q0a8"
    assert transcript.segments[0].text == "hello"
