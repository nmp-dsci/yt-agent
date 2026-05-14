from __future__ import annotations

from src.transcripts.fetcher import SuperdataTranscriptFetcher


def test_normalizes_supadata_segment_response() -> None:
    fetcher = SuperdataTranscriptFetcher("key")

    transcript = fetcher._normalize_response(
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        video_id="3hk7nO_q0a8",
        data={
            "content": [
                {"text": "hello", "offset": 0, "duration": 1000, "lang": "en"},
                {"text": "world", "offset": 1000, "duration": 1000, "lang": "en"},
            ],
            "lang": "en",
        },
    )

    assert transcript.raw_text == "hello world"
    assert transcript.segments[0].offset_ms == 0
    assert transcript.segments[0].duration_ms == 1000
    assert transcript.segments[0].start_seconds == 0
    assert transcript.segments[0].end_seconds == 1
    assert transcript.segments[0].language == "en"
    assert transcript.language == "en"


def test_normalizes_supadata_text_response() -> None:
    fetcher = SuperdataTranscriptFetcher("key")

    transcript = fetcher._normalize_response(
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        video_id="3hk7nO_q0a8",
        data={"content": "plain transcript", "lang": "en"},
    )

    assert transcript.raw_text == "plain transcript"
    assert transcript.segments == []
