from __future__ import annotations

import pytest

from src.transcripts.youtube import YouTubeUrlError, extract_video_id


def test_extracts_video_id_from_watch_url() -> None:
    assert (
        extract_video_id("https://www.youtube.com/watch?v=3hk7nO_q0a8")
        == "3hk7nO_q0a8"
    )


def test_extracts_video_id_from_short_url() -> None:
    assert extract_video_id("https://youtu.be/3hk7nO_q0a8") == "3hk7nO_q0a8"


def test_rejects_invalid_url() -> None:
    with pytest.raises(YouTubeUrlError):
        extract_video_id("https://example.com/nope")
