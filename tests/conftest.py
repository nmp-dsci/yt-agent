from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.transcripts.models import Transcript


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        video_id="3hk7nO_q0a8",
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        title="Sample",
        language="en",
        provider="supadata",
        raw_text="This transcript explains three practical findings about agent systems.",
        fetched_at=datetime(2026, 5, 14, tzinfo=timezone.utc),
    )
