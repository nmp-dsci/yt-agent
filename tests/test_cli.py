from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src import cli
from src.config import Settings
from src.transcripts.models import Transcript


class FakeStore:
    def __init__(self, path: Path) -> None:
        self.transcript = None
        self.upserts = 0

    def get(self, video_id: str):
        return self.transcript

    def upsert(self, transcript):
        self.transcript = transcript
        self.upserts += 1


class FakeFetcher:
    calls = 0

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch(self, url: str) -> Transcript:
        FakeFetcher.calls += 1
        return Transcript(
            video_id="3hk7nO_q0a8",
            url=url,
            raw_text="cached transcript",
            fetched_at=datetime.now(timezone.utc),
        )


class FakeAgent:
    @classmethod
    def from_settings(cls, settings, context_provider=None):
        agent = cls()
        agent.context_provider = context_provider
        agent.last_context = None
        return agent

    def summarize(self, request):
        self.last_context = self.context_provider.get_transcript(
            request.video_id, request.source_url
        )
        from src.agents.models import TranscriptSummary

        return TranscriptSummary(summary="summary", top_findings=["a", "b", "c"])

    def answer(self, request):
        self.last_context = self.context_provider.get_transcript(
            request.video_id, request.source_url
        )
        from src.agents.models import TranscriptAnswer

        return TranscriptAnswer(
            question=request.question,
            answer="answer",
            source_video_id=request.video_id,
        )


def test_cli_routes_summarize_with_cache_miss(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    FakeFetcher.calls = 0

    result = cli.main(
        ["summarize", "https://www.youtube.com/watch?v=3hk7nO_q0a8"]
    )

    assert result == 0
    assert FakeFetcher.calls == 1
    assert "Top 3 findings" in capsys.readouterr().out


def test_fetch_no_refresh_uses_cached_transcript(monkeypatch, tmp_path, capsys) -> None:
    transcript = Transcript(
        video_id="3hk7nO_q0a8",
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        raw_text="already cached",
        fetched_at=datetime.now(timezone.utc),
    )

    class CachedStore(FakeStore):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.transcript = transcript

    _patch_cli(monkeypatch, tmp_path, store_cls=CachedStore)
    FakeFetcher.calls = 0

    result = cli.main(
        ["fetch", "https://www.youtube.com/watch?v=3hk7nO_q0a8", "--no-refresh"]
    )

    assert result == 0
    assert FakeFetcher.calls == 0
    assert "Cache status: hit" in capsys.readouterr().out


def test_summarize_uses_cached_transcript_without_fetch(monkeypatch, tmp_path, capsys) -> None:
    transcript = Transcript(
        video_id="3hk7nO_q0a8",
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        raw_text="already cached",
        fetched_at=datetime.now(timezone.utc),
    )

    class CachedStore(FakeStore):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.transcript = transcript

    _patch_cli(monkeypatch, tmp_path, store_cls=CachedStore)
    FakeFetcher.calls = 0

    result = cli.main(
        ["summarize", "https://www.youtube.com/watch?v=3hk7nO_q0a8"]
    )

    assert result == 0
    assert FakeFetcher.calls == 0
    assert "summary" in capsys.readouterr().out


def _patch_cli(monkeypatch, tmp_path, store_cls=FakeStore) -> None:
    settings = Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-cli",
        log_transcript_artifacts=False,
    )
    monkeypatch.setattr(cli, "load_settings", lambda require_keys=True: settings)
    monkeypatch.setattr(cli, "ChromaTranscriptStore", store_cls)
    monkeypatch.setattr(cli, "SuperdataTranscriptFetcher", FakeFetcher)
    monkeypatch.setattr(cli, "TranscriptAgent", FakeAgent)
    monkeypatch.setattr(cli, "cli_run", _null_run)
    monkeypatch.setattr(cli, "log_transcript", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "log_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "log_answer", lambda *args, **kwargs: None)


class _null_run:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
