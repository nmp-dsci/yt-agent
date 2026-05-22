from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src import cli
from src.config import Settings
from src.transcripts.discovery import DiscoveredVideo
from src.transcripts.models import Transcript


class FakeStore:
    def __init__(self, path: Path, *args, **kwargs) -> None:
        self.transcript = None
        self.upserts = 0

    def get(self, video_id: str):
        return self.transcript

    def upsert(self, transcript):
        self.transcript = transcript
        self.upserts += 1


class FakeFetcher:
    calls = 0

    def __init__(self, api_key: str, *args, **kwargs) -> None:
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


class FakeRagAgent:
    last_request = None

    @classmethod
    def from_settings(cls, settings, context_provider=None):
        agent = cls()
        agent.context_provider = context_provider
        agent.last_context = None
        return agent

    def answer(self, request):
        FakeRagAgent.last_request = request
        from src.agents.models import RagTranscriptAnswer

        return RagTranscriptAnswer(question=request.question, answer="rag answer")


class FakeEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class FakeChunkStore:
    def __init__(self, *args, **kwargs) -> None:
        pass


class FakeIndexer:
    last_refresh = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    def index(self, source_url: str, refresh: bool = False, refresh_summary: bool = False):
        FakeIndexer.last_refresh = (source_url, refresh, refresh_summary)

        class Result:
            raw_document = None
            chunks = [object(), object()]
            summary_status = "hit"

        return Result()


class FakeMultiProvider:
    def __init__(self, *args, **kwargs) -> None:
        pass


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
        def __init__(self, path: Path, *args, **kwargs) -> None:
            super().__init__(path, *args, **kwargs)
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
        def __init__(self, path: Path, *args, **kwargs) -> None:
            super().__init__(path, *args, **kwargs)
            self.transcript = transcript

    _patch_cli(monkeypatch, tmp_path, store_cls=CachedStore)
    FakeFetcher.calls = 0

    result = cli.main(
        ["summarize", "https://www.youtube.com/watch?v=3hk7nO_q0a8"]
    )

    assert result == 0
    assert FakeFetcher.calls == 0
    assert "summary" in capsys.readouterr().out


def test_rag_ask_uses_rag_only_agent(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)
    FakeRagAgent.last_request = None

    result = cli.main(["rag-ask", "question", "--top-k", "7"])

    assert result == 0
    assert FakeRagAgent.last_request is not None
    assert FakeRagAgent.last_request.question == "question"
    assert FakeRagAgent.last_request.source_url is None
    assert FakeRagAgent.last_request.top_k == 7
    assert "rag answer" in capsys.readouterr().out


def test_rag_ask_passes_transcript_filter_flags(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)
    FakeRagAgent.last_request = None

    result = cli.main(
        [
            "rag-ask",
            "question",
            "--filter-transcripts",
            "--transcript-filter-top-k",
            "3",
            "--transcript-filter-min-score",
            "0.4",
        ]
    )

    assert result == 0
    assert FakeRagAgent.last_request is not None
    assert FakeRagAgent.last_request.filter_transcripts is True
    assert FakeRagAgent.last_request.transcript_filter_top_k == 3
    assert FakeRagAgent.last_request.transcript_filter_min_score == 0.4
    assert "rag answer" in capsys.readouterr().out


def test_rag_ask_passes_recursive_flags(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)
    FakeRagAgent.last_request = None

    result = cli.main(
        [
            "rag-ask",
            "question",
            "--recursive",
            "--max-depth",
            "1",
            "--max-followups",
            "4",
            "--followup-top-k",
            "6",
            "--novelty-min-chunks",
            "1",
            "--max-total-followups",
            "5",
        ]
    )

    assert result == 0
    request = FakeRagAgent.last_request
    assert request is not None
    assert request.recursive is True
    assert request.recursion_options.max_depth == 1
    assert request.recursion_options.max_followups == 4
    assert request.recursion_options.followup_top_k == 6
    assert request.recursion_options.novelty_min_chunks == 1
    assert request.recursion_options.max_total_followups == 5
    assert "rag answer" in capsys.readouterr().out


def test_rag_ask_uses_recursive_env_default_and_opt_out(
    monkeypatch, tmp_path, capsys
) -> None:
    settings = Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-cli",
        log_transcript_artifacts=False,
        rag_recursive_default=True,
    )
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda require_keys=True: settings)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)

    FakeRagAgent.last_request = None
    assert cli.main(["rag-ask", "question"]) == 0
    assert FakeRagAgent.last_request.recursive is True

    FakeRagAgent.last_request = None
    assert cli.main(["rag-ask", "question", "--no-recursive"]) == 0
    assert FakeRagAgent.last_request.recursive is False
    capsys.readouterr()


def test_index_rag_refreshes_pipeline_dashboard(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    calls = []
    monkeypatch.setattr(cli, "log_raw_transcript_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_summary_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_summary_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "_refresh_rag_pipeline_dashboard",
        lambda settings: calls.append(settings) or tmp_path / "dashboard/rag_pipeline.html",
    )

    result = cli.main(["index-rag", "https://www.youtube.com/watch?v=3hk7nO_q0a8"])

    assert result == 0
    assert calls
    output = capsys.readouterr().out
    assert "RAG pipeline dashboard:" in output


def test_bulk_index_channel_dry_run_writes_run_record(monkeypatch, tmp_path, capsys) -> None:
    class BulkRawStore(FakeStore):
        def get_raw_document(self, video_id: str):
            return None

    class BulkChunkStore(FakeChunkStore):
        def has_chunks(self, video_id: str) -> bool:
            return False

        def count_chunks(self, video_id: str) -> int:
            return 0

    _patch_cli(monkeypatch, tmp_path, store_cls=BulkRawStore)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", BulkChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "_build_summary_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_summary_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "discover_latest_channel_videos",
        lambda channel, limit, client: [
            DiscoveredVideo(
                video_id="aaaaaaaaaaa",
                source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
                title="Video A",
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "_refresh_rag_pipeline_dashboard",
        lambda settings: tmp_path / "dashboard/rag_pipeline.html",
    )

    result = cli.main(
        [
            "bulk-index",
            "channel",
            "--channel",
            "@channel",
            "--latest",
            "1",
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Discovered: 1" in output
    assert "aaaaaaaaaaa discovered" in output
    run_files = list((tmp_path / "ingestion_runs").glob("*.json"))
    assert len(run_files) == 1


def test_refresh_rag_pipeline_dashboard_passes_default_filter_question(
    monkeypatch,
    tmp_path,
) -> None:
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
    calls = {}
    monkeypatch.setattr(cli, "collect_pipeline_rows", lambda settings: [])
    monkeypatch.setattr(
        cli,
        "collect_filter_test_rows",
        lambda settings, rows, question: calls.setdefault("question", question) or [],
    )
    monkeypatch.setattr(cli, "write_dashboard", lambda **kwargs: calls.update(kwargs))

    output = cli._refresh_rag_pipeline_dashboard(settings)

    assert output == Path("dashboard/rag_pipeline.html")
    assert calls["question"] == cli.DEFAULT_FILTER_TEST_QUESTION
    assert calls["filter_test_question"] == cli.DEFAULT_FILTER_TEST_QUESTION


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
    monkeypatch.setattr(cli, "RawTranscriptStore", store_cls)
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
