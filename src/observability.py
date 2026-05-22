from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow

from src.agents.models import RecursionTrace, TranscriptAnswer, TranscriptSummary
from src.config import Settings
from src.rag.models import (
    ContextComparisonResult,
    RawTranscriptDocument,
    RetrievedChunk,
    RetrievedTranscriptSummary,
)
from src.transcripts.models import Transcript


def setup_mlflow(settings: Settings) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    try:
        mlflow.langchain.autolog()
    except Exception:
        # MLflow tracing should not prevent the CLI from running.
        pass


@contextmanager
def cli_run(command: str, settings: Settings, video_id: str | None = None) -> Iterator[None]:
    setup_mlflow(settings)
    with mlflow.start_run(run_name=f"{command}:{video_id or 'unknown'}"):
        mlflow.set_tag("command", command)
        mlflow.set_tag("status", "running")
        if video_id:
            mlflow.log_param("video_id", video_id)
        mlflow.log_param("model", settings.deepseek_model)
        try:
            yield
        except Exception:
            mlflow.set_tag("status", "failed")
            raise
        else:
            mlflow.set_tag("status", "success")


def log_transcript(
    transcript: Transcript,
    cache_status: str,
    settings: Settings,
) -> None:
    mlflow.log_param("video_id", transcript.video_id)
    mlflow.log_param("provider", transcript.provider)
    mlflow.log_param("cache_status", cache_status)
    mlflow.log_metric("transcript_chars", len(transcript.raw_text))
    mlflow.set_tags(
        {
            "video_id": transcript.video_id,
            "provider": transcript.provider,
            "cache_status": cache_status,
            "transcript_hash": hashlib.sha256(
                transcript.raw_text.encode("utf-8")
            ).hexdigest(),
        }
    )
    metadata = {
        "video_id": transcript.video_id,
        "url": str(transcript.url),
        "title": transcript.title,
        "language": transcript.language,
        "provider": transcript.provider,
        "fetched_at": transcript.fetched_at.isoformat(),
        "transcript_chars": len(transcript.raw_text),
    }
    _log_json_artifact(metadata, "transcript_metadata.json")
    if settings.log_transcript_artifacts:
        _log_text_artifact(transcript.raw_text, "transcript.txt")


def log_summary(summary: TranscriptSummary) -> None:
    _log_json_artifact(summary.model_dump(), "summary.json")


def log_answer(answer: TranscriptAnswer) -> None:
    _log_json_artifact(answer.model_dump(), "answer.json")


def log_recursion_trace(trace: RecursionTrace | None) -> None:
    if trace is None:
        return
    mlflow.log_param("rag_recursion_terminated_reason", trace.terminated_reason)
    mlflow.log_metric("rag_followups_proposed", trace.total_followups_proposed)
    mlflow.log_metric("rag_followups_executed", trace.total_followups_executed)
    mlflow.log_metric(
        "rag_recursion_llm_calls",
        sum(stage.llm_calls for stage in trace.stages),
    )
    mlflow.log_metric(
        "rag_recursion_retrievals",
        sum(stage.retrievals for stage in trace.stages),
    )
    _log_json_artifact(trace.model_dump(mode="json"), "rag_recursion_trace.json")


def log_context_details(
    context_mode: str,
    top_k: int | None = None,
    retrieved_chunks: list[RetrievedChunk] | None = None,
    raw_prompt_tokens_estimate: int | None = None,
    rag_prompt_tokens_estimate: int | None = None,
) -> None:
    mlflow.log_param("context_mode", context_mode)
    if top_k is not None:
        mlflow.log_param("top_k", top_k)
    if raw_prompt_tokens_estimate is not None:
        mlflow.log_metric("raw_prompt_tokens_estimate", raw_prompt_tokens_estimate)
    if rag_prompt_tokens_estimate is not None:
        mlflow.log_metric("rag_prompt_tokens_estimate", rag_prompt_tokens_estimate)
    if retrieved_chunks:
        mlflow.set_tag(
            "retrieved_chunk_ids",
            ",".join(f"chunk:{chunk.video_id}:{chunk.chunk_index}" for chunk in retrieved_chunks),
        )
        mlflow.set_tag(
            "retrieved_chunk_scores",
            ",".join("" if chunk.score is None else f"{chunk.score:.6f}" for chunk in retrieved_chunks),
        )
        mlflow.set_tag(
            "retrieved_chunk_time_ranges",
            ",".join(
                f"{chunk.start_seconds}-{chunk.end_seconds}" for chunk in retrieved_chunks
            ),
        )
        _log_json_artifact(
            {"chunks": [chunk.model_dump(mode="json") for chunk in retrieved_chunks]},
            "rag_chunks.json",
        )


def log_transcript_filter_details(
    enabled: bool,
    selected_transcripts: list[RetrievedTranscriptSummary] | None = None,
    filter_top_k: int | None = None,
    min_score: float | None = None,
    retrieved_chunks: list[RetrievedChunk] | None = None,
) -> None:
    selected_transcripts = selected_transcripts or []
    retrieved_chunks = retrieved_chunks or []
    mlflow.log_param("transcript_filter_enabled", enabled)
    if filter_top_k is not None:
        mlflow.log_param("transcript_filter_top_k", filter_top_k)
    if min_score is not None:
        mlflow.log_param("transcript_filter_min_score", min_score)
    mlflow.log_metric("selected_transcript_count", len(selected_transcripts))
    mlflow.log_metric("retrieved_chunk_count", len(retrieved_chunks))
    scores = [
        transcript.score
        for transcript in selected_transcripts
        if transcript.score is not None
    ]
    if scores:
        mlflow.log_metric("selected_transcript_score_max", max(scores))
        mlflow.log_metric("selected_transcript_score_min", min(scores))
    if selected_transcripts:
        mlflow.set_tag(
            "selected_video_ids",
            ",".join(transcript.video_id for transcript in selected_transcripts),
        )
        mlflow.set_tag(
            "selected_transcript_ids",
            ",".join(
                transcript.transcript_id for transcript in selected_transcripts
            ),
        )
    if retrieved_chunks:
        mlflow.set_tag(
            "retrieved_chunk_ids",
            ",".join(f"chunk:{chunk.video_id}:{chunk.chunk_index}" for chunk in retrieved_chunks),
        )
    _log_json_artifact(
        {
            "enabled": enabled,
            "filter_top_k": filter_top_k,
            "min_score": min_score,
            "selected_transcripts": [
                transcript.model_dump(mode="json")
                for transcript in selected_transcripts
            ],
        },
        "transcript_filter.json",
    )
    if retrieved_chunks:
        _log_json_artifact(
            {"chunks": [chunk.model_dump(mode="json") for chunk in retrieved_chunks]},
            "rag_chunks.json",
        )


def log_raw_transcript_metadata(document: RawTranscriptDocument) -> None:
    _log_json_artifact(
        {
            "transcript_id": document.transcript_id,
            "video_id": document.video_id,
            "source_url": str(document.source_url),
            "source_collection": document.source_collection,
            "provider": document.provider,
            "title": document.title,
            "language": document.language,
            "fetched_at": document.fetched_at,
            "segment_count": len(document.segments),
        },
        "raw_transcript_metadata.json",
    )


def log_context_comparison(comparison: ContextComparisonResult) -> None:
    mlflow.log_metric("semantic_similarity", comparison.semantic_similarity)
    mlflow.log_metric(
        "raw_prompt_tokens_estimate", comparison.raw_prompt_tokens_estimate
    )
    mlflow.log_metric(
        "rag_prompt_tokens_estimate", comparison.rag_prompt_tokens_estimate
    )
    mlflow.log_metric("token_savings_percent", comparison.token_savings_percent)
    _log_json_artifact(comparison.model_dump(mode="json"), "context_comparison.json")


def _log_json_artifact(payload: dict[str, object], artifact_name: str) -> None:
    _log_text_artifact(json.dumps(payload, indent=2), artifact_name)


def _log_text_artifact(content: str, artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / artifact_name
        path.write_text(content, encoding="utf-8")
        mlflow.log_artifact(str(path))
