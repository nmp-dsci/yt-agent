from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow

from src.agents.models import TranscriptAnswer, TranscriptSummary
from src.config import Settings
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


def _log_json_artifact(payload: dict[str, object], artifact_name: str) -> None:
    _log_text_artifact(json.dumps(payload, indent=2), artifact_name)


def _log_text_artifact(content: str, artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / artifact_name
        path.write_text(content, encoding="utf-8")
        mlflow.log_artifact(str(path))
