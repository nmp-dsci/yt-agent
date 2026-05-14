from __future__ import annotations

from pathlib import Path

import mlflow

from src.config import Settings
from src.observability import cli_run, log_summary, setup_mlflow
from src.agents.models import TranscriptSummary


def test_mlflow_setup_uses_configured_tracking_uri(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    setup_mlflow(settings)

    assert mlflow.get_tracking_uri() == f"file:{tmp_path / 'mlruns'}"


def test_cli_run_logs_summary_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with cli_run("summarize", settings, "3hk7nO_q0a8"):
        log_summary(TranscriptSummary(summary="s", top_findings=["a", "b", "c"]))

    runs = mlflow.search_runs(experiment_names=[settings.mlflow_experiment_name])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.command"] == "summarize"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-exp",
        log_transcript_artifacts=False,
    )
