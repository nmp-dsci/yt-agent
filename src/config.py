from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    superdata_api_key: str
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str | None
    chroma_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    log_transcript_artifacts: bool
    raw_transcript_collection: str = "raw_transcripts"
    chunk_collection: str = "transcript_chunks"
    transcript_summary_collection: str = "transcript_summaries"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 10
    transcript_filter_top_k: int = 5
    transcript_filter_min_score: float = 0.25
    rag_recursive_default: bool = False
    rag_max_depth: int = 1
    rag_max_followups: int = 3
    rag_followup_top_k: int | None = None
    rag_novelty_min_chunks: int = 2
    rag_max_total_followups: int | None = None
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 150
    supadata_timeout_seconds: float = 120.0
    supadata_poll_interval_seconds: float = 2.0
    supadata_max_poll_seconds: float = 600.0
    discovery_cache_ttl_hours: float = 24.0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return _project_root() / path


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def load_settings(require_keys: bool = True) -> Settings:
    env_path = Path(os.environ.get("YT_AGENT_ENV_PATH", "~/.env")).expanduser()
    if env_path.exists():
        load_dotenv(env_path, override=False)
    elif require_keys:
        raise ConfigError(f"Missing env file: {env_path}")

    superdata_api_key = os.environ.get("SUPERDATA_API_KEY") or os.environ.get(
        "SUPADATA_API_KEY", ""
    )
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    missing: list[str] = []
    if require_keys and not superdata_api_key:
        missing.append("SUPERDATA_API_KEY or SUPADATA_API_KEY")
    if require_keys and not deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required configuration: {joined}")

    configured_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4")
    api_model = "deepseek-v4-flash" if configured_model == "deepseek-v4" else configured_model

    return Settings(
        superdata_api_key=superdata_api_key,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=api_model,
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com",
        chroma_path=_resolve_project_path(
            os.environ.get("YT_AGENT_CHROMA_PATH", ".yt-agent/chroma")
        ),
        mlflow_tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI", "file:.yt-agent/mlruns"
        ),
        mlflow_experiment_name=os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "yt-agent-v1"
        ),
        log_transcript_artifacts=_bool_env(
            os.environ.get("YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS"), default=False
        ),
        raw_transcript_collection=os.environ.get(
            "YT_AGENT_RAW_TRANSCRIPT_COLLECTION", "raw_transcripts"
        ),
        chunk_collection=os.environ.get(
            "YT_AGENT_CHUNK_COLLECTION", "transcript_chunks"
        ),
        transcript_summary_collection=os.environ.get(
            "YT_AGENT_TRANSCRIPT_SUMMARY_COLLECTION", "transcript_summaries"
        ),
        embedding_model=os.environ.get(
            "YT_AGENT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        rag_top_k=_int_env("YT_AGENT_RAG_TOP_K", 10),
        transcript_filter_top_k=_int_env("YT_AGENT_TRANSCRIPT_FILTER_TOP_K", 5),
        transcript_filter_min_score=_float_env(
            "YT_AGENT_TRANSCRIPT_FILTER_MIN_SCORE", 0.25
        ),
        rag_recursive_default=_bool_env(
            os.environ.get("YT_AGENT_RAG_RECURSIVE_DEFAULT"), default=False
        ),
        rag_max_depth=_int_env("YT_AGENT_RAG_MAX_DEPTH", 1),
        rag_max_followups=_int_env("YT_AGENT_RAG_MAX_FOLLOWUPS", 3),
        rag_followup_top_k=_optional_int_env("YT_AGENT_RAG_FOLLOWUP_TOP_K"),
        rag_novelty_min_chunks=_int_env("YT_AGENT_RAG_NOVELTY_MIN_CHUNKS", 2),
        rag_max_total_followups=_optional_int_env("YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS"),
        chunk_target_chars=_int_env("YT_AGENT_CHUNK_TARGET_CHARS", 1200),
        chunk_overlap_chars=_int_env("YT_AGENT_CHUNK_OVERLAP_CHARS", 150),
        supadata_timeout_seconds=_float_env("SUPADATA_TIMEOUT_SECONDS", 120.0),
        supadata_poll_interval_seconds=_float_env(
            "SUPADATA_POLL_INTERVAL_SECONDS", 2.0
        ),
        supadata_max_poll_seconds=_float_env("SUPADATA_MAX_POLL_SECONDS", 600.0),
        discovery_cache_ttl_hours=_float_env(
            "YT_AGENT_DISCOVERY_CACHE_TTL_HOURS", 24.0
        ),
    )
