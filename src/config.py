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
    )
