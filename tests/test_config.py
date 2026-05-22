from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError, load_settings


def test_loads_settings_from_env_file(monkeypatch, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "SUPERDATA_API_KEY=super",
                "DEEPSEEK_API_KEY=deep",
                "DEEPSEEK_MODEL=deepseek-v4",
                "YT_AGENT_CHROMA_PATH=.cache/chroma",
                "SUPADATA_TIMEOUT_SECONDS=150",
                "SUPADATA_POLL_INTERVAL_SECONDS=3",
                "SUPADATA_MAX_POLL_SECONDS=900",
                "YT_AGENT_RAG_RECURSIVE_DEFAULT=true",
                "YT_AGENT_RAG_MAX_DEPTH=1",
                "YT_AGENT_RAG_MAX_FOLLOWUPS=4",
                "YT_AGENT_RAG_FOLLOWUP_TOP_K=6",
                "YT_AGENT_RAG_NOVELTY_MIN_CHUNKS=1",
                "YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS=5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(env))
    monkeypatch.delenv("SUPERDATA_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = load_settings()

    assert settings.superdata_api_key == "super"
    assert settings.deepseek_api_key == "deep"
    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.chroma_path.name == "chroma"
    assert settings.supadata_timeout_seconds == 150
    assert settings.supadata_poll_interval_seconds == 3
    assert settings.supadata_max_poll_seconds == 900
    assert settings.rag_recursive_default is True
    assert settings.rag_max_depth == 1
    assert settings.rag_max_followups == 4
    assert settings.rag_followup_top_k == 6
    assert settings.rag_novelty_min_chunks == 1
    assert settings.rag_max_total_followups == 5


def test_accepts_supadata_api_key_alias(monkeypatch, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "SUPADATA_API_KEY=super",
                "DEEPSEEK_API_KEY=deep",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(env))
    monkeypatch.delenv("SUPERDATA_API_KEY", raising=False)
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = load_settings()

    assert settings.superdata_api_key == "super"


def test_missing_env_file_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(tmp_path / "missing.env"))

    with pytest.raises(ConfigError):
        load_settings()
