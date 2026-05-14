# Spec: Agent V1 Transcript CLI

Status: ready
Date: 2026-05-14

## Summary

Build the first version of the YouTube transcript agent system: a Python CLI that fetches a YouTube transcript, stores the raw transcript in Chroma local persistence, and uses an LLM agent to answer questions and summarize the transcript.

This version should prove the direct transcript workflow before adding RAG. The implementation should keep transcript ingestion, storage, agent logic, and CLI code separated so a later RAG pipeline can reuse the stored transcript data.

## Goal

Create a LangChain-based transcript agent that can:

- Fetch the transcript for a YouTube video through Superdata.
- Store the raw transcript locally.
- Feed the transcript to `deepseek-v4`.
- Answer user questions about the transcript.
- Generate a useful summary of the transcript.
- Run from a Python CLI.

The first test transcript is:

```text
https://www.youtube.com/watch?v=3hk7nO_q0a8
```

## Non-Goals

- Do not build a RAG pipeline in V1.
- Do not support multiple transcripts in a single agent workflow yet.
- Do not build a web UI.
- Do not add an external database service, queue, or background worker.
- Do not commit downloaded transcript content unless explicitly requested.

## Technology Choices

- Language: Python.
- Python package and environment management: `uv`.
- Agent framework: LangChain.
- Transcript provider: Superdata.
- LLM: DeepSeek V4 family. Accept `deepseek-v4` in config, but resolve it to a concrete API model such as `deepseek-v4-flash` for live calls.
- Storage: Chroma local persistence for raw transcripts and metadata only.
- Observability: MLflow local tracking for CLI runs, transcript fetch/cache events, and LangChain LLM calls.
- Delivery surface: CLI tool.

## Proposed Project Structure

Implement this structure unless the codebase already establishes a better pattern. Keep modules small and reusable so `fetch`, `summarize`, and `ask` share the same config, fetcher, storage, models, prompts, and agent code.

```text
src/
  __init__.py             # Marks src as an importable package for python -m src.cli
  cli.py                  # argparse entrypoint and command routing only
  config.py               # load ~/.env or YT_AGENT_ENV_PATH and expose typed settings
  observability.py        # MLflow setup and run/span helpers
  transcripts/
    __init__.py           # Marks transcript utilities as an importable subpackage
    fetcher.py            # Superdata transcript fetching
    models.py             # Pydantic transcript data structures
    storage.py            # Chroma persistence
    youtube.py            # YouTube URL parsing and video_id extraction
  agents/
    __init__.py           # Marks agent utilities as an importable subpackage
    context.py            # Backend transcript context provider; replaceable by RAG later
    models.py             # Pydantic LLM request and response models
    transcript_agent.py   # Transcript summary and Q&A orchestration
    prompts.py            # System, summary, and Q&A prompt templates
tests/
  test_cli.py             # CLI argument parsing and command routing tests
  test_config.py          # Env loading, defaults, and required key validation tests
  test_observability.py   # MLflow setup and safe logging tests
  transcripts/
    test_fetcher.py       # Superdata response normalization tests with mocked calls
    test_models.py        # Transcript Pydantic model validation tests
    test_storage.py       # Chroma transcript store/cache behavior tests
    test_youtube.py       # YouTube URL and video_id parsing tests
  agents/
    test_models.py        # Agent request/response Pydantic model validation tests
    test_transcript_agent.py # Summary and Q&A orchestration tests with mocked LLM
    test_prompts.py       # Prompt construction and transcript/question insertion tests
```

Module responsibilities:

- `src/cli.py`: parse arguments, load settings, wire dependencies, print plain-text output. Do not put provider, storage, prompt, or LLM logic here.
- `src/config.py`: centralize env-file loading, required key validation, defaults, and path resolution.
- `src/observability.py`: centralize MLflow setup, experiment naming, run tags, and safe artifact logging.
- `src/transcripts/youtube.py`: extract and validate YouTube video IDs from supported URL forms.
- `src/transcripts/fetcher.py`: call Superdata and normalize provider responses into Pydantic models.
- `src/transcripts/storage.py`: own all Chroma client, collection, upsert, and lookup behavior.
- `src/transcripts/models.py`: define shared Pydantic data models.
- `src/agents/context.py`: provide transcript context to the agent from Chroma today, and define the boundary a future RAG context provider can replace.
- `src/agents/models.py`: define LLM agent request and response Pydantic models.
- `src/agents/prompts.py`: define system, summary, and Q&A prompt templates.
- `src/agents/transcript_agent.py`: orchestrate prompt construction, LangChain calls, summary behavior, and Q&A behavior.

Reuse requirements:

- `summarize` and `ask` must both call the same storage lookup path before fetching.
- `fetch`, `summarize`, and `ask` must all use the same `SuperdataTranscriptFetcher` implementation.
- All commands must use the same `ChromaTranscriptStore` implementation.
- All commands must load settings through `src/config.py`.
- All commands must initialize observability through `src/observability.py`.
- Summary and Q&A prompts must reuse the same system prompt constant.
- Tests should mock at module boundaries: Superdata fetcher, Chroma store, and LLM client.

## Configuration

Secrets and provider settings must be loaded from a local `~/.env` file. The implementation should read `~/.env` at startup and populate environment variables from it.

Do not commit `~/.env` or copy secrets into project files.

Expected `~/.env` values:

```text
SUPERDATA_API_KEY=<required for transcript fetching>
DEEPSEEK_API_KEY=<required for LLM calls>
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=<optional OpenAI-compatible DeepSeek API base URL>
YT_AGENT_CHROMA_PATH=.yt-agent/chroma
MLFLOW_TRACKING_URI=file:.yt-agent/mlruns
MLFLOW_EXPERIMENT_NAME=yt-agent-v1
YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=false
```

If `DEEPSEEK_MODEL` is not set, default to `deepseek-v4-flash`. If `DEEPSEEK_MODEL=deepseek-v4`, map it to `deepseek-v4-flash` because the live DeepSeek API requires a concrete V4 model ID such as `deepseek-v4-flash` or `deepseek-v4-pro`.

If `DEEPSEEK_BASE_URL` is not set, use the default base URL expected by the selected DeepSeek/LangChain adapter.

If `YT_AGENT_CHROMA_PATH` is not set, default to `.yt-agent/chroma` relative to the project root.

If `MLFLOW_TRACKING_URI` is not set, default to `file:.yt-agent/mlruns`.

If `MLFLOW_EXPERIMENT_NAME` is not set, default to `yt-agent-v1`.

If `YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS` is not set, default to `false`.

If `YT_AGENT_ENV_PATH` is already present in the process environment, load configuration from that file instead of `~/.env`. This makes config loading testable without touching the real home directory.

Add setup instructions to `readme.md` when implementing.

## Python Environment and Dependencies

Use `uv` for Python environment and package management. Do not introduce another package manager unless explicitly requested.

Expected setup files:

```text
pyproject.toml
uv.lock
```

Recommended setup commands:

```bash
uv init --package
uv add langchain chromadb pydantic python-dotenv mlflow
uv add --dev pytest
```

Add provider-specific packages as needed for the actual implementation, for example:

```bash
uv add langchain-openai
```

If Superdata provides an official Python client, use it and add it with `uv add`. If no suitable client is available, use a small HTTP client dependency such as `httpx` and document the Superdata endpoint assumptions in `readme.md`.

Runtime commands should be documented with `uv run`, for example:

```bash
uv run python -m src.cli fetch "https://www.youtube.com/watch?v=3hk7nO_q0a8"
uv run python -m src.cli summarize "https://www.youtube.com/watch?v=3hk7nO_q0a8"
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "What is this video about?"
uv run pytest
```

The implementation must update `readme.md` with:

- `uv` install/setup assumptions.
- Dependency installation commands.
- Required `~/.env` keys.
- CLI examples using `uv run`.
- Test command using `uv run pytest`.

## Data Models

Define explicit Pydantic models in `src/transcripts/models.py` and `src/agents/models.py`. Use these models as the boundary between fetching, storage, CLI commands, and agent logic. Do not pass raw provider dictionaries through the application.

Recommended transcript/storage models in `src/transcripts/models.py`:

```python
from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class TranscriptSegment(BaseModel):
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None


class Transcript(BaseModel):
    video_id: str
    url: HttpUrl
    title: str | None = None
    language: str | None = None
    provider: str = "superdata"
    raw_text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    fetched_at: datetime
```

Recommended LLM agent input/output models in `src/agents/models.py`:

```python
from pydantic import BaseModel, Field


class TranscriptSummary(BaseModel):
    summary: str
    top_findings: list[str] = Field(min_length=3, max_length=3)


class TranscriptAnswer(BaseModel):
    question: str
    answer: str
    source_video_id: str


class SummaryRequest(BaseModel):
    video_id: str
    source_url: str
    message: str = "Summarize this transcript."


class QuestionRequest(BaseModel):
    video_id: str
    source_url: str
    question: str
```

Model rules:

- `Transcript.raw_text` is the canonical text stored in Chroma's `transcripts` collection.
- `Transcript.segments` should be populated when Superdata returns timestamped transcript parts.
- `SummaryRequest` is the typed input to the summary agent path and should contain only user/task inputs such as `video_id`, source URL, and summary message. It must not contain raw transcript text.
- `QuestionRequest` is the typed input to the Q&A agent path and should contain only user/task inputs such as `video_id`, source URL, and question. It must not contain raw transcript text.
- `TranscriptSummary` is the typed output from the summary agent path.
- `TranscriptAnswer` is the typed output from the Q&A agent path.
- `TranscriptSummary.top_findings` must contain exactly three findings for the non-negotiable completion test.
- Convert `HttpUrl` values to strings before storing them in Chroma metadata.
- Chroma metadata must use scalar values only. Do not store nested Pydantic objects, lists of segments, or full provider payloads as metadata.

## Storage Design

Use Chroma with local persistence as the single storage solution. Chroma should store raw transcripts and metadata in V1. Do not store chunks or embeddings in V1.

Recommended Chroma path:

```text
.yt-agent/chroma
```

Recommended Chroma collections:

```text
transcripts
```

Use the `transcripts` collection as the canonical V1 transcript store:

- `id`: `transcript:{video_id}`
- `document`: full raw transcript text
- metadata:
  - `video_id`
  - `url`
  - `title`
  - `language`
  - `provider`
  - `fetched_at`

Chroma metadata must stay limited to simple scalar values. In V1, do not store the full nested Superdata provider payload in Chroma metadata. Store the normalized raw transcript text as the Chroma document and store only essential scalar metadata.

Do not add SQLite or another storage fallback unless explicitly requested.

Fetching the same video more than once must be idempotent. Use the stable `transcript:{video_id}` document id and upsert or replace the existing Chroma record instead of creating duplicates.

## Transcript Cache Behavior

Chroma local persistence is also the transcript cache. A cached transcript is a document in the `transcripts` collection with id `transcript:{video_id}`.

Cache rules:

- `summarize` and `ask` must check Chroma before calling Superdata.
- On cache hit, use the cached `Transcript.raw_text` and metadata. Do not call Superdata.
- On cache miss, call Superdata, normalize the response into a `Transcript`, store it in Chroma, then continue the requested command.
- `fetch` should refresh the transcript by default: call Superdata and upsert the Chroma record for `transcript:{video_id}`.
- Add an optional `--no-refresh` flag to `fetch`. When `--no-refresh` is used and the transcript is already cached, return the cached transcript metadata without calling Superdata.
- Cache lookup must be by parsed `video_id`, not by raw URL string, so different YouTube URL formats for the same video reuse the same cached transcript.
- Cache writes must be idempotent and must not create duplicate records for the same `video_id`.
- Cache metadata should include `fetched_at` so users can inspect when the transcript was last refreshed.

Do not implement chunking, embeddings, vector search, or RAG retrieval in V1. The only persisted transcript content in V1 is the full raw transcript document in the Chroma `transcripts` collection.

Recommended Chroma setup:

```python
import chromadb

client = chromadb.PersistentClient(path=".yt-agent/chroma")
transcripts = client.get_or_create_collection("transcripts")
```

When RAG is added later, use the same persisted Chroma path and add a separate chunk collection then.

## Observability and Tracing

Use MLflow as the V1 observability system. MLflow is a good fit because it can run with local file-based tracking for this prototype and has LangChain tracing support through `mlflow.langchain.autolog()`.

Observability goals:

- Track each CLI command as an MLflow run.
- Capture command type: `fetch`, `summarize`, or `ask`.
- Capture `video_id`, source URL, cache hit/miss, provider name, model name, and success/failure status.
- Capture transcript fetch duration, Chroma storage duration, and LLM call duration where practical.
- Capture LangChain LLM call traces through MLflow autologging.
- Capture summary and Q&A outputs as MLflow artifacts.

Default local tracking path:

```text
.yt-agent/mlruns
```

Recommended setup:

```python
import mlflow

mlflow.set_tracking_uri("file:.yt-agent/mlruns")
mlflow.set_experiment("yt-agent-v1")
mlflow.langchain.autolog()
```

Transcript logging rules:

- Always log transcript metadata: `video_id`, `url`, `title`, `language`, `provider`, `fetched_at`, transcript character count, and a stable hash of `raw_text`.
- Do not log full raw transcript text to MLflow by default.
- If `YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=true`, log the full normalized transcript as a local MLflow artifact named `transcript.txt`.
- Never log API keys or raw environment values.
- Log LLM prompts and outputs only through MLflow/LangChain tracing and output artifacts. If this becomes too noisy or sensitive, add a config flag later to disable prompt/output tracing.

Required MLflow artifacts:

- `summary.json` for `summarize`, serialized from `TranscriptSummary`.
- `answer.json` for `ask`, serialized from `TranscriptAnswer`.
- `transcript.txt` only when `YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=true`.

Required MLflow tags or params:

- `command`
- `video_id`
- `model`
- `provider`
- `cache_status`
- `transcript_chars`
- `status`

## CLI Behavior

The CLI should support three workflows.

Fetch transcript:

```bash
uv run python -m src.cli fetch "https://www.youtube.com/watch?v=3hk7nO_q0a8"
```

Summarize transcript:

```bash
uv run python -m src.cli summarize "https://www.youtube.com/watch?v=3hk7nO_q0a8"
```

Ask a question:

```bash
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "What are the main points?"
```

Expected behavior:

- `fetch` downloads the transcript and stores it in Chroma.
- `fetch --no-refresh` uses the cached transcript if available, otherwise fetches it.
- `summarize` uses the cached transcript if available, otherwise fetches and caches it first.
- `ask` uses the cached transcript if available, otherwise fetches and caches it first.
- Running `fetch` repeatedly for the same video updates or reuses the same Chroma transcript record.
- CLI output should be plain text.
- Errors should explain missing `~/.env`, missing required keys, transcript fetch failures, or LLM failures clearly.

For the `python -m src.cli` entrypoint, ensure `src` is an importable package by adding `src/__init__.py`. If the implementation chooses a named package such as `yt_agent`, update all CLI examples and `readme.md` consistently.

## Agent Behavior

The V1 agent should use the transcript as source material. It should not claim facts that are not supported by the transcript.

The LangChain user message should contain only the user's summary instruction or question plus output-format instructions. Transcript text must be pulled in by the backend context provider and supplied to the LLM as internal context, not passed as part of the user's question/request model. In V1, the context provider returns the cached raw transcript from Chroma. In a future RAG version, this provider can be replaced with one that returns retrieved transcript chunks.

For long transcripts, keep V1 behavior explicit:

- If the transcript fits in the model context, send the full transcript.
- If the transcript does not fit in the model context, fail clearly with a message that V1 only supports raw full-transcript prompting and that RAG/chunking is future work.
- Do not silently truncate the transcript.

## System Prompt Draft

Use this as the starting system prompt:

```text
You are a YouTube transcript analysis agent.

Your job is to answer questions and summarize videos using only the transcript text provided by the system. Be accurate, concise, and explicit about uncertainty.

Rules:
- Use only the transcript as evidence.
- If the transcript does not contain enough information to answer, say that the transcript does not provide enough information.
- Do not invent names, dates, claims, or conclusions.
- When answering a question, prefer a direct answer first, followed by brief supporting details.
- When summarizing, identify the main topic, key points, important examples, and any notable conclusions or recommendations.
- If the transcript appears incomplete, noisy, or ambiguous, mention that limitation.
```

## Implementation Notes

- Use LangChain abstractions for the LLM call and prompt composition.
- Use the DeepSeek LangChain integration or an OpenAI-compatible LangChain chat adapter configured with `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, and optional `DEEPSEEK_BASE_URL`.
- The transcript agent should accept `SummaryRequest` and `QuestionRequest` inputs and return `TranscriptSummary` and `TranscriptAnswer` outputs internally. The CLI can render those outputs as plain text.
- Load API keys and local configuration from `~/.env` before constructing Superdata or DeepSeek clients.
- Initialize MLflow before running fetch, summary, or ask workflows.
- Keep the Superdata integration behind a small transcript fetcher interface so it can be mocked in tests.
- Keep the DeepSeek integration behind the agent module so tests can use fake responses.
- Avoid broad abstractions in V1; the goal is a working direct transcript pipeline.
- Update `readme.md` with install, environment, and CLI usage instructions.
- Add dependencies only as needed and document them.

## Testing Requirements

Add focused tests for:

- Extracting the video ID from `https://www.youtube.com/watch?v=3hk7nO_q0a8`.
- Validating Pydantic transcript, segment, summary request, question request, summary output, and answer output models.
- Storing and loading a transcript from Chroma local persistence.
- Reusing a stored transcript instead of fetching again.
- Cache hit behavior for `summarize` and `ask` without calling Superdata.
- Cache miss behavior that fetches, stores, then continues the requested command.
- `fetch --no-refresh` behavior when a transcript is already cached.
- Building the summary prompt with transcript text.
- Building the Q&A prompt with transcript text and the user question.
- Loading configuration from a mocked env file through `YT_AGENT_ENV_PATH` or an equivalent config loader function parameter.
- MLflow setup uses the configured tracking URI and experiment name.
- CLI workflows log MLflow run metadata without logging API keys.
- Summary and answer outputs are logged as MLflow artifacts.
- CLI command routing for `fetch`, `summarize`, and `ask`.
- Duplicate fetch behavior for the same video ID.

External calls must be mocked in tests:

- Mock Superdata transcript fetching.
- Mock DeepSeek/LangChain LLM calls.

## Non-Negotiable Completion Test

V1 is not complete until the implementation can run one live end-to-end summary test using the first transcript and the DeepSeek API.

Required command:

```bash
uv run python -m src.cli summarize "https://www.youtube.com/watch?v=3hk7nO_q0a8"
```

Required conditions:

- `SUPERDATA_API_KEY` and `DEEPSEEK_API_KEY` are loaded from `~/.env` or from the env file pointed to by `YT_AGENT_ENV_PATH`.
- The transcript is fetched from Superdata or loaded from the local Chroma `transcripts` collection if it was already fetched.
- The summary is generated through the configured DeepSeek model, defaulting to `deepseek-v4-flash`.
- The run is recorded in MLflow under the configured experiment.
- The command exits successfully.
- The output includes:
  - A concise transcript summary.
  - A `Top 3 findings` section.
  - Exactly three numbered findings grounded in the transcript.
- The output must not include claims that are unsupported by the transcript.

If this live test cannot be run because API keys are unavailable, the implementation is not complete. In that case, stop and report the missing prerequisite instead of marking V1 done.

## Manual Verification

After implementation, verify the first transcript end to end:

```bash
uv run python -m src.cli fetch "https://www.youtube.com/watch?v=3hk7nO_q0a8"
uv run python -m src.cli summarize "https://www.youtube.com/watch?v=3hk7nO_q0a8"
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "What is this video about?"
```

The pipeline passes V1 verification when:

- The transcript is fetched from Superdata.
- The transcript is stored locally in Chroma.
- The summary command returns a transcript-grounded summary.
- The summary command creates an MLflow run with command, video, cache, model, status, and output artifact metadata.
- The summary command passes the non-negotiable live DeepSeek completion test with a `Top 3 findings` section.
- The ask command returns a transcript-grounded answer.
- Tests pass without live Superdata or DeepSeek calls.

## Acceptance Criteria

- A Python CLI exists for transcript fetch, Q&A, and summary.
- The first transcript URL can be fetched, stored, summarized, and queried.
- LangChain is used for prompt and LLM orchestration.
- Python dependencies and commands are managed with `uv`.
- `deepseek-v4-flash` is the default configured model, with `deepseek-v4` accepted as a compatibility alias.
- API keys are loaded from `~/.env`.
- Raw transcript text is persisted locally in Chroma.
- Cached transcripts are reused for `summarize` and `ask` without unnecessary Superdata calls.
- `fetch --no-refresh` reuses cached transcripts when present.
- MLflow local tracking captures CLI runs, cache status, LLM calls, and summary/Q&A artifacts.
- The implementation is structured so a later RAG pipeline can reuse stored transcripts.
- Chroma local persistence is configured at `.yt-agent/chroma` by default for raw transcript storage.
- The live DeepSeek summary completion test passes for `https://www.youtube.com/watch?v=3hk7nO_q0a8` and outputs exactly three top findings.
- `readme.md` documents setup, required environment variables, install steps, and example commands.
- Automated tests cover storage, prompt construction, CLI routing, and mocked provider calls.
