## Transcript Agent

Python CLI prototype for fetching a YouTube transcript, caching the raw transcript locally in Chroma, and using a DeepSeek-backed LangChain agent for transcript Q&A and summaries.

### Goals

1. Read YouTube transcripts.
2. Agent 1: LLM with transcript can: (a) Q&A (b) summary.
3. RAG pipelines for comparison.
4. Agent 2: RAG agent.

### Setup

This project uses `uv`.

```bash
uv sync
```

Create `~/.env` with:

```text
SUPADATA_API_KEY=<Supadata API key>
# SUPERDATA_API_KEY is also supported for compatibility with the V1 spec.
DEEPSEEK_API_KEY=<DeepSeek API key>
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
YT_AGENT_CHROMA_PATH=.yt-agent/chroma
MLFLOW_TRACKING_URI=file:.yt-agent/mlruns
MLFLOW_EXPERIMENT_NAME=yt-agent-v1
YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=false
```

`SUPADATA_API_KEY` is used with the Supadata transcript API at `https://api.supadata.ai/v1/transcript`. `SUPERDATA_API_KEY` is also supported for compatibility with the V1 spec wording.

For tests or local debugging, set `YT_AGENT_ENV_PATH` in the process environment to load a different env file.

If `DEEPSEEK_MODEL=deepseek-v4` is set, the CLI maps it to `deepseek-v4-flash`, because the DeepSeek API currently requires the concrete `deepseek-v4-flash` or `deepseek-v4-pro` model ID.

### Quick CLI Commands

Run these from the project root after `uv sync` and `~/.env` setup.

Summarize the first test transcript:

```bash
uv run python -m src.cli summarize "https://www.youtube.com/watch?v=3hk7nO_q0a8"
```

Ask a question about the first test transcript:

```bash
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "What is this video about?"
```

Fetch or refresh the cached transcript:

```bash
uv run python -m src.cli fetch "https://www.youtube.com/watch?v=3hk7nO_q0a8"
```

Use the cached transcript without refreshing:

```bash
uv run python -m src.cli fetch "https://www.youtube.com/watch?v=3hk7nO_q0a8" --no-refresh
```

### Local Storage

Raw transcripts are stored in Chroma under:

```text
.yt-agent/chroma
```

V1 stores only full raw transcript documents in the `transcripts` collection. It does not create chunks, embeddings, vector search, or RAG retrieval yet.

The CLI passes only the video URL and user task into the agent. The agent backend pulls transcript context from Chroma through a context provider, so a future RAG pipeline can replace that provider with retrieved chunks without changing the CLI command shape.

View saved transcript IDs and metadata:

```bash
uv run python - <<'PY'
import chromadb

client = chromadb.PersistentClient(path=".yt-agent/chroma")
collection = client.get_or_create_collection("transcripts")
result = collection.get(include=["metadatas"])

for transcript_id, metadata in zip(result["ids"], result["metadatas"]):
    print(transcript_id)
    print(metadata)
    print()
PY
```

View the first 1,000 characters of a saved transcript:

```bash
uv run python - <<'PY'
import chromadb

client = chromadb.PersistentClient(path=".yt-agent/chroma")
collection = client.get_or_create_collection("transcripts")
result = collection.get(include=["documents", "metadatas"])

for transcript_id, document, metadata in zip(
    result["ids"], result["documents"], result["metadatas"]
):
    print(f"ID: {transcript_id}")
    print(f"URL: {metadata.get('url')}")
    print(document[:1000])
    print()
PY
```

### Observability

MLflow local tracking is written to:

```text
.yt-agent/mlruns
```

Each CLI command creates a run with command metadata, cache status, transcript metadata, and summary/Q&A artifacts. Full transcript artifacts are disabled by default unless `YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=true`.

### Tests

```bash
uv run pytest
```

External Superdata/Supadata and DeepSeek calls are mocked in automated tests.

### Future

* Multiple transcripts in one agent.
* Track trends across transcripts in the same field.
* Build evaluation set and score accuracy.
* Build optimization of LLM system prompt to improve accuracy.
* Add RAG chunking and retrieval using the existing Chroma persistence path.

### Agent work documents

Store markdown plans, PRPs, and specs for coding agents in `agent-work/`.

```text
agent-work/
  plans/
  prps/
  specs/
  templates/
  archive/
```

Use the templates in `agent-work/templates/` when creating new implementation documents.
