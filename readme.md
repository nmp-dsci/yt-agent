## YouTube Transcript RAG Demo

CLI prototype that demonstrates the value of RAG over full-transcript prompting for YouTube transcript Q&A.

The main demo compares one question across three transcript input types:

- `raw_single`: full raw transcript for one video.
- `rag_single`: top 10 retrieved chunks for that same video.
- `rag_all`: top 10 retrieved chunks across all indexed videos.

The demo writes `dashboard/evaluation.html` with answers, token estimates, pairwise answer similarity, and retrieved chunks. The target outcome is similar answer quality with roughly 80%+ fewer prompt tokens for RAG.

### Setup

This project uses `uv`.

```bash
uv sync
```

The dashboard Chunk Space tab uses `scikit-learn` for deterministic PCA projection of stored chunk embeddings; it is installed by `uv sync`.

Create `~/.env`:

```text
SUPADATA_API_KEY=<Supadata API key>
# SUPERDATA_API_KEY is also supported for compatibility with earlier project wording.
DEEPSEEK_API_KEY=<DeepSeek API key>
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
YT_AGENT_CHROMA_PATH=.yt-agent/chroma
YT_AGENT_RAW_TRANSCRIPT_COLLECTION=raw_transcripts
YT_AGENT_CHUNK_COLLECTION=transcript_chunks
YT_AGENT_TRANSCRIPT_SUMMARY_COLLECTION=transcript_summaries
YT_AGENT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
YT_AGENT_RAG_TOP_K=10
YT_AGENT_TRANSCRIPT_FILTER_TOP_K=5
YT_AGENT_TRANSCRIPT_FILTER_MIN_SCORE=0.25
YT_AGENT_RAG_RECURSIVE_DEFAULT=false
YT_AGENT_RAG_MAX_DEPTH=1
YT_AGENT_RAG_MAX_FOLLOWUPS=3
YT_AGENT_RAG_FOLLOWUP_TOP_K=
YT_AGENT_RAG_NOVELTY_MIN_CHUNKS=2
YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS=
YT_AGENT_CHUNK_TARGET_CHARS=1200
YT_AGENT_CHUNK_OVERLAP_CHARS=150
YT_AGENT_DISCOVERY_CACHE_TTL_HOURS=24
SUPADATA_TIMEOUT_SECONDS=120
SUPADATA_POLL_INTERVAL_SECONDS=2
SUPADATA_MAX_POLL_SECONDS=600
MLFLOW_TRACKING_URI=file:.yt-agent/mlruns
MLFLOW_EXPERIMENT_NAME=yt-agent-v1
YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=false
```

`SUPADATA_API_KEY` is used with the Supadata transcript API. DeepSeek is called through the OpenAI-compatible LangChain client.

Supadata can return async jobs for longer videos. `SUPADATA_MAX_POLL_SECONDS=600` lets indexing wait up to 10 minutes for those jobs before timing out.

Recursion env vars are used only when recursive mode is effectively on via `--recursive` or `YT_AGENT_RAG_RECURSIVE_DEFAULT=true`. Empty `YT_AGENT_RAG_FOLLOWUP_TOP_K` defaults follow-up retrieval to `YT_AGENT_RAG_TOP_K`; empty `YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS` defaults to `max_depth * max_followups`.

### Command Sequence

Run commands from the project root after `uv sync` and env setup.

Set a reusable URL and question:

```bash
url="https://www.youtube.com/watch?v=3hk7nO_q0a8"
question="what does this video say for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount"
```

#### 1. Optional transcript fetch

Fetch and cache a transcript without building the RAG index:

```bash
uv run python -m src.cli fetch "$url"
```

Fetch raw timestamped transcript segments:

```bash
uv run python -m src.cli fetch-raw "$url"
```

Use `--no-refresh` with either command to read from cache only when available:

```bash
uv run python -m src.cli fetch "$url" --no-refresh
uv run python -m src.cli fetch-raw "$url" --no-refresh
```

#### 2. Index transcripts

Index one YouTube transcript for RAG:

```bash
uv run python -m src.cli index-rag "$url"
```

`index-rag` stores raw transcript segments, chunk embeddings, an LLM-generated transcript summary, and a transcript-level summary embedding used for optional summary-first filtering. Regenerate the summary and summary embedding with:

```bash
uv run python -m src.cli index-rag "$url" --refresh-summary
```

Force a full transcript refresh and rebuild chunks:

```bash
uv run python -m src.cli index-rag "$url" --refresh
```

Bulk-index the most recent videos from a YouTube channel via Supadata discovery:

```bash
uv run python -m src.cli bulk-index channel \
  --channel "https://www.youtube.com/@aiDotEngineer" \
  --latest 5 \
  --label "ai-engineer-latest-5"
```

Preview a channel discovery run without indexing:

```bash
uv run python -m src.cli bulk-index channel \
  --channel "https://www.youtube.com/@aiDotEngineer" \
  --latest 5 \
  --dry-run
```

Bulk-index every video a channel published in a date window:

```bash
uv run python -m src.cli bulk-index channel \
  --channel "@somechannel" \
  --since 2026-01-01 \
  --until 2026-05-17 \
  --max-results 50 \
  --label "somechannel-q1-q2"
```

Bulk-index the top N YouTube search results for a query:

```bash
uv run python -m src.cli bulk-index search \
  --query "australian capital gains tax reform" \
  --top-n 10 \
  --label "cgt-top10"
  --dry-run
```

Common `bulk-index` flags:

- `--dry-run` — run discovery only, do not index.
- `--skip-existing` / `--no-skip-existing` — default skips videos already fully indexed in both `raw_transcripts` and `transcript_chunks`.
- `--refresh-summary` — regenerate transcript summaries even when raw transcripts and chunks are reused.
- `--concurrency 1` — only sequential ingestion is currently supported.
- `--no-discovery-cache` — bypass the 24h discovery cache for this run.

Each `bulk-index` run writes one JSON record under `.yt-agent/ingestion_runs/` capturing per-candidate outcomes. The Ingestion Runs tab in `rag_pipeline.html` reads these records when any exist.

#### 3. Refresh the RAG dashboard

Render the local RAG pipeline review dashboard:

```bash
uv run python -m src.dashboard.rag_pipeline --output dashboard/rag_pipeline.html
```

Force-refit the chunk-space PCA projection:

```bash
uv run python -m src.dashboard.rag_pipeline \
  --output dashboard/rag_pipeline.html \
  --refresh-projection
```

Override the canonical question used in the Chunk Space tab:

```bash
uv run python -m src.dashboard.rag_pipeline \
  --output dashboard/rag_pipeline.html \
  --question "$question"
```

Open:

```text
dashboard/rag_pipeline.html
```

#### 4. Ask questions

Full transcript (raw): sends the whole single-video transcript to the LLM.

```bash
uv run python -m src.cli ask "$url" "$question" --context raw
```

Single-transcript RAG: retrieves chunks from one video before calling the LLM.

```bash
uv run python -m src.cli ask "$url" "$question" --context rag --top-k 10
```

Multi-transcript RAG (single-hop): retrieves chunks across every indexed video, or restricts the same agent to one URL with `--url`.

```bash
question="how do ai engineers leveage claude to fully develop features and only set & review"

uv run python -m src.cli rag-ask "$question" --top-k 20
uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
```

Multi-transcript RAG (single-hop, summary-filtered): first selects relevant transcript summaries, then retrieves chunks only from those videos.

```bash
uv run python -m src.cli rag-ask "$question" --filter-transcripts --top-k 20
uv run python -m src.cli rag-ask "$question" --filter-transcripts \
  --transcript-filter-top-k 8 --transcript-filter-min-score 0.3 --top-k 20
```

Multi-transcript RAG (single-hop, show follow-ups): still performs one retrieval and one LLM call, but prints the model's proposed follow-up retrieval queries.

```bash
uv run python -m src.cli rag-ask "$question" --show-followups
uv run python -m src.cli rag-ask "$question" --url "$url" --show-followups
uv run python -m src.cli rag-ask "$question" --filter-transcripts --show-followups
```

Multi-transcript RAG (recursive): acts on follow-up queries with bounded fan-out retrieval, then runs a final synthesis call.

```bash
uv run python -m src.cli rag-ask "$question" --recursive
uv run python -m src.cli rag-ask "$question" --recursive --url "$url"
uv run python -m src.cli rag-ask "$question" --recursive --filter-transcripts
uv run python -m src.cli rag-ask "$question" --recursive \
  --max-depth 1 --max-followups 4 --top-k 15 --followup-top-k 10
uv run python -m src.cli rag-ask "$question" --recursive \
  --max-total-followups 6 --novelty-min-chunks 3
uv run python -m src.cli rag-ask "$question" --recursive --print-trace
uv run python -m src.cli rag-ask "$question" --recursive --filter-transcripts \
  --url "$url" --max-followups 3 --print-trace
```

With `YT_AGENT_RAG_RECURSIVE_DEFAULT=true`, `rag-ask "$question"` runs recursively by default. Use `--no-recursive` to force the single-hop path.

Recursive RAG flags:

- `--recursive` — enable recursive multi-hop RAG; default is off unless `YT_AGENT_RAG_RECURSIVE_DEFAULT=true`.
- `--no-recursive` — disable recursive RAG even when the env default is on.
- `--max-depth N` — default `1`; S6 implements `0` and `1`, where `0` collapses to single-hop.
- `--max-followups N` — default `3`; maximum follow-up queries selected from the first pass.
- `--followup-top-k N` — default is `--top-k`; chunks retrieved for each follow-up query.
- `--novelty-min-chunks N` — default `2`; minimum new chunks required to include a follow-up in synthesis.
- `--max-total-followups N` — default `max_depth * max_followups`; hard cap on fan-out retrievals.
- `--show-followups` — print proposed follow-up queries in single-hop mode.
- `--print-trace` — print per-follow-up chunk previews in recursive mode.

Summarize one transcript:

```bash
uv run python -m src.cli summarize "$url"
```

#### 5. Compare and evaluate

Compare full-transcript prompting against single-transcript RAG in the terminal:

```bash
uv run python -m src.cli compare-context "$url" "$question" --top-k 10
```

Generate the HTML evaluation report:

```bash
uv run python -m src.evals.evaluation \
  --url "$url" \
  --question "$question" \
  --output dashboard/evaluation.html \
  --json-output dashboard/evaluation.json
```

Open:

```text
dashboard/evaluation.html
```

The report compares six variants in one run:

| Variant | Description |
|---|---|
| `raw_single` | Full transcript sent to the LLM (token baseline). |
| `rag_single` | Top-K chunks from the selected video only. |
| `rag_all` | Top-K chunks across all indexed videos (single-hop). |
| `rag_all_filtered` | Summary-filtered transcripts, then top-K chunks (single-hop). |
| `rag_recursive` | Multi-hop: first-pass + fan-out retrieval + synthesis (all transcripts). |
| `rag_recursive_filtered` | Multi-hop with summary filtering applied at every hop. |

The report shows:

- Answers for all six variants.
- Prompt token estimates and retrieved chunk counts for each.
- LLM call count and recursion terminated reason for recursive variants.
- Pairwise embedding similarity between answers.
- Expandable retrieved chunks with source URL and timestamp links.
- Recursion trace (stages, proposed/executed follow-ups, subtopic drill-downs) for recursive variants.

### Architecture

```text
src/
  transcripts/   # YouTube URL parsing, Supadata fetching, transcript models/storage
  rag/           # Raw segment storage, chunking, embeddings, retrieval, references
  agents/        # Full-transcript agent and RAG agent with optional recursive multi-hop retrieval
  evals/         # Demo/evaluation scripts and HTML report generation
  dashboard/     # Local HTML dashboards for reviewing indexed RAG state
tests/
```

Canonical storage:

- `raw_transcripts`: timestamped Supadata segment stream.
- `transcript_chunks`: embedded timestamped transcript chunks.
- `transcript_summaries`: embedded LLM transcript summaries for optional transcript-level filtering.

The legacy `transcripts` collection may exist from earlier prototype work, but current raw and RAG paths use `raw_transcripts` and `transcript_chunks`.

### Agent Architecture

There are two agent paths:

- `TranscriptAgent`: supports full raw transcript prompting and single-video RAG comparison.
- `RagTranscriptAgent`: RAG agent that can search all indexed transcript chunks, filter to one URL, and optionally run recursive multi-hop retrieval.

`RagTranscriptAgent` uses a unified first-pass LLM contract in both modes: the prompt always asks for an answer with references plus proposed subtopics and follow-up retrieval queries. Single-hop mode returns those follow-ups only when requested by `--show-followups`; recursive mode acts on them with extra retrieval and a final synthesis call.

Indexing flow:

```text
YouTube URL
  -> extract video_id
  -> Supadata transcript fetch with text=false
  -> timestamped segments
  -> raw_transcripts collection
  -> segment-aware chunking
  -> local embedding model
  -> transcript_chunks collection
```

Raw single-transcript Q&A flow:

```text
User question + URL
  -> src.cli ask --context raw
  -> TranscriptAgent
  -> RawTranscriptContextProvider
  -> raw_transcripts lookup by video_id
  -> join every segment into full transcript context
  -> DeepSeek LLM
  -> answer
```

Raw mode sends the whole transcript to the LLM. It is the quality baseline, but it uses the most prompt tokens.

Single-transcript RAG Q&A flow:

```text
User question + URL
  -> src.cli ask --context rag --top-k 10
  -> TranscriptAgent
  -> RagTranscriptContextProvider
  -> embed user question
  -> transcript_chunks vector search where video_id == URL video_id
  -> format top 10 chunks with timestamps
  -> DeepSeek LLM
  -> answer
```

Single-transcript RAG only sends the retrieved chunks to the LLM. This is the direct token-reduction comparison against `raw_single`.

All-transcript RAG Q&A flow:

```text
User question
  -> src.cli rag-ask --top-k 10
  -> RagTranscriptAgent
  -> MultiTranscriptRagContextProvider
  -> embed user question
  -> transcript_chunks vector search across all indexed videos
  -> format top 10 chunks with video URLs and timestamp links
  -> DeepSeek LLM
  -> answer with source references + proposed follow-ups
```

All-transcript RAG is the demo path for asking across the indexed corpus. It can be filtered back to one transcript with:

```bash
uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
```

Optional transcript-summary filtered RAG flow:

```text
User question
  -> src.cli rag-ask --filter-transcripts
  -> embed user question
  -> vector search transcript_summaries
  -> keep selected transcript video IDs
  -> vector search transcript_chunks restricted to those video IDs
  -> DeepSeek LLM
  -> answer with source references
```

Recursive RAG flow:

```text
User question
  -> src.cli rag-ask --recursive
  -> retrieve initial chunks with MultiTranscriptRagContextProvider
  -> first-pass DeepSeek call: answer + references + follow-up subtopics
  -> for each selected follow-up query, retrieve more chunks through the same provider
  -> drop duplicate or low-novelty follow-up evidence
  -> final DeepSeek synthesis call
  -> layered answer + combined references + recursion trace
```

Recursive mode inherits `--url` and `--filter-transcripts` because every hop reuses the same context provider and request filters.

Evaluation flow:

```text
src.evals.evaluation
  -> run raw_single
  -> run rag_single
  -> run rag_all
  -> estimate prompt tokens from context length
  -> embed the three answers
  -> compute pairwise cosine similarity
  -> write dashboard/evaluation.html
```

The evaluation proves the demo claim when RAG answers remain similar to raw answers while using substantially fewer prompt tokens.

### Dashboard Outputs

Generated review artifacts live under:

```text
dashboard/
  evaluation.html
  evaluation.json
  rag_pipeline.html
  chunk_space/
    projection.json     # PCA projection (chunk coords, components, mean) — committed
    question.json       # canonical question + nearest chunks — committed
```

`evaluation.html` compares answers for a question. `rag_pipeline.html` is a tabbed dashboard that reviews indexed transcripts, summaries, summary encodings, chunk inventory, ingestion history when run records exist, and the chunk-embedding scatter plot. The `chunk_space/` artifacts are committed so a fresh clone renders the Chunk Space tab without re-running ingestion.

### Observability

MLflow local tracking is written to:

```text
.yt-agent/mlruns
```

Each CLI command creates a run with command metadata, cache status, transcript metadata, and answer artifacts. Full transcript artifacts are disabled by default unless `YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=true`.

### Tests

```bash
uv run pytest
```

External Supadata, DeepSeek/LangChain, and embedding calls are mocked in automated tests where appropriate.

### Agent Work

Implementation specs and handoff notes live in `agent-work/`.

Generated dashboard outputs should live in `dashboard/`, not `agent-work/`.
