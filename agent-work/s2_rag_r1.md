# Spec: S2 RAG R1 Backend Pipeline

Status: ready
Date: 2026-05-14

## Summary

Add a backend RAG pipeline that builds on the existing Chroma persistence from `s1_agent_v1.md` and the current implementation. The current code is the source of truth.

V1 currently stores flattened raw transcripts in Chroma collection `transcripts` and uses `RawTranscriptContextProvider` to pass raw transcript context to the LangChain agent. S2 decommissions that flattened `transcripts` collection as the canonical source. The new canonical source is a Chroma collection populated from Supadata `text=false` responses, carrying the original segment strings and timestamps. Raw mode joins those segment strings into a full transcript for the LLM; RAG mode retrieves the top 10 embedded chunks built from the same segment source.

The goal is to compare two context modes through the same agent:

- `raw`: join all cached segment strings into a full transcript context.
- `rag`: retrieve top 10 timestamped transcript chunks and provide those as context.

S2 applies context-mode comparison to Q&A first. Summary remains raw-mode only in this release because the main evaluation target is answer accuracy and token consumption for transcript Q&A. RAG summary can be added later after Q&A retrieval behavior is proven.

Summary scope:

- Existing summarize behavior must continue to work using raw context from `raw_transcripts`.
- S2 does not implement or evaluate `summarize --context rag`.
- RAG summary is explicitly future work.

## Source Of Truth Review

Current implementation to build on:

- `src/transcripts/fetcher.py`: calls Supadata `/v1/transcript` with `text=false`, normalizes `content[]` into `TranscriptSegment`.
- `src/transcripts/models.py`: defines `Transcript` and `TranscriptSegment`; segments already include `start_seconds` and `end_seconds`.
- `src/transcripts/storage.py`: currently owns Chroma `transcripts` collection for flattened raw transcript storage. S2 should replace this canonical storage path with timestamped segment storage.
- `src/agents/context.py`: defines `TranscriptContextProvider`; `RawTranscriptContextProvider` is the replacement point for future RAG.
- `src/agents/transcript_agent.py`: accepts `SummaryRequest` / `QuestionRequest`, pulls backend context through the provider, then sends context to LangChain.
- `src/cli.py`: wires settings, Chroma store, fetcher, context provider, agent, and MLflow logging.

Important implementation constraints:

- Update the agent where needed to complete the context-mode design. Keep the user-facing CLI request shape and final answer models stable, but do not force RAG context to masquerade as a raw transcript.
- Add explicit context payload fields so raw and RAG providers can supply different context text while preserving shared transcript/video metadata.
- Do not change the user-facing ask command shape more than necessary. The command should remain URL + question, with optional mode flags for comparison.
- Establish one canonical ingestion path that writes `raw_transcripts`; raw mode and RAG mode must both read from that same source.

## Supadata Timestamp Data

Supadata transcript responses with `text=false` return timestamped segment objects in `content[]`:

```json
{
  "text": "Transcript segment text",
  "offset": 8150,
  "duration": 1200,
  "lang": "en"
}
```

`offset` is the start time in milliseconds and `duration` is duration in milliseconds. The fetcher must preserve both the original millisecond fields and computed second fields:

- `TranscriptSegment.offset_ms`
- `TranscriptSegment.duration_ms`
- `TranscriptSegment.start_seconds`
- `TranscriptSegment.end_seconds`

Use those segment timestamps when building chunks so answers can reference where in the video the retrieved evidence came from.

Sources:

- Supadata transcript endpoint: https://docs.supadata.ai/api-reference/endpoint/transcript/transcript
- Supadata transcript guide: https://supadata.ai/documentation/get-transcript

## Non-Goals

- Do not add a new database. Use the existing Chroma persistence path.
- Do not remove the raw transcript mode. Update it so raw context is derived by joining cached timestamped segments, not by reading a flattened raw transcript document.
- Do not build a web UI.
- Do not support multi-video RAG yet.
- Do not store full raw provider payloads in Chroma metadata. Store timestamped provider segments in a separate Chroma collection document body instead.

## Proposed Project Structure

Add these files:

```text
src/
  rag/
    __init__.py              # Marks RAG utilities as importable
    models.py                # Pydantic chunk, retrieval, and evaluation models
    chunking.py              # Timestamp-aware transcript chunking
    embeddings.py            # SentenceTransformer embedding setup
    storage.py               # Chroma raw_transcripts and transcript_chunks collections
    context.py               # RAGTranscriptContextProvider for agent context
    eval.py                  # Raw-vs-RAG answer similarity and token comparison
tests/
  rag/
    __init__.py
    test_models.py           # RAG model validation tests
    test_chunking.py         # Timestamp-preserving chunking tests
    test_storage.py          # Chroma chunk upsert/query tests with temp path
    test_context.py          # RAG context provider tests
    test_eval.py             # Similarity and token comparison tests
```

Update these files:

```text
src/config.py                # Add embedding model and RAG defaults
src/cli.py                   # Add chunk/index command and ask context mode flags
src/agents/context.py        # Add explicit context DTO fields for raw and RAG modes
src/agents/transcript_agent.py # Use explicit context text from raw or RAG provider
src/observability.py         # Log context mode, top_k, retrieved chunk metadata, token estimates
readme.md                    # Add RAG setup, indexing, ask, evaluation commands
pyproject.toml               # Add RAG dependencies and package discovery for src.rag
```

## Dependencies

Use `uv`.

Add:

```bash
uv add sentence-transformers langchain-huggingface
```

Use:

- `sentence-transformers` / `langchain-huggingface` for local embeddings.
- Existing `chromadb` for persistent vector storage.

Packaging requirement:

- Current packaging must discover future subpackages such as `src.rag`. Use setuptools package discovery, for example `include = ["src*"]`, rather than a fixed package list that would omit `src.rag`.

Do not add `langchain-text-splitters` in S2 unless implementation proves it is needed for a timestampless fallback. The primary chunking path is custom segment grouping over timestamped Supadata segments.

Recommended default embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This is small, common, fast enough locally, and good enough for a first RAG comparison. Make it configurable.

## Configuration

Add defaults:

```text
YT_AGENT_RAW_TRANSCRIPT_COLLECTION=raw_transcripts
YT_AGENT_CHUNK_COLLECTION=transcript_chunks
YT_AGENT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
YT_AGENT_RAG_TOP_K=10
YT_AGENT_CHUNK_TARGET_CHARS=1200
YT_AGENT_CHUNK_OVERLAP_CHARS=150
```

Existing:

```text
YT_AGENT_CHROMA_PATH=.yt-agent/chroma
```

## Data Models

Add `src/rag/models.py`.

```python
from pydantic import BaseModel, Field, HttpUrl


class RawTranscriptSegment(BaseModel):
    text: str
    offset_ms: int | None = None
    duration_ms: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    language: str | None = None


class RawTranscriptDocument(BaseModel):
    transcript_id: str
    video_id: str
    source_url: HttpUrl
    provider: str = "supadata"
    title: str | None = None
    language: str | None = None
    segments: list[RawTranscriptSegment] = Field(default_factory=list)
    fetched_at: str
    source_collection: str = "raw_transcripts"


class TranscriptChunk(BaseModel):
    transcript_id: str
    video_id: str
    source_url: HttpUrl
    chunk_index: int
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    start_segment_index: int | None = None
    end_segment_index: int | None = None
    segment_count: int = 0


class RetrievedChunk(TranscriptChunk):
    score: float | None = None


class RagContextResult(BaseModel):
    video_id: str
    source_url: HttpUrl
    query: str
    top_k: int
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class ContextComparisonResult(BaseModel):
    question: str
    raw_answer: str
    rag_answer: str
    semantic_similarity: float
    raw_prompt_tokens_estimate: int
    rag_prompt_tokens_estimate: int
    token_savings_percent: float
```

Rules:

- `transcript_id` should match the raw transcript document id: `raw_transcript:{video_id}`.
- `RawTranscriptDocument.segments` are serialized as JSON and stored as the Chroma document body in `raw_transcripts`, not as metadata.
- Raw LLM context is built by joining `segment.text` in order.
- Keep `offset_ms` and `duration_ms` when Supadata provides them; compute `start_seconds` and `end_seconds` from those values.
- `chunk_index` must be stable for repeatable upserts.
- `start_seconds` is the first segment start in the chunk.
- `end_seconds` is the last segment end in the chunk.
- `start_segment_index` and `end_segment_index` should identify the source segment range inside `RawTranscriptDocument.segments`.
- Chroma metadata must stay scalar-only.

## Chroma Storage Design

Legacy collection to decommission as canonical storage:

```text
transcripts
```

S2 should stop relying on `transcripts` for raw-mode context. It can remain on disk for backward compatibility or migration, but new code should read/write the canonical segment collection below.

Migration rule:

- Existing V1 cached rows in `transcripts` may be ignored, or used only as a temporary fallback if Supadata cannot be called.
- The preferred migration is to re-fetch the video from Supadata with `text=false` and populate `raw_transcripts`.
- Do not create new canonical flattened `raw_text` records after S2.

Add canonical Chroma collections:

```text
raw_transcripts
transcript_chunks
```

Store each raw timestamped transcript as:

- `id`: `raw_transcript:{video_id}`
- `document`: JSON string containing ordered segment records from Supadata `text=false`
- metadata:
  - `transcript_id`
  - `video_id`
  - `source_url`
  - `source_collection`
  - `provider`
  - `title`
  - `language`
  - `fetched_at`
  - `segment_count`

Document shape:

```json
{
  "segments": [
    {
      "text": "Transcript segment text",
      "offset_ms": 8150,
      "duration_ms": 1200,
      "start_seconds": 8.15,
      "end_seconds": 9.35,
      "language": "en"
    }
  ]
}
```

Store each chunk as:

- `id`: `chunk:{video_id}:{chunk_index}`
- `document`: chunk text
- `embedding`: generated by sentence-transformer embedding function
- metadata:
  - `transcript_id`
  - `video_id`
  - `source_url`
  - `source_collection`
  - `chunk_index`
  - `start_seconds`
  - `end_seconds`
  - `start_segment_index`
  - `end_segment_index`
  - `segment_count`

Chroma does not use the word “table”; in this project “separate table” means a separate Chroma collection.

Storage ownership:

- `raw_transcripts`: canonical Supadata `text=false` segment stream with text and timestamps.
- `transcript_chunks`: embedded timestamped chunks for RAG retrieval.
- `transcripts`: legacy flattened V1 collection, no longer canonical after S2.

Do not put segment arrays or provider JSON in Chroma metadata. Chroma metadata stays scalar-only; ordered segment JSON belongs in the `raw_transcripts` document body.

## Chunking Strategy

Use timestamp-aware transcript chunking designed around Supadata transcript segments.

Recommended approach:

1. Use `raw_transcripts` as the source for chunking because it preserves the original Supadata `content[]` text and timestamp shape.
2. If `raw_transcripts` is missing for a video, fetch Supadata with `text=false` and store `raw_transcripts` first.
3. Do not chunk from legacy flattened `transcripts` except as an explicit migration fallback. If used, timestamps are unavailable and chunks must mark timestamps as unavailable.
4. Group consecutive segments into chunks near `YT_AGENT_CHUNK_TARGET_CHARS`.
5. Add a small overlap using preceding segment text, targeting `YT_AGENT_CHUNK_OVERLAP_CHARS`.
6. Preserve metadata:
   - chunk start = first included segment `offset` converted to seconds, or `start_seconds`
   - chunk end = last included segment offset + duration converted to seconds, or `end_seconds`
   - start/end segment indexes from the raw transcript segment list
   - segment count = number of source segments
7. Only use LangChain text splitting as a fallback when segment timestamps are missing.

Source priority:

```text
raw_transcripts Supadata text=false segments -> fetch text=false and store -> legacy transcripts fallback only for migration
```

Why this strategy:

- Video transcripts are naturally ordered, timestamped segment streams.
- Grouping adjacent segments preserves chronology and lets each retrieved chunk link back to a video time range.
- Moderate chunk size plus overlap is a common RAG pattern for transcript/document retrieval.

Do not split in a way that loses timestamps.

## Embeddings

Use LangChain-compatible sentence transformer embeddings.

Recommended implementation:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
)
```

Use native `chromadb` for storage, matching the existing codebase, and compute embeddings explicitly through a small embedding adapter in `src/rag/embeddings.py`.

Reason:

- Current storage code already uses native `chromadb`.
- Keeping native Chroma avoids adding a second Chroma abstraction layer.
- Embedding generation stays easy to mock in tests.

Embedding adapter requirements:

- Define a small protocol such as `EmbeddingModel` with `embed_documents(texts: list[str]) -> list[list[float]]` and `embed_query(text: str) -> list[float]`.
- Implement that protocol with `HuggingFaceEmbeddings`.
- Inject the embedding adapter into `TranscriptChunkStore` instead of constructing it inside query methods.
- In tests, use a deterministic fake embedding adapter.

## CLI Commands

Add indexing command:

```bash
uv run python -m src.cli index-rag "https://www.youtube.com/watch?v=3hk7nO_q0a8"
```

Behavior:

- Ensure canonical raw transcript exists through the S2 raw transcript store.
- Fetch Supadata with `text=false` when `raw_transcripts` is missing or refresh is requested.
- Store or refresh Supadata segment data in `raw_transcripts`.
- Prefer `raw_transcripts` as the chunking input.
- Build timestamp-preserving chunks.
- Embed chunks.
- Upsert chunks into `transcript_chunks`.
- Print raw transcript collection, chunk collection, chunk count, and Chroma path.

Update ask command with context mode:

```bash
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "question" --context raw
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "question" --context rag --top-k 10
```

Defaults:

- `--context raw` for backward compatibility.
- `--top-k 10` for RAG.

RAG ask missing-index behavior:

- `ask --context rag` must auto-index the requested video if no chunks exist in `transcript_chunks`.
- Auto-indexing should use the same indexing path as `index-rag`: ensure `raw_transcripts`, chunk timestamped segments, embed chunks, and upsert into `transcript_chunks`.
- If auto-indexing fails, return a clear CLI error that includes the failed stage.
- If chunks exist, do not re-index unless a future refresh flag is added.

Add raw transcript ingestion command:

```bash
uv run python -m src.cli fetch-raw "https://www.youtube.com/watch?v=3hk7nO_q0a8"
```

Behavior:

- Fetch Supadata with `text=false`.
- Store ordered timestamped segment JSON in `raw_transcripts`.
- Do not write new canonical records to legacy `transcripts`.

Update existing fetch behavior:

- `fetch` becomes an alias for `fetch-raw`.
- `fetch` and `fetch --no-refresh` must read/write `raw_transcripts`.
- `fetch` must not write new records to legacy `transcripts`.
- Legacy `transcripts` can remain on disk but is not used by raw or RAG context.

Add comparison command:

```bash
uv run python -m src.cli compare-context "https://www.youtube.com/watch?v=3hk7nO_q0a8" "question" --top-k 10
```

Behavior:

- Run the same question through raw mode.
- Run the same question through RAG mode.
- Compare semantic similarity.
- Compare token consumption estimates.
- Log both runs and comparison metrics to MLflow.

## Agent Update

Update `TranscriptAgent` enough to support explicit raw and RAG context payloads cleanly.

Required context-provider shape:

- `RawTranscriptContextProvider`: returns full raw transcript.
- Add `RagTranscriptContextProvider`: retrieves top-k chunks and returns a context string built from chunks.
- `TranscriptContext` should include:
  - `transcript`: transcript/video metadata for logging and response defaults.
  - `context_text`: exact text sent to the LLM context prompt.
  - `cache_status`: raw transcript cache status.
  - `context_mode`: `raw` or `rag`.
  - `retrieved_chunks`: empty for raw mode, populated for RAG mode.
  - `top_k`: populated for RAG mode.

The LangChain user message should still contain only the user’s question or summary request. The backend provider supplies context.

Agent behavior:

- `TranscriptAgent.answer` must call the provider and pass `context.context_text` to the LLM.
- `TranscriptAgent.summarize` must use raw context only in S2.
- The context-size check must apply to `context.context_text`, not `context.transcript.raw_text`.
- Do not fake retrieved RAG chunks by storing them in `Transcript.raw_text`; keep raw transcript text and prompt context separate.

Update raw mode:

- `RawTranscriptContextProvider` should read `raw_transcripts`.
- It should build raw LLM context by joining every segment `text` in order.
- It should not depend on the legacy flattened `transcripts` document.

Storage API requirements:

- Add `RawTranscriptStore.upsert_raw_document(document: RawTranscriptDocument)`.
- Add `RawTranscriptStore.get_raw_document(video_id: str) -> RawTranscriptDocument | None`.
- Add `RawTranscriptStore.ensure_raw_document(source_url: str, refresh: bool = False) -> RawTranscriptDocument`.
- Add `RawTranscriptStore.join_raw_text(video_id: str) -> str`.
- Add `TranscriptChunkStore(embedding_model: EmbeddingModel, ...)`.
- Add `TranscriptChunkStore.upsert_chunks(chunks: list[TranscriptChunk])`, computing document embeddings through the injected embedding model.
- Add `TranscriptChunkStore.has_chunks(video_id: str) -> bool`.
- Add `TranscriptChunkStore.query(video_id: str, query: str, top_k: int) -> list[RetrievedChunk]`, computing the query embedding through the injected embedding model.

Update RAG mode:

- `RagTranscriptContextProvider` should query `transcript_chunks`.
- It should return top 10 chunks by default.

For RAG context, format retrieved chunks like:

```text
[1] 00:12:34-00:13:02
chunk text...

[2] 00:18:10-00:18:45
chunk text...
```

Add a shared timestamp formatter:

- `format_timestamp(seconds: float | None) -> str`
- Use `HH:MM:SS` for values >= one hour.
- Use `MM:SS` for values under one hour.
- Return `unknown` when timestamp is unavailable.

Update the system prompt to instruct the agent:

- Use only provided context.
- When context includes timestamp labels, include relevant timestamp references in answers.
- If retrieved context is insufficient, say so.

## Evaluation

Use this required test question:

```text
what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount
```

Run both:

```bash
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount" --context raw
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount" --context rag --top-k 10
```

Then run:

```bash
uv run python -m src.cli compare-context "https://www.youtube.com/watch?v=3hk7nO_q0a8" "what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount" --top-k 10
```

Similarity metric:

- Use cosine similarity between sentence-transformer embeddings of the raw answer and RAG answer.
- Recommended pass threshold: `semantic_similarity >= 0.80`.

S2 comparison intentionally uses embedding similarity only. The goal is to check whether the agent using RAG context produces a materially similar answer to the same agent using the raw transcript. Do not add factual checklist scoring in S2.

Token comparison:

- Estimate prompt tokens for raw context and RAG context.
- Prefer actual usage metadata from the LLM response if available.
- If actual usage is unavailable, use a documented approximation: `ceil(character_count / 4)`.
- Do not add `tiktoken` in S2 unless needed later for tighter accounting.
- Report:
  - `raw_prompt_tokens_estimate`
  - `rag_prompt_tokens_estimate`
  - `token_savings_percent`

Pass condition:

- RAG answer is semantically similar to raw answer.
- RAG uses fewer prompt tokens than raw mode.
- RAG answer includes at least one timestamp reference when relevant chunks have timestamps.

## Observability

Extend MLflow logging:

- `context_mode`: `raw` or `rag`
- `top_k`
- `retrieved_chunk_ids`
- `retrieved_chunk_scores`
- `retrieved_chunk_time_ranges`
- `raw_prompt_tokens_estimate`
- `rag_prompt_tokens_estimate`
- `semantic_similarity`
- `token_savings_percent`

Artifacts:

- `rag_chunks.json`: retrieved chunk text and metadata for each RAG ask.
- `raw_transcript_metadata.json`: raw transcript id and scalar metadata when indexing.
- `context_comparison.json`: raw answer, RAG answer, similarity, token comparison.

Do not log full raw transcript artifacts unless `YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS=true`.

## Testing Requirements

Add tests for:

- Supadata `text=false` segment stream is stored in `raw_transcripts` as serialized JSON document, not metadata.
- Raw mode joins all segment strings from `raw_transcripts` for the LLM context.
- Chunking prefers `raw_transcripts` over flattened raw transcript text.
- Chunking preserves order and timestamps.
- Chunking creates stable `chunk:{video_id}:{chunk_index}` ids.
- Chunking preserves `start_segment_index` and `end_segment_index`.
- Chroma `raw_transcripts` upsert is idempotent.
- Chroma `transcript_chunks` upsert is idempotent.
- `fetch` behaves as an alias for `fetch-raw` and writes `raw_transcripts`.
- RAG retrieval returns top-k chunks with metadata.
- `ask --context rag` auto-indexes when no chunks exist.
- RAG context provider formats timestamped context for the agent.
- Timestamp formatting is consistent for RAG context output.
- Raw and RAG modes both use the same `TranscriptAgent`.
- CLI `ask --context raw` uses `RawTranscriptContextProvider`.
- CLI `ask --context rag --top-k 10` uses `RagTranscriptContextProvider`.
- CLI `compare-context` runs both modes and computes semantic similarity/token comparison.
- Supadata segment timestamp normalization remains covered.

External calls must be mocked in automated tests:

- Supadata transcript fetching.
- DeepSeek/LangChain calls.
- Embedding model calls, unless using a tiny deterministic fake embedding function.

## Manual Verification

From project root:

```bash
uv sync
uv run python -m src.cli fetch-raw "https://www.youtube.com/watch?v=3hk7nO_q0a8"
uv run python -m src.cli index-rag "https://www.youtube.com/watch?v=3hk7nO_q0a8"
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount" --context raw
uv run python -m src.cli ask "https://www.youtube.com/watch?v=3hk7nO_q0a8" "what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount" --context rag --top-k 10
uv run python -m src.cli compare-context "https://www.youtube.com/watch?v=3hk7nO_q0a8" "what does this video say  for capital gains tax, is it being grandfathered or every now under new rules, does that mean if I sell before 30 June 2027 I can still access 50% discount" --top-k 10
uv run pytest
```

## Acceptance Criteria

- Legacy flattened `transcripts` collection is decommissioned as canonical storage.
- Supadata `text=false` segment data is stored in separate Chroma collection `raw_transcripts`.
- Each `raw_transcripts` document has a stable id that identifies the source video: `raw_transcript:{video_id}`.
- Raw transcript storage includes original `offset_ms` and `duration_ms` plus computed `start_seconds` and `end_seconds`.
- Raw transcript metadata includes title and language when available.
- Raw mode joins all strings from `raw_transcripts` for the LLM context.
- Existing summary command still works using raw context from `raw_transcripts`; RAG summary is out of scope for S2.
- Chunking uses `raw_transcripts` as the preferred source so timestamp metadata comes from original provider segments.
- Timestamped chunks are stored in separate Chroma collection `transcript_chunks`.
- Chunk metadata includes transcript id, video id, source URL, source collection, chunk index, start/end seconds, start/end segment indexes, and segment count.
- RAG retrieval returns top 10 chunks by default.
- Same `TranscriptAgent` can answer with raw context or RAG context by swapping providers.
- Required CGT question runs successfully in both raw and RAG modes.
- Similarity comparison reports semantic similarity and passes the configured threshold.
- Token comparison shows RAG context uses fewer prompt tokens than raw context.
- RAG answers include timestamp references when timestamped chunks are retrieved.
- MLflow records context mode, retrieval metadata, answer artifacts, similarity, and token comparison.
- Tests pass with external calls mocked.
