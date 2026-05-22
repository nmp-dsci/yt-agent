# Spec: S6 Recursive (Multi-Hop) RAG In The Existing Agent

Status: draft
Date: 2026-05-18

## Summary

Today `RagTranscriptAgent` (in `src/agents/rag_transcript_agent.py`) does a single retrieval hop: embed the user question, retrieve top-k chunks, ask the LLM to answer with citations, return. S6 extends that same agent — **no new agent class** — with an optional **recursive retrieval** mode (also known as multi-hop, iterative, or agentic RAG).

**Unified LLM contract.** Both single-hop and recursive modes go through the **same LLM call**. That call always returns an answer with inline citations **plus** a structured list of follow-up subtopics and follow-up retrieval queries that target depth gaps in the answer. The only difference between the two modes is what the **agent code** does with the follow-ups:

- Single-hop (default): the LLM is invoked **once**. The agent returns the answer along with the proposed subtopics and follow-up queries. **No additional retrieval happens.** The follow-ups are exposed on the response object for the caller (CLI, dashboard, evals) to inspect.
- Recursive (opt-in): the LLM is invoked, the agent **acts on the follow-ups** by routing each one back through the existing chunk retrieval path, merging the new chunks into accumulated context, and re-invoking the same LLM call. Loop until a stopping rule fires; the last invocation's answer is the final answer.

This keeps the prompt, the response schema, and the parsing code identical for both modes. Recursion is purely a question of "does the agent re-enter the retrieval loop after parsing the LLM's follow-ups, or stop?"

S6's absolute non-negotiable: **update the existing `RagTranscriptAgent` script in place**. Backwards compatibility means **interface and schema compatibility** — same CLI signature, same default-mode behavior shape (answer + references), same calling convention. The new follow-up fields on the response object are additive and default to empty lists, so consumers that don't read them are unaffected. The recursive path is **opt-in**, controlled by new fields on `RagQuestionRequest` and new CLI flags on `rag-ask`. The same question must be runnable through `rag-ask` under any of these configurations without code changes:

| Configuration | Flags |
|---|---|
| Existing single-hop, multi-transcript | (none) |
| Existing single-hop, single-transcript | `--url $URL` |
| Existing single-hop, summary-filtered | `--filter-transcripts` |
| New recursive, multi-transcript | `--recursive` |
| New recursive, summary-filtered | `--recursive --filter-transcripts` |
| New recursive, single-transcript | `--recursive --url $URL` |
| New recursive, tuned | `--recursive --max-followups 4 --top-k 15 --followup-top-k 10` |
| New recursive, env-var default | `YT_AGENT_RAG_RECURSIVE_DEFAULT=true` in `.env`, then run `rag-ask "$question"` |
| New explicit single-hop opt-out when env-default is on | `--no-recursive` |

A reviewer must be able to pick any row above and run the **same** `rag-ask` invocation with only flag changes.

## Current Source Of Truth

Build on the current implementation. Do not duplicate or fork:

- `src/agents/rag_transcript_agent.py`
  - `RagTranscriptAgent` — the single-hop multi-transcript agent. **This file is updated in place by S6.**
  - `RagContextTooLongError`, `_fallback_references`, `_json_object` — kept and reused.
- `src/agents/models.py`
  - `RagQuestionRequest`, `RagAnswerReference`, `RagTranscriptAnswer` — extended (additive only, no field rename, no field removal).
- `src/agents/prompts.py`
  - `RAG_SYSTEM_PROMPT`, `RAG_QUESTION_USER_PROMPT`, `build_rag_question_prompt`, `build_transcript_context_prompt` — reused.
  - New prompts added alongside, not replacing the existing ones.
- `src/rag/context.py`
  - `MultiTranscriptRagContextProvider.get_context(...)` — reused for **every** retrieval hop. S6 does not introduce a parallel retrieval path.
- `src/cli.py`
  - `rag-ask` subparser — extended with new flags. No new subcommand.
- `src/config.py`
  - Extended with recursion defaults (no behavior change when env vars are absent).
- `src/observability.py`
  - `log_context_details`, `log_transcript_filter_details`, `log_answer` — reused. New helper for per-hop logging.

Important current behavior that must be preserved:

- `rag-ask "$question"` with no other flags runs **one** retrieval hop across all indexed transcripts and prints the answer + references block as today.
- `RagTranscriptAgent.answer(RagQuestionRequest(...))` returns a `RagTranscriptAnswer` whose existing JSON fields (`question`, `answer`, `references[]`) are unchanged. New fields (`subtopics`, `followup_queries`, `recursion`) are additive and default to empty/None; callers that ignore them see identical behavior.
- `agent.last_context` continues to expose the **final** retrieved context for the dashboard and observability.

Behavioral note on the unified prompt: because both modes now route through the **same** LLM prompt — which always asks for follow-up subtopics in addition to the answer — the answer text in single-hop mode may differ slightly from `main` for the same question. This is accepted as part of S6: backward compatibility is **schema/interface** compatibility, not **byte-identical LLM output**. The existing answer schema is unchanged; the prompt is updated. Tests assert structural equivalence (valid `RagTranscriptAnswer`, same field set on default request, citations resolvable) rather than exact-string equality against frozen pre-S6 answer text.

## Goals

- Unify the LLM contract across modes: both single-hop and recursive use the **same** prompt and the **same** response parser. The prompt always asks for answer + subtopics + follow-up queries.
- In single-hop mode, expose the follow-ups on the response object but **do not retrieve** for them. The agent invokes the LLM exactly once.
- In recursive mode, **act on** the follow-ups by routing each one back through the existing chunk retrieval path and re-invoking the same LLM call until a stopping rule fires.
- Keep recursion **opt-in** via `RagQuestionRequest.recursive=False` default and a new `--recursive` CLI flag on `rag-ask`.
- Reuse `MultiTranscriptRagContextProvider.get_context(...)` for every hop, so recursion inherits single-URL filtering and S4 summary filtering for free.
- Make recursion configurable with explicit, bounded stopping rules (max depth, max follow-up queries per hop, novelty threshold, total context-char cap).
- Surface the follow-ups (single-hop) and the recursion trace (recursive) on the response object and on `agent.last_context` so the dashboard and observability layers can inspect both modes uniformly.
- Preserve every existing test that asserts response **shape** and CLI **structure**. Update or replace tests that asserted exact LLM answer text, since the unified prompt may shift that text.

## Non-Goals

- Do not introduce a second agent class. `RagTranscriptAgent` is the only RAG agent. (A wrapper `RecursiveRagAgent` is explicitly rejected; the same class handles both modes.)
- Do not change `TranscriptAgent` (`ask --context raw|rag`). Recursive RAG only applies to `rag-ask` and `RagTranscriptAgent`.
- Do not change retrieval, chunking, embeddings, or storage. S6 is an agent-layer change only.
- Do not change S4 transcript-summary filtering semantics. Summary filtering is applied per-hop using the same parameters the caller passed for the first hop.
- Do not introduce parallel LLM calls. The prototype keeps hops sequential.
- Do not add a new dashboard tab. Trace inspection in S6 is via CLI output, observability logs, and the existing `evaluation.html` flow. A dedicated recursion-trace dashboard tab is a follow-up.
- Do not add a second LLM provider. DeepSeek (via `ChatOpenAI`) remains the only LLM.
- Do not change the JSON schema of existing `RagTranscriptAnswer` fields. New fields are additive and default to empty.

## Agent System Architecture

After S6 there are still **two** agent classes in `src/agents/`, exactly as today. S6 does not add a third. The change is **internal to `RagTranscriptAgent`** — it gains a recursive code path behind one `request.recursive` flag while keeping the single-hop path intact.

```text
                          ┌──────────────────────────────┐
                          │            CLI               │
                          │  src/cli.py                  │
                          │                              │
                          │  ask    (raw | rag)          │  ── TranscriptAgent
                          │  summarize                   │  ── TranscriptAgent
                          │  compare-context             │  ── TranscriptAgent (×2)
                          │  rag-ask                     │  ── RagTranscriptAgent
                          │     +--recursive (S6)        │
                          │     +--show-followups (S6)   │
                          │     +--print-trace (S6)      │
                          └──────────────┬───────────────┘
                                         │
            ┌────────────────────────────┴───────────────────────────┐
            │                                                        │
            ▼                                                        ▼
  ┌──────────────────────┐                          ┌─────────────────────────────┐
  │  TranscriptAgent     │                          │  RagTranscriptAgent         │
  │  (unchanged in S6)   │                          │  (UPDATED IN PLACE by S6)   │
  │                      │                          │                             │
  │  context modes:      │                          │  modes:                     │
  │   • raw              │                          │   • single-hop  (default)   │
  │   • rag (single URL) │                          │   • recursive   (S6, opt-in)│
  └─────────┬────────────┘                          │                             │
            │                                       │  shared first-pass prompt   │
            │                                       │  (single-hop AND stage 1    │
            │                                       │   of recursive):            │
            │                                       │   answer + refs + subtopics │
            │                                       │   + follow-up queries       │
            │                                       │                             │
            │                                       │  recursive-only synthesis   │
            │                                       │  prompt (stage 3):          │
            │                                       │   preserved answer +        │
            │                                       │   per-subtopic drill-downs  │
            │                                       └──────────────┬──────────────┘
            │                                                      │
            │                                                      │
            ▼                                                      ▼
  ┌─────────────────────────┐               ┌──────────────────────────────────────┐
  │ Context providers       │               │ MultiTranscriptRagContextProvider    │
  │  • RawTranscriptCP      │               │  .get_context(question, top_k,       │
  │  • RagTranscriptCP      │               │       source_url, filter_transcripts,│
  │    (single URL)         │               │       transcript_filter_*)          │
  └─────────────┬───────────┘               └──────────────┬───────────────────────┘
                │                                          │
                ▼                                          ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                          Storage / Retrieval (unchanged)                   │
  │   raw_transcripts  ·  transcript_chunks  ·  transcript_summaries (S4)      │
  └────────────────────────────────────────────────────────────────────────────┘

           ╔══════════════════════════════════════════════════════════════════════╗
           ║   Recursive mode pipeline (inside RagTranscriptAgent, two stages)    ║
           ║                                                                      ║
           ║   STAGE 1 — first-pass (identical to single-hop)                     ║
           ║      question                                                        ║
           ║         │                                                            ║
           ║         ▼                                                            ║
           ║      retrieve(question)  ──►  LLM (first-pass prompt)                ║
           ║                                  │                                   ║
           ║                                  ▼                                   ║
           ║         {first_answer, first_references, subtopics[],                ║
           ║          followup_queries[]}                                         ║
           ║                                                                      ║
           ║   STAGE 2 — fan-out retrieval (NO LLM calls)                         ║
           ║      for each surviving subtopic s_i:                                ║
           ║         retrieve(s_i.followup_query)                                 ║
           ║         keep novel chunks only (dedup vs. first-pass + earlier s_*)  ║
           ║      ──► SubtopicEvidence[i] = {topic, query, chunks[i]}             ║
           ║                                                                      ║
           ║   STAGE 3 — final synthesis (one LLM call, synthesis prompt)         ║
           ║      input = (question, first_answer, first_references,              ║
           ║               [SubtopicEvidence_1, ..., SubtopicEvidence_n])         ║
           ║                                  │                                   ║
           ║                                  ▼                                   ║
           ║      LLM produces:                                                   ║
           ║         preserved_answer (tightened first_answer, same refs)         ║
         ║         + subtopic_answers[i] cited with scoped labels [s{i}.*]      ║
           ║         + layered_answer_markdown                                    ║
           ║                                                                      ║
           ║   LLM call count: 2 (first-pass + synthesis).                        ║
           ║   Retrieval count: 1 + len(executed_followups).                      ║
           ║                                                                      ║
           ║   Short-circuit (skip stage 3, return first-pass) on:                ║
           ║     · no_followups_requested  · all_followups_filtered               ║
           ║     · no_new_evidence         · max_total_followups_reached          ║
           ║     · response_parse_degraded                                        ║
           ╚══════════════════════════════════════════════════════════════════════╝
```

### Every way the agent can be run

The full catalogue of supported invocations after S6. Existing rows are unchanged; new rows are introduced by S6 and only enabled by adding flags — the same `rag-ask` subcommand covers all single-hop and recursive variants.

**Existing (unchanged by S6)** — `TranscriptAgent` paths via `ask`, `summarize`, `compare-context`:

- Full raw transcript Q&A (single video, no retrieval):
  ```bash
  uv run python -m src.cli ask "$url" "$question" --context raw
  ```
- Single-transcript RAG Q&A (single video, retrieved chunks only):
  ```bash
  uv run python -m src.cli ask "$url" "$question" --context rag --top-k 10
  ```
- One-shot summary of a single transcript:
  ```bash
  uv run python -m src.cli summarize "$url"
  ```
- Side-by-side comparison of raw vs single-transcript RAG for one question:
  ```bash
  uv run python -m src.cli compare-context "$url" "$question" --top-k 10
  ```

**Existing (unchanged by S6)** — `RagTranscriptAgent` single-hop paths via `rag-ask`:

- Multi-transcript RAG across the full indexed corpus:
  ```bash
  uv run python -m src.cli rag-ask "$question" --top-k 20
  ```
- Multi-transcript RAG with S4 transcript-summary filtering applied before chunk retrieval:
  ```bash
  uv run python -m src.cli rag-ask "$question" --filter-transcripts --top-k 20
  ```
- Multi-transcript-RAG agent restricted to one transcript (same agent, scoped to a single video):
  ```bash
  uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
  ```
- S4 summary filtering with tuned filter knobs:
  ```bash
  uv run python -m src.cli rag-ask "$question" --filter-transcripts \
    --transcript-filter-top-k 8 --transcript-filter-min-score 0.3 --top-k 20
  ```

**New in S6** — `RagTranscriptAgent` single-hop paths that **surface** the model's proposed follow-ups (no extra retrieval, still one LLM call):

- Single-hop, show what a deeper pass would have chased:
  ```bash
  uv run python -m src.cli rag-ask "$question" --show-followups
  ```
- Single-transcript, show follow-ups:
  ```bash
  uv run python -m src.cli rag-ask "$question" --url "$url" --show-followups
  ```
- Summary-filtered single-hop, show follow-ups:
  ```bash
  uv run python -m src.cli rag-ask "$question" --filter-transcripts --show-followups
  ```

**New in S6** — `RagTranscriptAgent` recursive paths (acts on follow-ups, multi-hop retrieval):

- Recursive across the full corpus, defaults:
  ```bash
  uv run python -m src.cli rag-ask "$question" --recursive
  ```
- Recursive, restricted to one transcript:
  ```bash
  uv run python -m src.cli rag-ask "$question" --recursive --url "$url"
  ```
- Recursive with S4 summary filtering (uniform across all hops):
  ```bash
  uv run python -m src.cli rag-ask "$question" --recursive --filter-transcripts
  ```
- Recursive, tuned depth and breadth:
  ```bash
  uv run python -m src.cli rag-ask "$question" --recursive \
    --max-depth 3 --max-followups 4 --top-k 15 --followup-top-k 10
  ```
- Recursive with explicit follow-up retrieval cap and novelty threshold:
  ```bash
  uv run python -m src.cli rag-ask "$question" --recursive \
    --max-total-followups 6 --novelty-min-chunks 3
  ```
- Recursive with full trace (chunk previews per expanded subtopic):
  ```bash
  uv run python -m src.cli rag-ask "$question" --recursive --print-trace
  ```
- Recursive + filtering + single-transcript + trace (the everything-on variant for demos):
  ```bash
  uv run python -m src.cli rag-ask "$question" --recursive --filter-transcripts \
    --url "$url" --max-followups 3 --print-trace
  ```
- Env-var-driven recursive default (set once, every `rag-ask` runs recursive):
  ```bash
  # in .env
  YT_AGENT_RAG_RECURSIVE_DEFAULT=true

  uv run python -m src.cli rag-ask "$question"            # recursive
  uv run python -m src.cli rag-ask "$question" --no-recursive   # explicit single-hop opt-out
  ```

The same `$question` can be replayed across every row above by changing only flags (or by flipping the env var). No code changes, no different subcommand.

### readme.md updates required by S6

`readme.md` currently documents the existing four `TranscriptAgent` invocations and the three single-hop `rag-ask` invocations under "Ask questions" and "Agent Architecture". S6 must update `readme.md` so a fresh reader can run every row in the catalogue above without reading this spec. Concretely:

- Update the "Ask questions" section to list **all** supported invocations grouped as: Full transcript (raw) · Single-transcript RAG · Multi-transcript RAG (single-hop) · Multi-transcript RAG (single-hop, show follow-ups) · Multi-transcript RAG (recursive). Each group shows a representative command and a one-line description of what the flags do.
- Add a "Recursive RAG flags" subsection that lists every new flag (`--recursive`, `--max-depth`, `--max-followups`, `--followup-top-k`, `--novelty-min-chunks`, `--max-total-followups`, `--show-followups`, `--print-trace`) with default values and a one-line meaning.
- Add an "Env vars (recursion)" entry to the env-var block at the top of `readme.md` documenting `YT_AGENT_RAG_RECURSIVE_DEFAULT`, `YT_AGENT_RAG_MAX_DEPTH`, `YT_AGENT_RAG_MAX_FOLLOWUPS`, `YT_AGENT_RAG_FOLLOWUP_TOP_K`, `YT_AGENT_RAG_NOVELTY_MIN_CHUNKS`, `YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS`. Mention that these are read only when `--recursive` is set.
- Update the "Agent Architecture" section in `readme.md` to describe the unified LLM contract (one prompt produces answer + subtopics + follow-up queries in both modes) and to add a recursive-mode flow diagram alongside the existing flow diagrams. The diagram should mirror (in shorter form) the recursive-loop box from the Agent System Architecture section of this spec.
- Update the "Architecture" `src/` tree comment in `readme.md` so the description of `agents/` reflects "Full-transcript agent and RAG agent with optional recursive multi-hop retrieval" rather than the current "Full-transcript agent and RAG-only transcript agent" wording.
- Preserve all existing wording about `index-rag`, `bulk-index`, `fetch`, `fetch-raw`, the dashboard, evals, and tests. S6 does not touch those flows; readme edits must be scoped to the agent / Q&A sections.

A reviewer must be able to pick any command from the catalogue, paste it from `readme.md`, and have it run.

## Users and Use Cases

- Demo reviewer: asks a broad question ("what are the risks of migrating our app to Kubernetes" / "what does this corpus say about AI-engineer feature velocity") and wants a layered answer that follows up on subtopics the first hop only gestured at.
- Demo reviewer: wants to compare single-hop and recursive answers for the **same question** by toggling one flag, so the value of recursion is observable.
- Demo reviewer: wants to bound cost and latency with explicit `--max-depth` and `--max-followups` while iterating.
- Evaluator: wants the recursion trace (subtopics, follow-up queries, per-hop chunks) printed and logged so they can judge whether recursion drifted or stayed on topic.

## Behavior

### 1. Mode selection

`RagTranscriptAgent.answer(request)` reads `request.recursive`. Both branches go through the **same** `_invoke_llm(...)` method, which uses the **same** system + user prompts and parses the **same** response schema (`answer`, `references[]`, `subtopics[]`, `followups_requested`). The branches differ only in whether the agent re-enters retrieval after parsing:

- `recursive=False` (default): one retrieval, one LLM call, return. The parsed `subtopics` are attached to the response so callers can see what the model would have followed up on, but the agent does not retrieve for them.
- `recursive=True`: one retrieval, one LLM call, then loop — for each follow-up query the model proposed, retrieve more chunks, merge into accumulated context, call the LLM again. Stop when a rule fires (see Stopping rules). The last LLM call's answer is the final answer.

`agent.last_context` is populated with the final retrieved context in either mode (hop 0 in single-hop; merged accumulated context in recursive) so `log_context_details` and the dashboard keep working without branching.

### 2. Single-hop flow (default)

```text
hop_0 = retrieve(question, top_k, source_url, filter_transcripts)
synth_0 = llm_call(question, hop_0.context_text)
#   synth_0.answer        — inline-cited answer
#   synth_0.references    — list[RagAnswerReference]
#   synth_0.subtopics     — list[{topic, rationale, followup_query, confidence}]
#   synth_0.followups_requested — bool

return RagTranscriptAnswer(
    question=question,
    answer=synth_0.answer,
    references=synth_0.references or fallback_references(synth_0.answer, hop_0),
    subtopics=synth_0.subtopics,                # NEW, additive
    followup_queries=[s.followup_query for s in synth_0.subtopics],  # NEW, additive
    recursion=None,                             # not in recursive mode
)
```

The agent stops here. No additional retrieval is performed even if `followups_requested` is true. This is the entire point of single-hop mode: the user gets a fast answer plus visibility into what a deeper pass would chase, without paying for the deeper pass.

### 3. Recursive flow (opt-in)

Recursive mode is a **two-stage pipeline**. There is one and only one fan-out round at the default depth (`max_depth=1`). The shape:

1. **Stage 1 — First LLM call (shared with single-hop).** Identical to single-hop in every respect: same retrieval, same prompt, same response schema. Produces `first_answer`, `first_references`, and `subtopics` (each with `topic`, `rationale`, `followup_query`, `confidence`).
2. **Stage 2 — Fan-out retrieval.** For each selected follow-up `subtopic`, call `MultiTranscriptRagContextProvider.get_context(question=subtopic.followup_query, ...)` and collect its retrieved chunks. **No LLM call is made per follow-up.** This stage is pure retrieval and is what justifies the `--recursive` flag's cost: one extra retrieval per follow-up.
3. **Stage 3 — Final synthesis LLM call (new, distinct prompt).** A second LLM call whose input is a structured block: original question + `first_answer` + per-subtopic sections, each containing `{topic, rationale, followup_query, retrieved_chunks}`. The model is instructed to produce a **layered final answer**: it preserves and lightly tightens the first answer (it does not contradict it), and under each subtopic it writes a focused sub-answer grounded only in that subtopic's retrieved chunks, with inline citations scoped to those chunks.

Pseudocode (intent only — implement in the agent file directly, not as a separate module):

```text
# ----- Stage 1: first LLM call (same as single-hop) -----
hop_0_context = retrieve(question, top_k, source_url, filter_transcripts)
first = invoke_llm_first_pass(question, hop_0_context.context_text)
#   first.answer        — initial answer with [1] [2] citations against hop_0
#   first.references    — list[RagAnswerReference] for hop_0
#   first.subtopics     — list[FollowupSubtopic]
#   first.followups_requested

# Early exits that keep the cost of an accidental --recursive low:
if not first.followups_requested or not first.subtopics:
    return RagTranscriptAnswer(
        question=question,
        answer=first.answer,
        references=first.references,
        subtopics=first.subtopics,
        recursion=RecursionTrace(
            stages=[Stage(name="first_pass", llm_calls=1, retrievals=1)],
            terminated_reason="no_followups_requested",
            total_followups_executed=0,
            subtopic_answers=[],
        ),
    )

selected = select_followups(first.subtopics, stopping_rules)   # dedup, cap by max_followups, drop low-confidence

# ----- Stage 2: fan-out retrieval, no LLM calls -----
subtopic_blocks: list[SubtopicEvidence] = []
seen_chunk_keys = {(c.video_id, c.chunk_index) for c in hop_0_context.retrieved_chunks}
for s in selected:
    retrieval = retrieve(
        question=s.followup_query,
        top_k=followup_top_k,
        source_url=source_url,             # inherited from the original request
        filter_transcripts=filter_transcripts,
    )
    novel = [c for c in retrieval.retrieved_chunks if key(c) not in seen_chunk_keys]
    if len(novel) < novelty_min_chunks:
        subtopic_blocks.append(SubtopicEvidence(subtopic=s, chunks=[], outcome="no_new_evidence"))
        continue
    seen_chunk_keys |= {key(c) for c in novel}
    subtopic_blocks.append(SubtopicEvidence(subtopic=s, chunks=novel, outcome="merged"))
    if total_followups_executed >= max_total_followups:
        break

executed = [b for b in subtopic_blocks if b.outcome == "merged"]
if not executed:
    # No follow-up produced new evidence — degrade to first-pass result so the
    # user still gets an answer, but mark the trace clearly.
    return RagTranscriptAnswer(
        question=question,
        answer=first.answer,
        references=first.references,
        subtopics=first.subtopics,
        recursion=RecursionTrace(
            stages=[
                Stage(name="first_pass", llm_calls=1, retrievals=1),
                Stage(name="fan_out", llm_calls=0, retrievals=len(selected)),
            ],
            terminated_reason="no_new_evidence",
            total_followups_executed=len(selected),
            subtopic_answers=[],
        ),
    )

# ----- Stage 3: final synthesis LLM call (new, distinct prompt) -----
final = invoke_llm_final_synthesis(
    question=question,
    first_answer=first.answer,
    first_references=first.references,
    subtopic_evidence=executed,     # one block per subtopic with its own retrieved chunks
)
#   final.preserved_answer       — same shape as first.answer, lightly tightened
#   final.preserved_references   — references for preserved_answer (must be subset of first.references)
#   final.subtopic_answers       — list[SubtopicAnswer]: one entry per executed block, each with its
#                                  own answer text + references scoped to that subtopic's chunks
#   final.layered_answer_markdown — a pre-rendered markdown view (preserved + drill-downs)

return RagTranscriptAnswer(
    question=question,
    answer=final.layered_answer_markdown,
    references=final.preserved_references + flatten([sa.references for sa in final.subtopic_answers]),
    subtopics=first.subtopics,
    recursion=RecursionTrace(
        stages=[
            Stage(name="first_pass", llm_calls=1, retrievals=1),
            Stage(name="fan_out", llm_calls=0, retrievals=len(selected)),
            Stage(name="final_synthesis", llm_calls=1, retrievals=0),
        ],
        terminated_reason="completed",
        total_followups_executed=len(executed),
        subtopic_answers=final.subtopic_answers,
        preserved_first_answer=first.answer,    # for diffability against final.preserved_answer
    ),
)
```

LLM-call accounting (matches the user-described pipeline):

- `--recursive` with default settings makes **exactly two LLM calls**: one first-pass, one final synthesis.
- Retrieval calls: `1 + len(selected_followups)` — one for the original question, one per follow-up that survives dedup and confidence trimming.
- The final synthesis call **does not retrieve**. All chunks it sees were collected during fan-out.

**Depth semantics.** The user-described shape is the default (`max_depth=1`): one fan-out round followed by one final synthesis. `max_depth=0` collapses to single-hop (no fan-out, no final synthesis — used only for symmetric testing). `max_depth>=2` is reserved for a follow-up spec where the final synthesis is itself allowed to propose new follow-ups that trigger another fan-out + synthesis round; S6 implements only depth ≤ 1 to keep the prompt count and the output shape bounded. Setting `--max-depth 2+` on the CLI is accepted but currently treated as `1` with a warning logged to observability. See Open Questions.

**Why a distinct synthesis prompt is correct here.** The first-pass prompt asks the model to answer-and-decompose a question against a single retrieval block. The synthesis prompt asks the model to do something genuinely different: preserve a prior answer, route evidence into the correct subtopic, and write per-subtopic drill-downs with citations scoped to that subtopic's chunks only. The user's earlier "same LLM call for both modes" constraint applies to the **first** call (which is identical to single-hop) — the final synthesis is an additional call that exists only in recursive mode.

### 4. Stopping rules

All bounds are enforced in **code**, before any LLM call or retrieval, never relying on the model to self-stop:

- `max_depth` (default `1`, accepted range `0..3` in the CLI; S6 implements `0` and `1` only — see Open Questions). `0` collapses recursive mode to single-hop and runs no fan-out and no final synthesis. `1` runs the two-stage pipeline described in §3.
- `max_followups` (default `3`, range `1..6`): max follow-up queries that survive selection from `first.subtopics`. The agent trims by `confidence` desc, then applies dedup. The LLM may propose more; only the top `max_followups` are retrieved.
- `novelty_min_chunks` (default `2`): a fan-out retrieval must contribute at least this many chunks not already in `hop_0`'s retrieved set. Below this threshold the subtopic is marked `no_new_evidence` and excluded from final synthesis.
- `query_dedup`: a `followup_query` whose normalized text (lowercased, whitespace-collapsed, stripped of trailing punctuation) is within Levenshtein-1 of `question` or of an already-selected follow-up is skipped with outcome `duplicate_query`. Cheap string check; no embedding similarity in S6.
- `max_context_chars` (existing field on `RagTranscriptAgent`, default 40_000): enforced over the **synthesis-call** input (i.e. the structured prompt that bundles `first_answer` + all subtopic evidence blocks). If the bundle would exceed the cap, trim from the lowest-scoring chunks of the **lowest-confidence** subtopic block first; never trim `hop_0` or `first_answer`. If the cap cannot be honoured without trimming hop_0, raise `RagContextTooLongError`.
- `max_total_followups` (default `max_depth * max_followups`, so `3` at the defaults): hard cap on fan-out retrievals across the whole run.

Recursive mode short-circuits to a first-pass-only return (with `terminated_reason` set to one of the values below) when **any** of these holds, **before** the final synthesis call is made:

1. `first.followups_requested=false` or `first.subtopics=[]` → `no_followups_requested`.
2. All selected follow-ups failed dedup → `all_followups_filtered`.
3. Every fan-out retrieval returned `no_new_evidence` → `no_new_evidence`.
4. `max_total_followups` reached before any block succeeded → `max_total_followups_reached`.
5. First-pass response failed JSON validation → `response_parse_degraded` (also short-circuits in single-hop, returning the fallback answer).

Otherwise (`>=1` subtopic block has novel chunks), the final synthesis call runs and `terminated_reason="completed"`.

Short-circuit returns must be indistinguishable from `recursive=False` to non-recursion-aware consumers: `answer` and `references` are taken from the first-pass call exactly as single-hop would have returned them. The `recursion` field is still populated so the trace records why no synthesis happened.

### 5. Prompt contract

S6 introduces **two** prompts in `src/agents/prompts.py`:

- **First-pass prompt** (used by single-hop AND by stage 1 of recursive). This is the existing `RAG_SYSTEM_PROMPT` / `RAG_QUESTION_USER_PROMPT` updated in place to ask for subtopics + follow-up queries. The first-pass prompt is **byte-identical** between modes — single-hop and recursive hop 0 send the exact same strings, asserted by test.
- **Final-synthesis prompt** (new, used only by stage 3 of recursive). A distinct system + user prompt pair: `RECURSIVE_SYNTHESIS_SYSTEM_PROMPT` and `RECURSIVE_SYNTHESIS_USER_PROMPT`. Never invoked in single-hop mode.

#### First-pass prompt (shared)

Updated system prompt (replaces the current `RAG_SYSTEM_PROMPT`):

```text
RAG_SYSTEM_PROMPT = """You are a YouTube transcript RAG agent.

Your job, on every call, is to do TWO things using only the retrieved
transcript chunks provided by the system:

1. Answer the user's question with inline citations like [1], [2].
2. Identify subtopics where the retrieved chunks are thin, conflicting, or
   reference concepts that are not themselves explained in the provided
   chunks, and propose ONE focused follow-up retrieval query for each.

Always emit subtopics and follow-up queries when meaningful gaps exist,
regardless of whether the caller plans to act on them. The caller decides
whether to retrieve for the follow-ups; you only propose them.

Rules:
- Use only the retrieved transcript chunks as evidence for the answer.
- Cite supporting chunks inline using labels like [1] and [2].
- Do not invent names, dates, claims, or conclusions.
- If the retrieved chunks do not contain enough information, say so.
- Never propose follow-up queries that paraphrase the original question.
- Never propose follow-up queries that paraphrase each other.
- Prefer follow-up queries that name specific entities, mechanisms, or
  claims that appeared in the retrieved chunks.
- If no meaningful follow-up exists (the chunks fully answer the question),
  return an empty subtopics list and followups_requested=false.
"""
```

Updated user prompt template (replaces the current `RAG_QUESTION_USER_PROMPT`). Same call shape for both modes:

```text
RAG_QUESTION_USER_PROMPT = """Answer the user question using only the retrieved
transcript chunks, and propose follow-up subtopics for any depth gaps.

Return JSON with this exact shape:
{{
  "question": "{question}",
  "answer": "direct answer with inline citations like [1]",
  "references": [
    {{
      "label": "[1]",
      "source_url": "https://www.youtube.com/watch?v=...",
      "timestamp_url": "https://www.youtube.com/watch?v=...&t=593s",
      "start_seconds": 593.36,
      "end_seconds": 665.44,
      "chunk_index": 10,
      "video_id": "..."
    }}
  ],
  "answer_confidence": 0.0,
  "followups_requested": false,
  "subtopics": [
    {{
      "topic": "short subtopic name",
      "rationale": "why this subtopic deserves a follow-up retrieval",
      "followup_query": "focused retrieval query, not a paraphrase of the original question",
      "confidence": 0.0
    }}
  ]
}}

Question:
{question}
"""
```

#### Final-synthesis prompt (recursive only)

New system prompt:

```text
RECURSIVE_SYNTHESIS_SYSTEM_PROMPT = """You are a YouTube transcript RAG synthesis agent.

You are given:
- The user's original question.
- A FIRST-PASS ANSWER produced by another agent using a first retrieval of
  transcript chunks. Treat this answer as provisional but generally correct.
- A list of SUBTOPICS, each with its own follow-up retrieval query and its
  OWN set of retrieved transcript chunks. The chunks under one subtopic are
  the only evidence for that subtopic's drill-down answer.

Your job is to produce a LAYERED FINAL ANSWER:

1. Preserve the first-pass answer as the top-level response. You may tighten
   phrasing, fix obvious contradictions exposed by new evidence, and remove
   sentences that are now clearly unsupported. You MUST NOT introduce new
   claims at this level. Keep the citations from the first pass that are
   still supported.

2. Under each subtopic, write a focused sub-answer (3-8 sentences) grounded
   ONLY in that subtopic's retrieved chunks. Cite chunks with the labels
   provided in that subtopic's block (e.g. [s1.1], [s1.2] for subtopic 1's
   chunks). Do NOT mix evidence between subtopics in a single sub-answer.

3. If a subtopic's chunks do not actually answer its follow-up query, say so
   in one sentence under that subtopic. Do not fabricate.

Rules:
- Use only the chunks supplied in the structured input. No external knowledge.
- Citation labels are scoped: top-level uses the first-pass labels [1],[2];
  each subtopic uses labels prefixed by its index, e.g. [s1.1],[s2.3].
- Never re-cite a first-pass chunk under a subtopic, and vice versa, unless
  the structured input explicitly lists the same chunk under both blocks.
- If a subtopic block is marked outcome=no_new_evidence, skip it; do not
  invent a drill-down for it.
"""
```

New user prompt template:

```text
RECURSIVE_SYNTHESIS_USER_PROMPT = """Question:
{question}

FIRST-PASS ANSWER (from initial retrieval):
{first_answer}

FIRST-PASS REFERENCES:
{first_references_block}     # labels [1], [2], ... with source_url + timestamp + chunk text

SUBTOPIC EVIDENCE:
{subtopic_evidence_block}    # for each subtopic i with novel chunks:
                             #   - topic, rationale, followup_query
                             #   - chunks labelled [s{i}.1], [s{i}.2], ...
                             #   - chunk text, source_url, timestamp

Return JSON with this exact shape:
{{
  "preserved_answer": "tightened version of the first-pass answer, citing [1] [2] ...",
  "preserved_references": [
    {{"label": "[1]", "source_url": "...", "timestamp_url": "...",
      "start_seconds": 0.0, "end_seconds": 0.0,
      "chunk_index": 0, "video_id": "..."}}
  ],
  "subtopic_answers": [
    {{
      "subtopic_index": 1,
      "topic": "...",
      "followup_query": "...",
      "answer": "focused sub-answer with [s1.1] [s1.2] citations",
      "references": [
        {{"label": "[s1.1]", "source_url": "...", "timestamp_url": "...",
          "start_seconds": 0.0, "end_seconds": 0.0,
          "chunk_index": 0, "video_id": "..."}}
      ]
    }}
  ],
  "layered_answer_markdown": "rendered markdown: preserved_answer on top, then one '## {{topic}}' section per subtopic with its sub-answer and inline citations"
}}
"""
```

Notes:

- The `preserved_references` set MUST be a subset of the first-pass references. The agent validates this and rejects synthesis responses that introduce new top-level references (a clear hallucination signal).
- Each subtopic's `references` MUST cite only labels from that subtopic's block (`[s{i}.*]`). Cross-subtopic citations are rejected.
- `layered_answer_markdown` is rendered by the LLM rather than templated by code so the model can choose how to phrase transitions and which subtopics to elevate; the structured fields are the source of truth for programmatic consumers.
- Both `preserved_answer` and per-subtopic `answer` fields are required; missing fields short-circuit to first-pass-only return with `terminated_reason="synthesis_parse_degraded"`.

A malformed first-pass response is non-fatal in both modes: the existing `_fallback_references` path extracts `answer` + `references` and the subtopics list becomes `[]`. In recursive mode this triggers the `no_followups_requested` short-circuit.

### 5. Prompt contract (unified for both modes)

There is **one** prompt pair used by both single-hop and recursive modes. The existing `RAG_SYSTEM_PROMPT` and `RAG_QUESTION_USER_PROMPT` in `src/agents/prompts.py` are **updated in place** to ask for subtopics + follow-up queries alongside the answer. No mode-specific prompt branches.

Updated system prompt (replaces the current `RAG_SYSTEM_PROMPT`):

```text
RAG_SYSTEM_PROMPT = """You are a YouTube transcript RAG agent.

Your job, on every call, is to do TWO things using only the retrieved
transcript chunks provided by the system:

1. Answer the user's question with inline citations like [1], [2].
2. Identify subtopics where the retrieved chunks are thin, conflicting, or
   reference concepts that are not themselves explained in the provided
   chunks, and propose ONE focused follow-up retrieval query for each.

Always emit subtopics and follow-up queries when meaningful gaps exist,
regardless of whether the caller plans to act on them. The caller decides
whether to retrieve for the follow-ups; you only propose them.

Rules:
- Use only the retrieved transcript chunks as evidence for the answer.
- Cite supporting chunks inline using labels like [1] and [2].
- Do not invent names, dates, claims, or conclusions.
- If the retrieved chunks do not contain enough information, say so.
- Never propose follow-up queries that paraphrase the original question.
- Never propose follow-up queries that paraphrase each other.
- Prefer follow-up queries that name specific entities, mechanisms, or
  claims that appeared in the retrieved chunks.
- If no meaningful follow-up exists (the chunks fully answer the question),
  return an empty subtopics list and followups_requested=false.
"""
```

Updated user prompt template (replaces the current `RAG_QUESTION_USER_PROMPT`). The JSON schema is the same call shape for both modes — single-hop ignores everything below `references` for retrieval purposes but still surfaces it on the response:

```text
RAG_QUESTION_USER_PROMPT = """Answer the user question using only the retrieved
transcript chunks, and propose follow-up subtopics for any depth gaps.

Return JSON with this exact shape:
{{
  "question": "{question}",
  "answer": "direct answer with inline citations like [1]",
  "references": [
    {{
      "label": "[1]",
      "source_url": "https://www.youtube.com/watch?v=...",
      "timestamp_url": "https://www.youtube.com/watch?v=...&t=593s",
      "start_seconds": 593.36,
      "end_seconds": 665.44,
      "chunk_index": 10,
      "video_id": "..."
    }}
  ],
  "answer_confidence": 0.0,
  "followups_requested": false,
  "subtopics": [
    {{
      "topic": "short subtopic name",
      "rationale": "why this subtopic deserves a follow-up retrieval",
      "followup_query": "focused retrieval query, not a paraphrase of the original question",
      "confidence": 0.0
    }}
  ]
}}

Question:
{question}
"""
```

Notes:

- The existing `references` block is kept first in the schema, so the existing parser and the existing dashboard/eval consumers continue to find their fields in the same place.
- `subtopics` is allowed to be empty. `followups_requested=false` with an empty `subtopics` is the model's way of saying "no deeper pass needed." In recursive mode this is a terminal signal at any depth.
- `answer_confidence` is informational in S6 (not used to gate stopping). It is logged for evaluation so a later spec can add a confidence-based early stop.
- A malformed response (invalid JSON, schema violation) is non-fatal: parsing falls back to extracting only `answer` and `references` via the existing `_fallback_references` path; `subtopics` becomes `[]` and `followups_requested` becomes `false`. In recursive mode this terminates the loop with `terminated_reason="response_parse_degraded"`.

### 6. CLI surface (`rag-ask`)

Extend the existing subparser. **Do not add a new subcommand.** All flags below default to "single-hop, unchanged behavior" when omitted.

```bash
# Existing (unchanged)
uv run python -m src.cli rag-ask "$question" --top-k 20
uv run python -m src.cli rag-ask "$question" --url "$url" --top-k 10
uv run python -m src.cli rag-ask "$question" --filter-transcripts --top-k 20

# New: opt in to recursive RAG
uv run python -m src.cli rag-ask "$question" --recursive
uv run python -m src.cli rag-ask "$question" --recursive --max-depth 2 --max-followups 3

# New: recursive + summary filtering
uv run python -m src.cli rag-ask "$question" --recursive --filter-transcripts

# New: recursive restricted to one transcript
uv run python -m src.cli rag-ask "$question" --recursive --url "$url"

# New: recursive with a different top-k for follow-up hops
uv run python -m src.cli rag-ask "$question" --recursive --top-k 20 --followup-top-k 10
```

New flags on `rag-ask`:

```text
--recursive                  # opt in to recursive RAG (default: off)
--max-depth N                # default from settings, fallback 2
--max-followups N            # default from settings, fallback 3
--followup-top-k N           # default = --top-k
--novelty-min-chunks N       # default 2
--max-total-followups N      # default max_depth * max_followups
--print-trace                # print per-hop subtopics, queries, retrieved chunk count
```

`--recursive` without further tuning must work end-to-end using the defaults. The other flags are tuning knobs for evaluation.

When `--recursive` is **not** passed, none of the recursive flags are honored — they are silently ignored to keep `rag-ask --recursive=false --max-depth=2` from looking meaningful. The argparse layer keeps them addressable as a group: a `RecursionOptions` dataclass is built only when `--recursive` is set, and the request omits the recursion fields otherwise so the request looks identical to today's wire shape.

### 7. Output

Existing CLI output is preserved in structure. The `Answer` and `References` blocks always print first, in the same order they print today, so any existing parsing of CLI output is unaffected.

Two new optional blocks are appended **after** the existing blocks:

- `Proposed follow-ups`: always available because the unified LLM call always returns subtopics. Printed in single-hop mode **only when** `--show-followups` is passed (default: hidden, to keep default stdout structurally identical to today). Printed unconditionally in recursive mode as the first part of the trace.
- `Recursion trace`: only when `--recursive` is set. Shows per-hop activity.

Single-hop with `--show-followups`:

```text
Answer                          # existing
...

References                      # existing
...

Proposed follow-ups             # new, only with --show-followups in single-hop
1. secrets management on Kubernetes  (confidence 0.78)
   query: "Kubernetes secret rotation flow KMS migration"
2. observability gaps during migration  (confidence 0.65)
   query: "kubernetes migration observability dashboards alerts"
```

Recursive output is **layered**: the preserved first-pass answer prints first under the existing `Answer` header, then one section per executed subtopic under a `Drill-downs` header, each with its own scoped citations, then the `References` block (combined preserved + per-subtopic references in label order), then `Recursion trace`:

```text
Answer                          # preserved first-pass answer, tightened
The corpus shows three recurring patterns AI engineers use Claude for: [1]
spec authoring, [2] PR review, and [3] feature shipping with manual review
gates. Risk emphasis: scope drift and silent regressions in untested code [2].

Drill-downs                     # new, only when --recursive and synthesis ran
## 1. spec-driven workflow vs. ad-hoc prompting   (confidence 0.78)
   follow-up: "spec-driven Claude workflow checklist examples"
   Engineers report writing a short spec before delegating, with a checklist
   the agent must satisfy [s1.1]. The reported failure mode is forgetting to
   list test commands, which lets the agent skip verification [s1.3].

## 2. review-only vs. write-access human gates    (confidence 0.65)
   follow-up: "review gates AI generated code feature delivery"
   Two speakers describe a review-only gate where the human runs and reads
   the diff before merge [s2.2]. One contrasts this with autonomous merge
   over a feature flag, with rollback wired in [s2.4].

## 3. testing strategy for AI-generated features  (no new evidence — skipped)
   follow-up: "AI generated code test coverage strategy"

References                      # combined: preserved + per-subtopic, in label order
[1] https://www.youtube.com/watch?v=... t=120s video=...
[2] https://www.youtube.com/watch?v=... t=480s video=...
[3] https://www.youtube.com/watch?v=... t=901s video=...
[s1.1] https://www.youtube.com/watch?v=... t=15s  video=...
[s1.3] https://www.youtube.com/watch?v=... t=88s  video=...
[s2.2] https://www.youtube.com/watch?v=... t=210s video=...
[s2.4] https://www.youtube.com/watch?v=... t=355s video=...

Recursion trace                 # always printed when --recursive
Stages: first_pass (1 LLM call, 1 retrieval) →
        fan_out  (0 LLM calls, 3 retrievals) →
        final_synthesis (1 LLM call, 0 retrievals)
Terminated: completed
Follow-ups proposed: 3
Follow-ups executed: 2          # one was dropped as no_new_evidence
Total LLM calls: 2
```

When `--print-trace` is set, each subtopic line is followed by the chunk preview (first ~120 chars of every chunk fed into that subtopic block), matching the existing dashboard truncation pattern.

Default `rag-ask "$question"` (no `--recursive`, no `--show-followups`) prints exactly the existing two blocks (`Answer`, `References`) in the existing order. The new blocks are gated behind flags so default stdout structure is unchanged.

When recursive mode short-circuits to a first-pass-only return (`terminated_reason ∈ {no_followups_requested, all_followups_filtered, no_new_evidence, max_total_followups_reached, response_parse_degraded}`), the output collapses back to the single-hop layout: `Answer` + `References` blocks taken from the first pass, plus a one-line `Recursion trace` note explaining why synthesis was skipped. No `Drill-downs` header is printed.

### 8. Backward compatibility

Backward compatibility in S6 is **interface and schema compatibility**, not byte-identical LLM output. The unified prompt asks the model to do more (answer + subtopics), so the answer text itself may shift. The contract guarantees:

1. **CLI signature unchanged.** `rag-ask "$question"` with no new flags runs and exits 0 on every input that worked on `main`.
2. **CLI default stdout structure unchanged.** Default `rag-ask` prints `Answer` and `References` blocks in the same order and format as today. The new `Proposed follow-ups` and `Recursion trace` blocks are gated behind `--show-followups` and `--recursive` respectively. A default invocation prints neither.
3. **Response schema additive only.** `RagTranscriptAnswer` keeps `question`, `answer`, `references[]` with their existing types. New fields (`subtopics`, `followup_queries`, `recursion`) are additive; deserializers that ignore unknown fields are unaffected.
4. **Default request shape unchanged.** `RagQuestionRequest(question="...")` is still a valid call. New fields default to `recursive=False` and `recursion_options=None`, so callers that don't know about S6 build identical requests.
5. **`agent.last_context` semantics unchanged.** A `TranscriptContext` reflecting the final retrieved chunks. In single-hop mode this is hop 0. In recursive mode this is the merged accumulated context.
6. **Env vars gated by effective recursive state.** Recursion-tuning env vars (`YT_AGENT_RAG_MAX_DEPTH`, `YT_AGENT_RAG_MAX_FOLLOWUPS`, `YT_AGENT_RAG_FOLLOWUP_TOP_K`, `YT_AGENT_RAG_NOVELTY_MIN_CHUNKS`, `YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS`) are read **only when** recursive mode is effectively on for the request. Setting `YT_AGENT_RAG_MAX_DEPTH=3` while recursion is off must not change any behavior. `YT_AGENT_RAG_RECURSIVE_DEFAULT` is the single switch that flips the CLI default — when unset or `false`, `rag-ask` is non-recursive by default and matches today's behavior exactly.
7. **No store schema changes.** Chroma collections and on-disk layouts are unchanged. S6 writes no new files outside MLflow.

Out of scope for "backward compatibility":

- Exact LLM answer text. The unified prompt is a deliberate change; pre-S6 answer fixtures that asserted exact strings are updated as part of S6.
- Token counts on the prompt. The system prompt grew; token estimates in observability will shift slightly.

## Interfaces

### `src/agents/models.py` — additive only

```python
class RecursionOptions(BaseModel):
    max_depth: int = 1                       # S6 implements 0..1; CLI accepts up to 3 with warn
    max_followups: int = 3
    followup_top_k: int | None = None        # defaults to top_k when None
    novelty_min_chunks: int = 2
    max_total_followups: int | None = None   # defaults to max_depth * max_followups


class RagQuestionRequest(BaseModel):
    question: str
    source_url: HttpUrl | None = None
    top_k: int = 10
    filter_transcripts: bool = False
    transcript_filter_top_k: int = 5
    transcript_filter_min_score: float = 0.25
    # New fields (default-off so today's callers are unchanged):
    recursive: bool = False
    recursion_options: RecursionOptions | None = None


class FollowupSubtopic(BaseModel):
    """A single proposed follow-up emitted by the first-pass LLM. Produced in BOTH modes."""
    topic: str
    rationale: str
    followup_query: str
    confidence: float


class SubtopicEvidence(BaseModel):
    """One subtopic block fed into the synthesis prompt (recursive only)."""
    subtopic_index: int                       # 1-based; used to scope citation labels [s{i}.*]
    subtopic: FollowupSubtopic
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    outcome: str                              # "merged" | "no_new_evidence" | "duplicate_query"


class SubtopicAnswer(BaseModel):
    """One drill-down sub-answer produced by the synthesis LLM."""
    subtopic_index: int
    topic: str
    followup_query: str
    answer: str                               # cited with [s{i}.*] labels
    references: list[RagAnswerReference] = Field(default_factory=list)


class RecursionStage(BaseModel):
    name: str                                 # "first_pass" | "fan_out" | "final_synthesis"
    llm_calls: int                            # 1 | 0 | 1 respectively at defaults
    retrievals: int                           # 1 | len(selected_followups) | 0


class RecursionTrace(BaseModel):
    stages: list[RecursionStage]
    subtopic_evidence: list[SubtopicEvidence] # one per follow-up considered (incl. dropped)
    subtopic_answers: list[SubtopicAnswer]    # one per executed (merged) follow-up; [] on short-circuit
    preserved_first_answer: str | None = None # only set when synthesis ran; for diff vs. preserved_answer
    terminated_reason: str                    # "completed" | "no_followups_requested" |
                                              # "all_followups_filtered" | "no_new_evidence" |
                                              # "max_total_followups_reached" |
                                              # "response_parse_degraded" |
                                              # "synthesis_parse_degraded"
    total_followups_proposed: int
    total_followups_executed: int


class RagTranscriptAnswer(BaseModel):
    question: str
    # `answer`:
    #   single-hop          -> first-pass answer text (today's shape)
    #   recursive completed -> layered_answer_markdown from the synthesis call
    #   recursive short-circuit -> first-pass answer text (matches single-hop exactly)
    answer: str
    # `references`:
    #   single-hop          -> first-pass references
    #   recursive completed -> preserved_references + flatten(subtopic_answers[*].references), label order
    #   recursive short-circuit -> first-pass references
    references: list[RagAnswerReference] = Field(default_factory=list)
    # Always populated when the first-pass model proposes any. Single-hop callers
    # can read these without paying for synthesis.
    subtopics: list[FollowupSubtopic] = Field(default_factory=list)
    followups_requested: bool = False
    answer_confidence: float | None = None
    # Populated only when recursive=True. Present even on short-circuit so the
    # caller can see why synthesis was skipped.
    recursion: RecursionTrace | None = None
```

`FollowupSubtopic` is parsed from the first-pass LLM response and surfaced on the answer in both modes. `SubtopicEvidence`, `SubtopicAnswer`, and `RecursionStage` are recursive-only types used to structure the synthesis input and trace.

### `src/agents/rag_transcript_agent.py` — same class, same constructor, same `from_settings`

```python
class RagTranscriptAgent:
    def __init__(self, llm, context_provider, max_context_chars: int = 40_000) -> None:
        ...

    @classmethod
    def from_settings(cls, settings, context_provider=None) -> "RagTranscriptAgent":
        ...  # unchanged signature

    def answer(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        if not request.recursive:
            return self._answer_single_hop(request)
        return self._answer_recursive(request)

    # ---- Shared first-pass core (used by single-hop AND stage 1 of recursive) ----
    # One retrieval + one LLM call against the first-pass prompt. Produces
    # answer + references + subtopics + follow-up queries. The prompt strings
    # used here are byte-identical between single-hop and recursive hop 0.
    def _invoke_first_pass(self, question: str, context_text: str) -> _FirstPassResult: ...

    # ---- Single-hop ----
    # Call _invoke_first_pass once, wrap in RagTranscriptAnswer, return.
    def _answer_single_hop(self, request) -> RagTranscriptAnswer: ...

    # ---- Recursive (two-stage pipeline) ----
    # 1. _invoke_first_pass
    # 2. Fan-out: for each surviving subtopic, call context_provider.get_context()
    #    — no LLM calls in this stage.
    # 3. _invoke_final_synthesis with the structured evidence bundle.
    # Short-circuits to first-pass-only return when no subtopic block survives.
    def _answer_recursive(self, request) -> RagTranscriptAnswer: ...

    # Stage-3 only. Uses the synthesis prompt, NOT the first-pass prompt.
    # Input is a structured bundle: question + first_answer + per-subtopic
    # evidence blocks (each with its own scoped chunk labels).
    def _invoke_final_synthesis(
        self,
        question: str,
        first_answer: str,
        first_references: list[RagAnswerReference],
        evidence: list[SubtopicEvidence],
    ) -> _SynthesisResult: ...
```

`_FirstPassResult` mirrors the first-pass response schema (`answer`, `references`, `subtopics`, `followups_requested`, `answer_confidence`). `_SynthesisResult` mirrors the synthesis response schema (`preserved_answer`, `preserved_references`, `subtopic_answers`, `layered_answer_markdown`). Both are Pydantic models local to the file and are the only deserialization targets for their respective LLM responses.

### `src/agents/prompts.py` — update existing + add synthesis pair

Two changes:

1. **Update in place** `RAG_SYSTEM_PROMPT` and `RAG_QUESTION_USER_PROMPT` to the first-pass contract shown in §5. Keep `build_rag_question_prompt(question)` and `build_transcript_context_prompt(context_text)` as the public builders — their signatures do not change; only the templates they format do. These are used by **both** single-hop and recursive stage 1.

2. **Add** `RECURSIVE_SYNTHESIS_SYSTEM_PROMPT`, `RECURSIVE_SYNTHESIS_USER_PROMPT`, and a `build_recursive_synthesis_prompt(question, first_answer, first_references, evidence)` builder. Used **only** by recursive stage 3. No mode branches inside the first-pass prompt itself.

`SYSTEM_PROMPT`, `SUMMARY_USER_PROMPT`, and `QUESTION_USER_PROMPT` (used by the older `TranscriptAgent` for raw-transcript and single-video RAG via the `ask` command) are **not** touched. S6 only changes the multi-transcript RAG agent's prompts.

### `src/cli.py` — extend `rag-ask` subparser only

```python
recursive_group = rag_ask.add_mutually_exclusive_group()
recursive_group.add_argument("--recursive", dest="recursive", action="store_true",
                             default=None)   # None = "let Settings decide"
recursive_group.add_argument("--no-recursive", dest="recursive", action="store_false")
rag_ask.add_argument("--max-depth", type=int, default=None)
rag_ask.add_argument("--max-followups", type=int, default=None)
rag_ask.add_argument("--followup-top-k", type=int, default=None)
rag_ask.add_argument("--novelty-min-chunks", type=int, default=None)
rag_ask.add_argument("--max-total-followups", type=int, default=None)
rag_ask.add_argument("--show-followups", action="store_true")  # single-hop too
rag_ask.add_argument("--print-trace", action="store_true")     # recursive only
```

The CLI parses `recursive` as a tristate: explicit `--recursive` → True, explicit `--no-recursive` → False, neither flag → fall back to `Settings.rag_recursive_default` (from `YT_AGENT_RAG_RECURSIVE_DEFAULT`, which itself defaults to `false`). This is the env-var-driven recursive mode the user asked for: setting `YT_AGENT_RAG_RECURSIVE_DEFAULT=true` in `.env` makes every `rag-ask` invocation recursive by default, with `--no-recursive` as the explicit opt-out.

When recursive mode is effectively on, build `RecursionOptions` from CLI flags with fallback to `Settings` and pass it on the request. `--show-followups` controls whether the `Proposed follow-ups` block is appended to stdout in single-hop mode; it is implicitly on in recursive mode where `Drill-downs` already exposes the follow-ups with their drill-down answers.

### `src/config.py` — additive settings (no behavior change when env vars absent)

```text
YT_AGENT_RAG_RECURSIVE_DEFAULT=false         # flips the CLI default for --recursive
YT_AGENT_RAG_MAX_DEPTH=1                     # S6 implements 0..1
YT_AGENT_RAG_MAX_FOLLOWUPS=3
YT_AGENT_RAG_FOLLOWUP_TOP_K=                 # empty -> defaults to RAG_TOP_K
YT_AGENT_RAG_NOVELTY_MIN_CHUNKS=2
YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS=            # empty -> max_depth * max_followups
```

`YT_AGENT_RAG_RECURSIVE_DEFAULT=true` is the supported env-var switch for turning recursion on by default. The other `YT_AGENT_RAG_*` recursion vars are read only when recursion is effectively on for the request, so setting them while `YT_AGENT_RAG_RECURSIVE_DEFAULT=false` is a no-op on non-recursive runs.

Document each in `readme.md` (see Verification).

## Data

- Inputs:
  - User question via `rag-ask`.
  - Existing Chroma collections (`raw_transcripts`, `transcript_chunks`, `transcript_summaries`) — unchanged.
  - Recursion knobs via CLI flags or env defaults.
- Outputs:
  - Same `RagTranscriptAnswer` shape, with the new optional `recursion` field populated in recursive mode.
  - Same CLI stdout in single-hop mode; an appended `Recursion trace` block in recursive mode.
  - Existing MLflow run gains new params (`recursive`, `max_depth`, `max_followups`, `followup_top_k`) and new metrics (`hop_count`, `total_followups_executed`, `new_chunks_total`) when recursive.
- Persistence:
  - S6 does not write any new files. No new on-disk artifacts under `.yt-agent/` or `dashboard/`. The recursion trace lives on the in-memory answer object and the MLflow run.

## Constraints

- **Backwards compatibility is the binding constraint.** Any change that alters single-hop behavior is out of bounds, including reordering existing log calls, renaming env vars, or changing the default value of an existing flag. Single-hop output must be byte-identical on the frozen test fixture.
- One agent class, one CLI subcommand. The recursive path lives inside `src/agents/rag_transcript_agent.py`. Do not introduce `src/agents/recursive_rag_agent.py` or a new `recursive-rag-ask` command.
- Sequential hops only. No async, no parallel LLM calls, no parallel retrieval calls. The prototype prioritises observability over throughput.
- The same DeepSeek model used for single-hop answers is used for decomposition and final synthesis. No second model, no model-routing logic in S6.
- Every retrieval hop goes through `MultiTranscriptRagContextProvider.get_context(...)` with the same `source_url`, `filter_transcripts`, `transcript_filter_top_k`, and `transcript_filter_min_score` that the caller passed for the first hop. S4 summary filtering is therefore applied uniformly across all hops.
- `max_context_chars` is enforced across the **accumulated** context, not per-hop. The cap remains 40_000 by default.
- LLM outputs are still JSON parsed via the existing `_json_object` helper. Schema validation uses Pydantic. A malformed decomposition response is **non-fatal**: the agent treats it as `followups_requested: false` and falls back to single-hop, but logs the parse failure to observability so it is visible.
- Tests mock the LLM and the context provider. No live Supadata, DeepSeek, or embedding calls in CI.
- No new third-party dependencies. Recursion uses only existing packages (LangChain, Pydantic, the Chroma stack already pulled by `uv sync`).
- The recursion trace must not include raw transcript text. It includes chunk identifiers (`video_id`, `chunk_index`, `timestamp`) and short previews via the existing reference formatting, so observability artifacts stay compact.

## Completion Test

From a clean clone with env set up and the existing corpus indexed:

```bash
question="what does this corpus say about how AI engineers leverage Claude to fully develop features, and what are the risks they call out"

# 1. Baseline (existing, unchanged)
uv run python -m src.cli rag-ask "$question" --top-k 10

# 2. Recursive, defaults
uv run python -m src.cli rag-ask "$question" --recursive

# 3. Recursive + summary filtering
uv run python -m src.cli rag-ask "$question" --recursive --filter-transcripts

# 4. Recursive, tuned depth
uv run python -m src.cli rag-ask "$question" --recursive --max-depth 2 --max-followups 4 --top-k 15 --print-trace
```

Expected:

- Run 1 prints `Answer` and `References` blocks in the same structure as today, exits 0, and does NOT print `Proposed follow-ups` or `Recursion trace`. The LLM is invoked exactly once. The answer text itself may differ from pre-S6 because the unified prompt is in effect; structural fixtures pass.
- Internally, the response object from run 1 has `subtopics` populated (when the model proposes any) and `recursion=None`. A reviewer can confirm this via the MLflow run params/metrics, even though the CLI hides the follow-ups by default.
- Runs 2, 3, 4 print the existing answer + references blocks, followed by `Proposed follow-ups (final)` and `Recursion trace` blocks. The trace lists at least one expanded subtopic with a follow-up query distinct from the original question, and shows a hop count ≥ 2.
- Run 4 prints chunk previews under each subtopic because `--print-trace` is set.
- Across runs 2–4, `agent.last_context` (visible via the existing MLflow logging) reflects the merged accumulated context and `log_context_details` reports a `retrieved_chunks` count strictly greater than run 1.
- Re-running run 1 after merging S6 produces the same MLflow run params as before plus the new params with default values (`recursive=false`); existing MLflow consumers do not break.

## Acceptance Criteria

- `RagTranscriptAgent` gains a recursive code path inside the existing file. No new agent class is added.
- The **first-pass** LLM call is shared verbatim between single-hop and recursive stage 1: same system prompt, same user-prompt template, same response parser, same response schema. A test asserts the prompt strings are byte-identical between modes.
- Recursive mode adds a **separate final-synthesis LLM call** (stage 3) with its own distinct prompt pair (`RECURSIVE_SYNTHESIS_SYSTEM_PROMPT` / `RECURSIVE_SYNTHESIS_USER_PROMPT`). This call is **never** invoked in single-hop mode.
- `RAG_SYSTEM_PROMPT` and `RAG_QUESTION_USER_PROMPT` are updated in place to the first-pass contract (answer + references + subtopics + follow-up queries).
- `RagQuestionRequest` gains `recursive: bool = False` and `recursion_options: RecursionOptions | None = None`. Existing fields are unchanged.
- `RagTranscriptAnswer` gains `subtopics`, `followups_requested`, `answer_confidence`, `recursion`. All additive. Existing fields unchanged.
- In **single-hop** mode the LLM is invoked **exactly once** (first-pass only). The returned `RagTranscriptAnswer` has `subtopics` populated when the model proposes any, `recursion=None`, and default CLI stdout structure unchanged from today (`Proposed follow-ups` block appears only with `--show-followups`).
- In **recursive** mode that **runs to completion** (`terminated_reason="completed"`), the LLM is invoked **exactly twice**: one first-pass call + one final-synthesis call. Retrieval count is `1 + len(executed_followups)`. The final-synthesis call sees the first-pass answer and per-subtopic evidence blocks; it never re-retrieves.
- In **recursive** mode that **short-circuits** (no surviving follow-ups), the LLM is invoked **exactly once** and the returned answer is bit-for-bit the first-pass answer. The `recursion` field is still populated with the trace explaining why synthesis was skipped.
- `rag-ask` gains the new flags listed above. The existing flags (`--url`, `--top-k`, `--filter-transcripts`, `--transcript-filter-top-k`, `--transcript-filter-min-score`) work unchanged.
- Stopping rules are enforced by code, not by the LLM. Tests cover: no_followups_requested, all_followups_filtered, no_new_evidence, max_total_followups_reached, query dedup, response_parse_degraded, synthesis_parse_degraded.
- Synthesis-response validation: `preserved_references` MUST be a subset of first-pass references; each `subtopic_answers[i].references` MUST cite only labels from `subtopic_evidence[i].chunks`. Cross-subtopic citations and net-new top-level references are rejected.
- A malformed first-pass response degrades to fallback `answer` + `references`, `subtopics=[]`, and (in recursive) short-circuits with `terminated_reason="response_parse_degraded"`.
- A malformed synthesis response (recursive only) short-circuits to first-pass-only return with `terminated_reason="synthesis_parse_degraded"`. The user still gets a valid answer.
- `agent.last_context` reflects the first-pass retrieved context in single-hop mode. In recursive mode it reflects the **union** of the first-pass context and every executed subtopic's retrieved chunks.
- `max_context_chars` is enforced over the synthesis-call bundle (`first_answer` + per-subtopic blocks). A new test triggers the trim path on the lowest-confidence subtopic block and asserts the first-pass block is preserved.
- MLflow run captures: `recursive`, `max_depth`, `max_followups`, `followup_top_k`, `total_llm_calls` (1 or 2), `total_retrievals`, `total_followups_proposed`, `total_followups_executed`, `terminated_reason`.
- `readme.md` is updated per the "readme.md updates required by S6" subsection of Agent System Architecture. Specifically: the "Ask questions" section lists every supported invocation grouped by mode; a "Recursive RAG flags" subsection documents every new flag with defaults; the env-var block lists every new `YT_AGENT_RAG_*` recursion var; the "Agent Architecture" section describes the unified LLM contract and includes a recursive-mode flow diagram; the `src/` architecture tree comment is updated to reflect optional recursive multi-hop retrieval. Existing wording about `index-rag`, `bulk-index`, `fetch`, dashboard, evals, and tests is preserved.

## Verification

- Tests:
  - Schema check: `RagTranscriptAgent.answer(RagQuestionRequest(question="..."))` (default request) returns a `RagTranscriptAnswer` with the original fields populated and the new fields defaulted (`subtopics=[]` or model-proposed list, `followups_requested` from the model, `recursion=None`). Asserted against a structural fixture (field set + types), not exact pre-S6 answer text.
  - Same first-pass prompt across modes: assert the system + user prompt strings sent to the LLM in single-hop and in recursive stage 1 are byte-identical (same template, same parameter values).
  - Single-hop LLM-call count: with a mock that returns `followups_requested=true` and two subtopics, assert the mocked LLM is invoked **exactly once** and no follow-up retrieval happens.
  - Single-hop follow-up surface: the response exposes both subtopics on `answer.subtopics`; CLI stdout includes them only when `--show-followups` is passed.
  - Recursive happy-path: with a first-pass mock returning two subtopics and a context provider that returns ≥`novelty_min_chunks` novel chunks per follow-up, assert:
    - LLM invoked **exactly twice** (first-pass + synthesis), with the second call using the synthesis prompt.
    - `MultiTranscriptRagContextProvider.get_context` invoked **3 times** (question + 2 follow-ups).
    - `RecursionTrace.stages` = `[first_pass, fan_out, final_synthesis]` with `llm_calls` totals (1, 0, 1) and `retrievals` totals (1, 2, 0).
    - `RecursionTrace.subtopic_answers` has 2 entries; each cites only its own `[s{i}.*]` labels.
    - `answer` is the synthesis `layered_answer_markdown`; `references` is preserved + per-subtopic in label order.
  - Recursive short-circuits (each is a separate test, all assert exactly **one** LLM call and that `answer` matches first-pass answer bit-for-bit):
    - `no_followups_requested`: first-pass returns `followups_requested=false`.
    - `all_followups_filtered`: every proposed follow-up is dropped by dedup (paraphrase of `question`).
    - `no_new_evidence`: every follow-up's context provider returns chunks that overlap fully with hop 0.
    - `max_total_followups_reached`: cap set to 0 → first follow-up retrieval is blocked.
    - `response_parse_degraded`: first-pass mock returns invalid JSON.
  - Synthesis parse failure: first-pass succeeds with 2 subtopics, fan-out succeeds, but the synthesis mock returns invalid JSON. Assert short-circuit to first-pass answer with `terminated_reason="synthesis_parse_degraded"` and LLM-call count = 2.
  - Citation validation: synthesis mock returns `preserved_references` containing a label not present in first-pass references → rejected, falls back to first-pass with `terminated_reason="synthesis_parse_degraded"`.
  - Cross-subtopic citation rejection: synthesis mock returns `subtopic_answers[0].references` citing a chunk from subtopic 1's block → rejected as above.
  - Query dedup: two subtopics whose `followup_query` differ only by trailing punctuation produce one executed follow-up and one `outcome="duplicate_query"` evidence entry; synthesis runs over the surviving block only.
  - `max_total_followups`: with `max_followups=3, max_total_followups=2`, only 2 fan-out retrievals run and the third subtopic is recorded as `duplicate_query=False, outcome="max_total_followups_reached"`.
  - `max_context_chars`: with a small cap and 3 surviving subtopic blocks, assert the trim path drops the lowest-confidence subtopic block first; first-pass block is preserved.
  - CLI structural compat: `rag-ask "$q"` (no flags) stdout contains `Answer` and `References` blocks and does NOT contain `Drill-downs`, `Proposed follow-ups`, or `Recursion trace`. Block headers and order match today's output.
  - CLI recursive happy-path: `rag-ask "$q" --recursive` stdout contains `Answer`, `Drill-downs` (one section per executed subtopic), `References` (combined, label order), and `Recursion trace` in that order.
  - CLI recursive short-circuit: when no follow-up survives, stdout has `Answer` + `References` (single-hop layout) + a one-line `Recursion trace` note; no `Drill-downs` header.
  - Env-var-driven default: setting `YT_AGENT_RAG_RECURSIVE_DEFAULT=true` in `.env` flips the CLI default so `rag-ask "$q"` runs recursive; passing `--no-recursive` (added when the env-default is opt-in) overrides back to single-hop.
  - Recursion env vars are read **only when** `--recursive` is effectively true; with the default `false`, setting `YT_AGENT_RAG_MAX_FOLLOWUPS=10` does not affect single-hop behavior.
- Manual checks:
  - Run the four commands in the Completion Test against the live corpus. Confirm recursive runs cite chunks from videos that did not appear in run 1's references for at least one subtopic.
  - Confirm MLflow UI shows the new recursion params and metrics on the recursive runs and the default values on the baseline run.
  - Confirm `dashboard/rag_pipeline.html` regeneration is unaffected (the dashboard does not depend on recursion fields).
  - **Readme paste-test:** copy every command block from the updated `readme.md` Q&A section (raw, single-transcript RAG, multi-transcript single-hop, multi-transcript single-hop with `--show-followups`, multi-transcript recursive in each documented variant) into a shell against the live corpus. Every command must exit 0 and produce the documented output structure. A reviewer should not need this spec to run any documented invocation.

## Open Questions

- Should `agent.last_context` expose a `TranscriptContext` per hop (list) in addition to the merged context? S6 deliberately keeps it as the merged final context to avoid breaking observability call sites; per-hop contexts live on the `RecursionTrace`. Revisit if the dashboard needs them.
- Should follow-up hops bypass S4 summary filtering on the assumption that follow-up queries are more specific and benefit from the full chunk index? S6 keeps filtering uniform across hops; this is the simpler default and the prototype audience benefits from a single dial. A per-hop override is a candidate for a follow-up spec.
- Confidence-driven early stop: the current rule treats confidence as informational. A future spec may stop recursion once `answer_confidence` exceeds a threshold; out of scope here to keep the bound count testable.
