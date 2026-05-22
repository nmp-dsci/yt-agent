from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.context import RawTranscriptContextProvider
from src.agents.models import (
    FollowupSubtopic,
    QuestionRequest,
    RagQuestionRequest,
    RecursionOptions,
    RecursionTrace,
)
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.agents.transcript_agent import TranscriptAgent
from src.config import ConfigError, load_settings
from src.dashboard.theme import dark_style_block
from src.rag.context import MultiTranscriptRagContextProvider, RagTranscriptContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel, cosine_similarity
from src.rag.eval import estimate_tokens
from src.rag.indexing import RagIndexer
from src.rag.references import youtube_timestamp_url
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.rag.summaries import TranscriptSummaryStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.youtube import extract_video_id


DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=3hk7nO_q0a8"
DEFAULT_QUESTION = (
    "what does this video say  for capital gains tax, is it being grandfathered "
    "or every now under new rules, does that mean if I sell before 30 June 2027 "
    "I can still access 50% discount "
)


@dataclass(frozen=True)
class EvaluationRun:
    name: str
    input_type: str
    answer: str
    context_text: str
    retrieved_chunks: list[Any]
    source_url: str | None = None
    recursion: RecursionTrace | None = None
    subtopics: list[FollowupSubtopic] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.context_text)

    @property
    def total_llm_calls(self) -> int:
        if self.recursion:
            return sum(s.llm_calls for s in self.recursion.stages)
        return 1

    @property
    def terminated_reason(self) -> str | None:
        return self.recursion.terminated_reason if self.recursion else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description=(
            "Compare raw single-transcript, RAG single-transcript, and RAG "
            "all-transcripts answers."
        ),
    )
    parser.add_argument("--url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("dashboard/evaluation.html"))
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(
            source_url=args.url,
            question=args.question,
            top_k=args.top_k,
        )
    except (ConfigError, Exception) as exc:
        parser.exit(1, f"Error: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report["html"], encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report["json"], indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote {args.output}")
    return 0


def run_evaluation(
    source_url: str = DEFAULT_VIDEO_URL,
    question: str = DEFAULT_QUESTION,
    top_k: int | None = None,
) -> dict[str, Any]:
    settings = load_settings(require_keys=True)
    video_id = extract_video_id(source_url)
    resolved_top_k = top_k or settings.rag_top_k

    fetcher = SuperdataTranscriptFetcher(
        settings.superdata_api_key,
        timeout_seconds=settings.supadata_timeout_seconds,
        poll_interval_seconds=settings.supadata_poll_interval_seconds,
        max_poll_seconds=settings.supadata_max_poll_seconds,
    )
    raw_store = RawTranscriptStore(
        settings.chroma_path,
        fetcher=fetcher,
        collection_name=settings.raw_transcript_collection,
    )
    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    chunk_store = TranscriptChunkStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        collection_name=settings.chunk_collection,
    )
    indexer = RagIndexer(
        raw_store=raw_store,
        chunk_store=chunk_store,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )

    summary_store = TranscriptSummaryStore(
        settings.chroma_path,
        embedding_model=embedding_model,
        embedding_model_name=settings.embedding_model,
        raw_store=raw_store,
        collection_name=settings.transcript_summary_collection,
    )

    raw_agent = TranscriptAgent.from_settings(
        settings,
        RawTranscriptContextProvider(raw_store, fetcher),
    )
    rag_single_agent = TranscriptAgent.from_settings(
        settings,
        RagTranscriptContextProvider(
            raw_store=raw_store,
            chunk_store=chunk_store,
            indexer=indexer,
            top_k=resolved_top_k,
        ),
    )
    rag_all_agent = RagTranscriptAgent.from_settings(
        settings,
        MultiTranscriptRagContextProvider(
            raw_store=raw_store,
            chunk_store=chunk_store,
            indexer=indexer,
            summary_store=summary_store,
        ),
    )

    recursion_options = RecursionOptions(
        max_depth=settings.rag_max_depth,
        max_followups=settings.rag_max_followups,
        followup_top_k=settings.rag_followup_top_k,
        novelty_min_chunks=settings.rag_novelty_min_chunks,
        max_total_followups=settings.rag_max_total_followups,
    )

    raw_answer = raw_agent.answer(
        QuestionRequest(video_id=video_id, source_url=source_url, question=question)
    )
    rag_single_answer = rag_single_agent.answer(
        QuestionRequest(video_id=video_id, source_url=source_url, question=question)
    )

    rag_all_answer = rag_all_agent.answer(
        RagQuestionRequest(question=question, top_k=resolved_top_k)
    )
    rag_all_context = rag_all_agent.last_context

    rag_all_filtered_answer = rag_all_agent.answer(
        RagQuestionRequest(
            question=question,
            top_k=resolved_top_k,
            filter_transcripts=True,
            transcript_filter_top_k=settings.transcript_filter_top_k,
            transcript_filter_min_score=settings.transcript_filter_min_score,
        )
    )
    rag_all_filtered_context = rag_all_agent.last_context

    rag_recursive_answer = rag_all_agent.answer(
        RagQuestionRequest(
            question=question,
            top_k=resolved_top_k,
            recursive=True,
            recursion_options=recursion_options,
        )
    )
    rag_recursive_context = rag_all_agent.last_context

    rag_recursive_filtered_answer = rag_all_agent.answer(
        RagQuestionRequest(
            question=question,
            top_k=resolved_top_k,
            recursive=True,
            filter_transcripts=True,
            transcript_filter_top_k=settings.transcript_filter_top_k,
            transcript_filter_min_score=settings.transcript_filter_min_score,
            recursion_options=recursion_options,
        )
    )
    rag_recursive_filtered_context = rag_all_agent.last_context

    if (
        raw_agent.last_context is None
        or rag_single_agent.last_context is None
        or rag_all_context is None
        or rag_all_filtered_context is None
        or rag_recursive_context is None
        or rag_recursive_filtered_context is None
    ):
        raise RuntimeError("Evaluation did not capture all context payloads")

    runs = [
        EvaluationRun(
            name="raw_single",
            input_type="raw",
            source_url=source_url,
            answer=raw_answer.answer,
            context_text=raw_agent.last_context.context_text or "",
            retrieved_chunks=[],
        ),
        EvaluationRun(
            name="rag_single",
            input_type="rag single",
            source_url=source_url,
            answer=rag_single_answer.answer,
            context_text=rag_single_agent.last_context.context_text or "",
            retrieved_chunks=rag_single_agent.last_context.retrieved_chunks or [],
        ),
        EvaluationRun(
            name="rag_all",
            input_type="rag all",
            answer=rag_all_answer.answer,
            context_text=rag_all_context.context_text or "",
            retrieved_chunks=rag_all_context.retrieved_chunks or [],
            subtopics=rag_all_answer.subtopics,
            recursion=rag_all_answer.recursion,
        ),
        EvaluationRun(
            name="rag_all_filtered",
            input_type="rag all filtered",
            answer=rag_all_filtered_answer.answer,
            context_text=rag_all_filtered_context.context_text or "",
            retrieved_chunks=rag_all_filtered_context.retrieved_chunks or [],
            subtopics=rag_all_filtered_answer.subtopics,
            recursion=rag_all_filtered_answer.recursion,
        ),
        EvaluationRun(
            name="rag_recursive",
            input_type="rag recursive",
            answer=rag_recursive_answer.answer,
            context_text=rag_recursive_context.context_text or "",
            retrieved_chunks=rag_recursive_context.retrieved_chunks or [],
            subtopics=rag_recursive_answer.subtopics,
            recursion=rag_recursive_answer.recursion,
        ),
        EvaluationRun(
            name="rag_recursive_filtered",
            input_type="rag recursive filtered",
            answer=rag_recursive_filtered_answer.answer,
            context_text=rag_recursive_filtered_context.context_text or "",
            retrieved_chunks=rag_recursive_filtered_context.retrieved_chunks or [],
            subtopics=rag_recursive_filtered_answer.subtopics,
            recursion=rag_recursive_filtered_answer.recursion,
        ),
    ]
    embeddings = embedding_model.embed_documents([run.answer for run in runs])
    similarities = _pairwise_similarities(runs, embeddings)
    payload = _json_payload(
        question=question,
        source_url=source_url,
        top_k=resolved_top_k,
        runs=runs,
        similarities=similarities,
    )
    return {
        "html": render_html_report(
            question=question,
            source_url=source_url,
            top_k=resolved_top_k,
            runs=runs,
            similarities=similarities,
        ),
        "json": payload,
    }


def _pairwise_similarities(
    runs: list[EvaluationRun],
    embeddings: list[list[float]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for left_index, left in enumerate(runs):
        for right_index in range(left_index + 1, len(runs)):
            right = runs[right_index]
            values[f"{left.name}__{right.name}"] = cosine_similarity(
                embeddings[left_index], embeddings[right_index]
            )
    return values


def render_html_report(
    question: str,
    source_url: str,
    top_k: int,
    runs: list[EvaluationRun],
    similarities: dict[str, float],
) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Transcript Agent Evaluation</title>",
            *dark_style_block(),
            "</head>",
            "<body>",
            "<header><h1>Transcript Agent Evaluation</h1></header>",
            "<main>",
            f"<p><strong>Question:</strong> {html.escape(question)}</p>",
            f"<p><strong>Single transcript URL:</strong> {html.escape(source_url)}</p>",
            f"<p><strong>RAG top K:</strong> {top_k}</p>",
            "<h2>Summary</h2>",
            _summary_table(runs),
            "<h2>Pairwise Similarity</h2>",
            _similarity_table(similarities),
            "<h2>Answers</h2>",
            *[_run_section(run) for run in runs],
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _summary_table(runs: list[EvaluationRun]) -> str:
    rows = [
        "<tr><th>Run</th><th>Transcript input type</th><th>Filter</th>"
        "<th>Token estimate</th><th>Retrieved chunks</th><th>Answer chars</th>"
        "<th>LLM calls</th><th>Terminated</th></tr>"
    ]
    for run in runs:
        rows.append(
            "<tr>"
            f"<td>{html.escape(run.name)}</td>"
            f"<td>{html.escape(run.input_type)}</td>"
            f"<td>{html.escape(run.source_url or 'all indexed transcripts')}</td>"
            f"<td class=\"metric\">{run.token_estimate}</td>"
            f"<td class=\"metric\">{len(run.retrieved_chunks)}</td>"
            f"<td class=\"metric\">{len(run.answer)}</td>"
            f"<td class=\"metric\">{run.total_llm_calls}</td>"
            f"<td>{html.escape(run.terminated_reason or '—')}</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _similarity_table(similarities: dict[str, float]) -> str:
    rows = ["<tr><th>Pair</th><th>Embedding cosine similarity</th></tr>"]
    for pair, score in similarities.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(pair)}</td>"
            f"<td class=\"metric\">{score:.3f}</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _run_section(run: EvaluationRun) -> str:
    chunks = "\n".join(
        _chunk_details(index, chunk)
        for index, chunk in enumerate(run.retrieved_chunks, 1)
    )
    if not chunks:
        chunks = "<p>No retrieved chunks for raw transcript input.</p>"
    parts = [
        "<article>",
        f"<h3>{html.escape(run.name)} ({html.escape(run.input_type)})</h3>",
        f"<p><strong>Filter:</strong> {html.escape(run.source_url or 'all indexed transcripts')}</p>",
        f"<p><strong>Token estimate:</strong> <span class=\"metric\">{run.token_estimate}</span></p>",
        "<h4>Answer</h4>",
        f"<pre>{html.escape(run.answer)}</pre>",
        "<h4>Retrieved chunks</h4>",
        chunks,
    ]
    if run.recursion is not None:
        parts.append(_recursion_trace_section(run.recursion))
    parts.append("</article>")
    return "\n".join(parts)


def _recursion_trace_section(recursion: RecursionTrace) -> str:
    stage_rows = "".join(
        f"<tr><td>{html.escape(s.name)}</td>"
        f"<td class=\"metric\">{s.llm_calls}</td>"
        f"<td class=\"metric\">{s.retrievals}</td></tr>"
        for s in recursion.stages
    )
    stage_table = (
        "<table><tr><th>Stage</th><th>LLM calls</th><th>Retrievals</th></tr>"
        + stage_rows
        + "</table>"
    )
    parts = [
        "<h4>Recursion trace</h4>",
        stage_table,
        f"<p><strong>Terminated:</strong> {html.escape(recursion.terminated_reason)}</p>",
        f"<p><strong>Follow-ups proposed:</strong> {recursion.total_followups_proposed}"
        f" &nbsp;|&nbsp; <strong>executed:</strong> {recursion.total_followups_executed}</p>",
    ]
    if recursion.subtopic_answers:
        parts.append("<h4>Subtopic drill-downs</h4>")
        for sa in recursion.subtopic_answers:
            parts.append(
                "<details>"
                f"<summary>{sa.subtopic_index}. {html.escape(sa.topic)}</summary>"
                f"<p><strong>Follow-up query:</strong> {html.escape(sa.followup_query)}</p>"
                f"<pre>{html.escape(sa.answer)}</pre>"
                "</details>"
            )
    elif recursion.subtopic_evidence:
        parts.append("<h4>Proposed follow-ups</h4>")
        for ev in recursion.subtopic_evidence:
            label = f"{ev.subtopic_index}. {ev.subtopic.topic} [{ev.outcome}]"
            parts.append(
                "<details>"
                f"<summary>{html.escape(label)}</summary>"
                f"<p><strong>Query:</strong> {html.escape(ev.subtopic.followup_query)}</p>"
                "</details>"
            )
    return "\n".join(parts)


def _chunk_details(index: int, chunk) -> str:
    timestamp_url = youtube_timestamp_url(str(chunk.source_url), chunk.start_seconds)
    summary = (
        f"[{index}] {chunk.video_id} "
        f"{chunk.start_seconds}-{chunk.end_seconds}s "
        f"score={chunk.score}"
    )
    return "\n".join(
        [
            "<details>",
            f"<summary>{html.escape(summary)}</summary>",
            f'<p><a href="{html.escape(timestamp_url)}">Open video at timestamp</a></p>',
            f"<p>chunk_index={chunk.chunk_index}</p>",
            f"<pre>{html.escape(chunk.text)}</pre>",
            "</details>",
        ]
    )


def _json_payload(
    question: str,
    source_url: str,
    top_k: int,
    runs: list[EvaluationRun],
    similarities: dict[str, float],
) -> dict[str, Any]:
    return {
        "eval_name": "raw_vs_rag_single_vs_rag_all",
        "question": question,
        "source_url": source_url,
        "top_k": top_k,
        "runs": [
            {
                "name": run.name,
                "input_type": run.input_type,
                "source_url": run.source_url,
                "answer": run.answer,
                "token_estimate": run.token_estimate,
                "total_llm_calls": run.total_llm_calls,
                "terminated_reason": run.terminated_reason,
                "retrieved_chunks": [
                    {
                        "rank": index,
                        "score": chunk.score,
                        "video_id": chunk.video_id,
                        "source_url": str(chunk.source_url),
                        "timestamp_url": youtube_timestamp_url(
                            str(chunk.source_url), chunk.start_seconds
                        ),
                        "start_seconds": chunk.start_seconds,
                        "end_seconds": chunk.end_seconds,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    }
                    for index, chunk in enumerate(run.retrieved_chunks, 1)
                ],
                "recursion": run.recursion.model_dump(mode="json") if run.recursion else None,
                "subtopics": [s.model_dump(mode="json") for s in run.subtopics],
            }
            for run in runs
        ],
        "similarities": similarities,
    }


if __name__ == "__main__":
    raise SystemExit(main())
