from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from src.agents.context import TranscriptContext
from src.agents.models import (
    FollowupSubtopic,
    RagAnswerReference,
    RagQuestionRequest,
    RagTranscriptAnswer,
    RecursionOptions,
    RecursionStage,
    RecursionTrace,
    SubtopicAnswer,
    SubtopicEvidence,
)
from src.agents.prompts import (
    RECURSIVE_SYNTHESIS_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    build_rag_question_prompt,
    build_recursive_synthesis_prompt,
    build_transcript_context_prompt,
)
from src.config import Settings
from src.rag.context import MultiTranscriptRagContextProvider
from src.rag.embeddings import HuggingFaceEmbeddingModel
from src.rag.indexing import RagIndexer
from src.rag.references import youtube_timestamp_url
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore
from src.transcripts.fetcher import SuperdataTranscriptFetcher


logger = logging.getLogger(__name__)


class RagContextTooLongError(RuntimeError):
    pass


class ChatModel(Protocol):
    def invoke(self, messages: list[SystemMessage | HumanMessage]) -> object:
        ...


class _FirstPassResult(BaseModel):
    question: str
    answer: str
    references: list[RagAnswerReference] = Field(default_factory=list)
    subtopics: list[FollowupSubtopic] = Field(default_factory=list)
    followups_requested: bool = False
    answer_confidence: float | None = None
    parse_degraded: bool = False


class _SynthesisResult(BaseModel):
    preserved_answer: str
    preserved_references: list[RagAnswerReference] = Field(default_factory=list)
    subtopic_answers: list[SubtopicAnswer] = Field(default_factory=list)
    layered_answer_markdown: str


@dataclass
class _RetrievalResult:
    context: TranscriptContext
    context_text: str


class RagTranscriptAgent:
    def __init__(
        self,
        llm: ChatModel,
        context_provider: MultiTranscriptRagContextProvider,
        max_context_chars: int = 40_000,
    ) -> None:
        self.llm = llm
        self.context_provider = context_provider
        self.max_context_chars = max_context_chars
        self.last_context: TranscriptContext | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        context_provider: MultiTranscriptRagContextProvider | None = None,
    ) -> "RagTranscriptAgent":
        kwargs: dict[str, object] = {
            "api_key": settings.deepseek_api_key,
            "model": settings.deepseek_model,
        }
        if settings.deepseek_base_url:
            kwargs["base_url"] = settings.deepseek_base_url
        if context_provider is None:
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
            context_provider = MultiTranscriptRagContextProvider(
                raw_store=raw_store,
                chunk_store=chunk_store,
                indexer=indexer,
            )
        return cls(ChatOpenAI(**kwargs), context_provider)

    def answer(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        if request.recursive:
            return self._answer_recursive(request)
        return self._answer_single_hop(request)

    def _answer_single_hop(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        retrieval = self._retrieve(
            question=request.question,
            source_url=str(request.source_url) if request.source_url else None,
            top_k=request.top_k,
            filter_transcripts=request.filter_transcripts,
            transcript_filter_top_k=request.transcript_filter_top_k,
            transcript_filter_min_score=request.transcript_filter_min_score,
        )
        first = self._invoke_first_pass(request.question, retrieval.context_text)
        references = first.references or _fallback_references(
            first.answer, retrieval.context
        )
        return RagTranscriptAnswer(
            question=request.question,
            answer=first.answer,
            references=references,
            subtopics=first.subtopics,
            followups_requested=first.followups_requested,
            answer_confidence=first.answer_confidence,
        )

    def _answer_recursive(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        options = request.recursion_options or RecursionOptions()
        if options.max_depth > 1:
            logger.warning(
                "Recursive RAG max_depth=%s requested; S6 currently treats it as 1",
                options.max_depth,
            )
        max_depth = max(0, min(options.max_depth, 1))
        max_followups = max(1, min(options.max_followups, 6))
        max_total_followups = options.max_total_followups
        if max_total_followups is None:
            max_total_followups = max_depth * max_followups
        max_total_followups = max(0, max_total_followups)

        retrieval = self._retrieve(
            question=request.question,
            source_url=str(request.source_url) if request.source_url else None,
            top_k=request.top_k,
            filter_transcripts=request.filter_transcripts,
            transcript_filter_top_k=request.transcript_filter_top_k,
            transcript_filter_min_score=request.transcript_filter_min_score,
        )
        first = self._invoke_first_pass(request.question, retrieval.context_text)
        first_references = first.references or _fallback_references(
            first.answer, retrieval.context
        )
        proposed_count = len(first.subtopics)
        if (
            max_depth == 0
            or first.parse_degraded
            or not first.followups_requested
            or not first.subtopics
        ):
            reason = "response_parse_degraded" if first.parse_degraded else "no_followups_requested"
            return self._first_pass_answer_with_trace(
                request=request,
                first=first,
                references=first_references,
                reason=reason,
                stages=[RecursionStage(name="first_pass", llm_calls=1, retrievals=1)],
                evidence=[],
                proposed_count=proposed_count,
                executed_count=0,
            )

        selected, duplicate_evidence = _select_followups(
            request.question, first.subtopics, max_followups
        )
        if selected and max_total_followups == 0:
            return self._first_pass_answer_with_trace(
                request=request,
                first=first,
                references=first_references,
                reason="max_total_followups_reached",
                stages=[
                    RecursionStage(name="first_pass", llm_calls=1, retrievals=1),
                    RecursionStage(name="fan_out", llm_calls=0, retrievals=0),
                ],
                evidence=duplicate_evidence,
                proposed_count=proposed_count,
                executed_count=0,
            )
        selected = selected[:max_total_followups]
        if not selected:
            return self._first_pass_answer_with_trace(
                request=request,
                first=first,
                references=first_references,
                reason="all_followups_filtered",
                stages=[
                    RecursionStage(name="first_pass", llm_calls=1, retrievals=1),
                    RecursionStage(name="fan_out", llm_calls=0, retrievals=0),
                ],
                evidence=duplicate_evidence,
                proposed_count=proposed_count,
                executed_count=0,
            )

        seen_chunk_keys = {
            _chunk_key(chunk) for chunk in retrieval.context.retrieved_chunks
        }
        evidence: list[SubtopicEvidence] = [*duplicate_evidence]
        merged_contexts = [retrieval.context]
        followup_top_k = options.followup_top_k or request.top_k
        retrievals = 0
        executed_count = 0
        for subtopic_index, subtopic in selected:
            if retrievals >= max_total_followups:
                break
            retrievals += 1
            followup = self._retrieve(
                question=subtopic.followup_query,
                source_url=str(request.source_url) if request.source_url else None,
                top_k=followup_top_k,
                filter_transcripts=request.filter_transcripts,
                transcript_filter_top_k=request.transcript_filter_top_k,
                transcript_filter_min_score=request.transcript_filter_min_score,
            )
            novel_chunks = [
                chunk
                for chunk in followup.context.retrieved_chunks
                if _chunk_key(chunk) not in seen_chunk_keys
            ]
            if len(novel_chunks) < options.novelty_min_chunks:
                evidence.append(
                    SubtopicEvidence(
                        subtopic_index=subtopic_index,
                        subtopic=subtopic,
                        chunks=[],
                        outcome="no_new_evidence",
                    )
                )
                continue
            seen_chunk_keys.update(_chunk_key(chunk) for chunk in novel_chunks)
            evidence.append(
                SubtopicEvidence(
                    subtopic_index=subtopic_index,
                    subtopic=subtopic,
                    chunks=novel_chunks,
                    outcome="merged",
                )
            )
            merged_contexts.append(followup.context)
            executed_count += 1

        self.last_context = _merge_contexts(merged_contexts)
        executed = [item for item in evidence if item.outcome == "merged"]
        stages = [
            RecursionStage(name="first_pass", llm_calls=1, retrievals=1),
            RecursionStage(name="fan_out", llm_calls=0, retrievals=retrievals),
        ]
        if not executed:
            reason = (
                "max_total_followups_reached"
                if max_total_followups == 0
                else "no_new_evidence"
            )
            return self._first_pass_answer_with_trace(
                request=request,
                first=first,
                references=first_references,
                reason=reason,
                stages=stages,
                evidence=evidence,
                proposed_count=proposed_count,
                executed_count=0,
            )

        synthesis = self._invoke_final_synthesis(
            question=request.question,
            first_answer=first.answer,
            first_references=first_references,
            evidence=executed,
        )
        if synthesis is None:
            return self._first_pass_answer_with_trace(
                request=request,
                first=first,
                references=first_references,
                reason="synthesis_parse_degraded",
                stages=stages,
                evidence=evidence,
                proposed_count=proposed_count,
                executed_count=executed_count,
            )
        stages.append(
            RecursionStage(name="final_synthesis", llm_calls=1, retrievals=0)
        )
        references = _sort_references(
            [
                *synthesis.preserved_references,
                *[
                    reference
                    for answer in synthesis.subtopic_answers
                    for reference in answer.references
                ],
            ]
        )
        if not references:
            references = first_references
        return RagTranscriptAnswer(
            question=request.question,
            answer=synthesis.layered_answer_markdown,
            references=references,
            subtopics=first.subtopics,
            followups_requested=first.followups_requested,
            answer_confidence=first.answer_confidence,
            recursion=RecursionTrace(
                stages=stages,
                subtopic_evidence=evidence,
                subtopic_answers=synthesis.subtopic_answers,
                preserved_first_answer=first.answer,
                terminated_reason="completed",
                total_followups_proposed=proposed_count,
                total_followups_executed=executed_count,
            ),
        )

    def _retrieve(
        self,
        question: str,
        source_url: str | None,
        top_k: int,
        filter_transcripts: bool,
        transcript_filter_top_k: int,
        transcript_filter_min_score: float,
    ) -> _RetrievalResult:
        context = self.context_provider.get_context(
            question=question,
            source_url=source_url,
            top_k=top_k,
            filter_transcripts=filter_transcripts,
            transcript_filter_top_k=transcript_filter_top_k,
            transcript_filter_min_score=transcript_filter_min_score,
        )
        self.last_context = context
        context_text = context.context_text or ""
        if len(context_text) > self.max_context_chars:
            raise RagContextTooLongError("Retrieved RAG context is too long")
        return _RetrievalResult(context=context, context_text=context_text)

    def _invoke_first_pass(self, question: str, context_text: str) -> _FirstPassResult:
        content = self._invoke(
            system_prompt=RAG_SYSTEM_PROMPT,
            context_text=context_text,
            user_prompt=build_rag_question_prompt(question),
        )
        try:
            data = _json_object(content)
            data.setdefault("question", question)
            return _FirstPassResult.model_validate(data)
        except (ValueError, ValidationError):
            return _FirstPassResult(
                question=question,
                answer=content.strip(),
                references=[],
                subtopics=[],
                followups_requested=False,
                parse_degraded=True,
            )

    def _invoke_final_synthesis(
        self,
        question: str,
        first_answer: str,
        first_references: list[RagAnswerReference],
        evidence: list[SubtopicEvidence],
    ) -> _SynthesisResult | None:
        first_references_block = _format_references_block(first_references)
        evidence_block = _format_subtopic_evidence(evidence)
        user_prompt = build_recursive_synthesis_prompt(
            question=question,
            first_answer=first_answer,
            first_references_block=first_references_block,
            subtopic_evidence_block=evidence_block,
        )
        if len(user_prompt) > self.max_context_chars:
            raise RagContextTooLongError("Recursive synthesis context is too long")
        content = self._invoke(
            system_prompt=RECURSIVE_SYNTHESIS_SYSTEM_PROMPT,
            context_text="",
            user_prompt=user_prompt,
        )
        try:
            result = _SynthesisResult.model_validate(_json_object(content))
        except (ValueError, ValidationError):
            return None
        if not _synthesis_references_valid(result, first_references, evidence):
            return None
        return result

    def _first_pass_answer_with_trace(
        self,
        request: RagQuestionRequest,
        first: _FirstPassResult,
        references: list[RagAnswerReference],
        reason: str,
        stages: list[RecursionStage],
        evidence: list[SubtopicEvidence],
        proposed_count: int,
        executed_count: int,
    ) -> RagTranscriptAnswer:
        return RagTranscriptAnswer(
            question=request.question,
            answer=first.answer,
            references=references,
            subtopics=first.subtopics,
            followups_requested=first.followups_requested,
            answer_confidence=first.answer_confidence,
            recursion=RecursionTrace(
                stages=stages,
                subtopic_evidence=evidence,
                subtopic_answers=[],
                terminated_reason=reason,
                total_followups_proposed=proposed_count,
                total_followups_executed=executed_count,
            ),
        )

    def _invoke(self, system_prompt: str, context_text: str, user_prompt: str) -> str:
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=system_prompt),
        ]
        if context_text:
            messages.append(SystemMessage(content=build_transcript_context_prompt(context_text)))
        messages.append(HumanMessage(content=user_prompt))
        response = self.llm.invoke(
            messages
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)


def _fallback_references(
    answer_text: str, context: TranscriptContext
) -> list[RagAnswerReference]:
    cited = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", answer_text)
        if match.isdigit()
    }
    chunks = context.retrieved_chunks or []
    if not cited:
        cited = set(range(1, len(chunks) + 1))
    references: list[RagAnswerReference] = []
    for label_index in sorted(cited):
        chunk_index = label_index - 1
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue
        chunk = chunks[chunk_index]
        references.append(
            RagAnswerReference(
                label=f"[{label_index}]",
                source_url=chunk.source_url,
                timestamp_url=youtube_timestamp_url(
                    str(chunk.source_url), chunk.start_seconds
                ),
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                chunk_index=chunk.chunk_index,
                video_id=chunk.video_id,
            )
        )
    return references


def _select_followups(
    question: str,
    subtopics: list[FollowupSubtopic],
    max_followups: int,
) -> tuple[list[tuple[int, FollowupSubtopic]], list[SubtopicEvidence]]:
    selected: list[tuple[int, FollowupSubtopic]] = []
    duplicate_evidence: list[SubtopicEvidence] = []
    seen = [_normalize_query(question)]
    ranked = sorted(
        enumerate(subtopics, 1),
        key=lambda item: item[1].confidence,
        reverse=True,
    )
    for original_index, subtopic in ranked:
        normalized = _normalize_query(subtopic.followup_query)
        if any(_levenshtein_at_most_one(normalized, existing) for existing in seen):
            duplicate_evidence.append(
                SubtopicEvidence(
                    subtopic_index=original_index,
                    subtopic=subtopic,
                    chunks=[],
                    outcome="duplicate_query",
                )
            )
            continue
        seen.append(normalized)
        selected.append((original_index, subtopic))
        if len(selected) >= max_followups:
            break
    return selected, duplicate_evidence


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower()).rstrip("?.!,;:")


def _levenshtein_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    edits = 0
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(left) == len(right):
            i += 1
        j += 1
    return True


def _chunk_key(chunk) -> tuple[str, int]:
    return (chunk.video_id, chunk.chunk_index)


def _merge_contexts(contexts: list[TranscriptContext]) -> TranscriptContext:
    base = contexts[0]
    chunks = []
    selected_transcripts = []
    seen_chunks = set()
    seen_transcripts = set()
    for context in contexts:
        for chunk in context.retrieved_chunks:
            key = _chunk_key(chunk)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            chunks.append(chunk)
        for transcript in context.selected_transcripts:
            key = getattr(transcript, "summary_id", transcript.video_id)
            if key in seen_transcripts:
                continue
            seen_transcripts.add(key)
            selected_transcripts.append(transcript)
    return TranscriptContext(
        transcript=base.transcript,
        cache_status=base.cache_status,
        context_text="\n\n".join(context.context_text or "" for context in contexts),
        context_mode=base.context_mode,
        retrieved_chunks=chunks,
        selected_transcripts=selected_transcripts,
        top_k=base.top_k,
    )


def _format_references_block(references: list[RagAnswerReference]) -> str:
    lines = []
    for reference in references:
        lines.append(
            f"{reference.label} video={reference.video_id} "
            f"url={reference.source_url} timestamp={reference.timestamp_url} "
            f"chunk_index={reference.chunk_index}"
        )
    return "\n".join(lines) or "No first-pass references."


def _format_subtopic_evidence(evidence: list[SubtopicEvidence]) -> str:
    blocks = []
    for item in evidence:
        lines = [
            f"Subtopic {item.subtopic_index}: {item.subtopic.topic}",
            f"rationale: {item.subtopic.rationale}",
            f"followup_query: {item.subtopic.followup_query}",
            f"confidence: {item.subtopic.confidence}",
            f"outcome: {item.outcome}",
        ]
        for chunk_number, chunk in enumerate(item.chunks, 1):
            label = f"[s{item.subtopic_index}.{chunk_number}]"
            lines.extend(
                [
                    f"{label} video={chunk.video_id} url={chunk.source_url} "
                    f"start={chunk.start_seconds} end={chunk.end_seconds} "
                    f"chunk_index={chunk.chunk_index}",
                    chunk.text,
                ]
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _synthesis_references_valid(
    result: _SynthesisResult,
    first_references: list[RagAnswerReference],
    evidence: list[SubtopicEvidence],
) -> bool:
    first_labels = {reference.label for reference in first_references}
    if any(reference.label not in first_labels for reference in result.preserved_references):
        return False
    allowed_subtopic_labels: dict[int, set[str]] = {}
    for item in evidence:
        allowed_subtopic_labels[item.subtopic_index] = {
            f"[s{item.subtopic_index}.{index}]"
            for index, _chunk in enumerate(item.chunks, 1)
        }
    for answer in result.subtopic_answers:
        allowed = allowed_subtopic_labels.get(answer.subtopic_index, set())
        if any(reference.label not in allowed for reference in answer.references):
            return False
    return True


def _sort_references(references: list[RagAnswerReference]) -> list[RagAnswerReference]:
    def sort_key(reference: RagAnswerReference) -> tuple[int, int, int]:
        subtopic = re.match(r"\[s(\d+)\.(\d+)\]", reference.label)
        if subtopic:
            return (1, int(subtopic.group(1)), int(subtopic.group(2)))
        numeric = re.match(r"\[(\d+)\]", reference.label)
        if numeric:
            return (0, int(numeric.group(1)), 0)
        return (2, 0, 0)

    seen = set()
    result = []
    for reference in sorted(references, key=sort_key):
        if reference.label in seen:
            continue
        seen.add(reference.label)
        result.append(reference)
    return result


def _json_object(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {content}") from exc
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    return value
