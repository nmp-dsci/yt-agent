from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from src.agents.context import TranscriptContext
from src.agents.models import RagQuestionRequest
from src.agents.rag_transcript_agent import RagTranscriptAgent
from src.rag.models import RetrievedChunk
from src.transcripts.models import Transcript


class FakeLlm:
    def __init__(self, response: str | list[str]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.messages = None
        self.calls = []

    def invoke(self, messages):
        self.messages = messages
        self.calls.append(messages)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return AIMessage(content=response)


class FakeProvider:
    def __init__(self) -> None:
        self.calls = []

    def get_context(
        self,
        question: str,
        source_url: str | None = None,
        top_k: int = 10,
        filter_transcripts: bool = False,
        transcript_filter_top_k: int = 5,
        transcript_filter_min_score: float = 0.25,
    ):
        self.calls.append(
            (
                question,
                source_url,
                top_k,
                filter_transcripts,
                transcript_filter_top_k,
                transcript_filter_min_score,
            )
        )
        transcript = Transcript(
            video_id="all",
            url="https://www.youtube.com/watch?v=abc",
            raw_text="chunk text",
            fetched_at=datetime.now(timezone.utc),
        )
        chunk = RetrievedChunk(
            transcript_id="raw_transcript:abc",
            video_id="abc",
            source_url="https://www.youtube.com/watch?v=abc",
            chunk_index=4,
            text="capital gains tax",
            start_seconds=593,
            end_seconds=665,
            segment_count=1,
        )
        return TranscriptContext(
            transcript=transcript,
            cache_status="hit",
            context_text="[1] video=abc url=https://www.youtube.com/watch?v=abc&t=593s\ncapital gains tax",
            context_mode="rag",
            retrieved_chunks=[chunk],
            top_k=top_k,
        )


def test_rag_transcript_agent_answers_and_backfills_references() -> None:
    llm = FakeLlm('{"question": "q", "answer": "answer from chunk [1]"}')
    provider = FakeProvider()
    agent = RagTranscriptAgent(llm, provider)

    answer = agent.answer(
        RagQuestionRequest(
            question="q",
            source_url="https://www.youtube.com/watch?v=abc",
            top_k=3,
        )
    )

    assert answer.answer == "answer from chunk [1]"
    assert answer.references[0].timestamp_url.unicode_string().endswith("t=593s")
    assert provider.calls == [
        ("q", "https://www.youtube.com/watch?v=abc", 3, False, 5, 0.25)
    ]
    assert "retrieved transcript chunks" in llm.messages[0].content


def test_single_hop_surfaces_followups_without_extra_retrieval() -> None:
    llm = FakeLlm(
        """
        {
          "question": "q",
          "answer": "answer from chunk [1]",
          "followups_requested": true,
          "subtopics": [
            {
              "topic": "detail",
              "rationale": "thin evidence",
              "followup_query": "specific detail query",
              "confidence": 0.8
            }
          ]
        }
        """
    )
    provider = FakeProvider()
    agent = RagTranscriptAgent(llm, provider)

    answer = agent.answer(RagQuestionRequest(question="q", top_k=3))

    assert answer.followups_requested is True
    assert answer.subtopics[0].followup_query == "specific detail query"
    assert answer.recursion is None
    assert len(provider.calls) == 1
    assert len(llm.calls) == 1


def test_recursive_retrieves_followups_and_synthesizes_answer() -> None:
    first_response = """
    {
      "question": "q",
      "answer": "first answer [1]",
      "references": [
        {
          "label": "[1]",
          "source_url": "https://www.youtube.com/watch?v=abc",
          "timestamp_url": "https://www.youtube.com/watch?v=abc&t=593s",
          "start_seconds": 593,
          "end_seconds": 665,
          "chunk_index": 4,
          "video_id": "abc"
        }
      ],
      "followups_requested": true,
      "subtopics": [
        {
          "topic": "detail",
          "rationale": "thin evidence",
          "followup_query": "specific detail query",
          "confidence": 0.8
        }
      ]
    }
    """
    synthesis_response = """
    {
      "preserved_answer": "first answer [1]",
      "preserved_references": [
        {
          "label": "[1]",
          "source_url": "https://www.youtube.com/watch?v=abc",
          "timestamp_url": "https://www.youtube.com/watch?v=abc&t=593s",
          "start_seconds": 593,
          "end_seconds": 665,
          "chunk_index": 4,
          "video_id": "abc"
        }
      ],
      "subtopic_answers": [
        {
          "subtopic_index": 1,
          "topic": "detail",
          "followup_query": "specific detail query",
          "answer": "detail answer [s1.1]",
          "references": [
            {
              "label": "[s1.1]",
              "source_url": "https://www.youtube.com/watch?v=abc",
              "timestamp_url": "https://www.youtube.com/watch?v=abc&t=700s",
              "start_seconds": 700,
              "end_seconds": 760,
              "chunk_index": 5,
              "video_id": "abc"
            }
          ]
        }
      ],
      "layered_answer_markdown": "first answer [1]\\n\\n## detail\\ndetail answer [s1.1]"
    }
    """

    class NovelProvider(FakeProvider):
        def get_context(self, *args, **kwargs):
            context = super().get_context(*args, **kwargs)
            if len(self.calls) > 1:
                chunk = context.retrieved_chunks[0].model_copy(
                    update={
                        "chunk_index": 5,
                        "text": "follow-up detail",
                        "start_seconds": 700,
                        "end_seconds": 760,
                    }
                )
                context = replace(
                    context,
                    context_text="[1] 11:40-12:40\nfollow-up detail",
                    retrieved_chunks=[chunk],
                )
            return context

    llm = FakeLlm([first_response, synthesis_response])
    provider = NovelProvider()
    agent = RagTranscriptAgent(llm, provider)

    answer = agent.answer(
        RagQuestionRequest(
            question="q",
            recursive=True,
            recursion_options={"novelty_min_chunks": 1},
        )
    )

    assert answer.answer.startswith("first answer")
    assert answer.recursion is not None
    assert answer.recursion.terminated_reason == "completed"
    assert answer.recursion.total_followups_executed == 1
    assert provider.calls[1][0] == "specific detail query"
    assert len(llm.calls) == 2
