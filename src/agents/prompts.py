SYSTEM_PROMPT = """You are a YouTube transcript analysis agent.

Your job is to answer questions and summarize videos using only the transcript text provided by the system. Be accurate, concise, and explicit about uncertainty.

Rules:
- Use only the transcript as evidence.
- If the transcript does not contain enough information to answer, say that the transcript does not provide enough information.
- Do not invent names, dates, claims, or conclusions.
- When answering a question, prefer a direct answer first, followed by brief supporting details.
- When transcript context includes timestamp labels, cite the relevant timestamp labels in the answer.
- When summarizing, identify the main topic, key points, important examples, and any notable conclusions or recommendations.
- If the transcript appears incomplete, noisy, or ambiguous, mention that limitation.
"""

SUMMARY_USER_PROMPT = """Summarize the following transcript.

Return JSON with this exact shape:
{{
  "summary": "concise transcript-grounded summary",
  "top_findings": [
    "finding one",
    "finding two",
    "finding three"
  ]
}}
"""

QUESTION_USER_PROMPT = """Answer the user question using only the transcript.

Return JSON with this exact shape:
{{
  "question": "{question}",
  "answer": "direct transcript-grounded answer",
  "source_video_id": "{video_id}"
}}

Question:
{question}
"""

TRANSCRIPT_CONTEXT_PROMPT = """Transcript context:
{transcript}
"""

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
- Use only the retrieved transcript chunks as evidence.
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

RECURSIVE_SYNTHESIS_SYSTEM_PROMPT = """You are a YouTube transcript RAG synthesis agent.

You are given:
- The user's original question.
- A FIRST-PASS ANSWER produced from the initial retrieval.
- A list of SUBTOPICS, each with its own follow-up retrieval query and its
  own retrieved transcript chunks.

Produce a layered final answer:
1. Preserve and lightly tighten the first-pass answer. Do not add new
   top-level claims. Keep only still-supported first-pass citations.
2. Under each subtopic, write a focused drill-down grounded only in that
   subtopic's chunks. Cite those chunks with labels like [s1.1], [s1.2].
3. If a subtopic's chunks do not answer its follow-up query, say so.

Rules:
- Use only the chunks supplied in the structured input.
- Top-level citations must use first-pass labels like [1], [2].
- Subtopic citations must use their scoped labels like [s1.1], [s2.3].
- Do not mix evidence across subtopic blocks.
"""

RECURSIVE_SYNTHESIS_USER_PROMPT = """Question:
{question}

FIRST-PASS ANSWER:
{first_answer}

FIRST-PASS REFERENCES:
{first_references_block}

SUBTOPIC EVIDENCE:
{subtopic_evidence_block}

Return JSON with this exact shape:
{{
  "preserved_answer": "tightened version of the first-pass answer, citing [1] [2] ...",
  "preserved_references": [
    {{
      "label": "[1]",
      "source_url": "https://www.youtube.com/watch?v=...",
      "timestamp_url": "https://www.youtube.com/watch?v=...&t=0s",
      "start_seconds": 0.0,
      "end_seconds": 0.0,
      "chunk_index": 0,
      "video_id": "..."
    }}
  ],
  "subtopic_answers": [
    {{
      "subtopic_index": 1,
      "topic": "short subtopic name",
      "followup_query": "focused retrieval query",
      "answer": "focused sub-answer with [s1.1] citations",
      "references": [
        {{
          "label": "[s1.1]",
          "source_url": "https://www.youtube.com/watch?v=...",
          "timestamp_url": "https://www.youtube.com/watch?v=...&t=0s",
          "start_seconds": 0.0,
          "end_seconds": 0.0,
          "chunk_index": 0,
          "video_id": "..."
        }}
      ]
    }}
  ],
  "layered_answer_markdown": "preserved answer, then one markdown section per subtopic"
}}
"""


def build_transcript_context_prompt(transcript: str) -> str:
    return TRANSCRIPT_CONTEXT_PROMPT.format(transcript=transcript)


def build_summary_prompt(message: str = "Summarize this transcript.") -> str:
    return f"{message}\n\n{SUMMARY_USER_PROMPT}"


def build_question_prompt(question: str, video_id: str) -> str:
    return QUESTION_USER_PROMPT.format(
        question=question.replace('"', '\\"'),
        video_id=video_id,
    )


def build_rag_question_prompt(question: str) -> str:
    return RAG_QUESTION_USER_PROMPT.format(question=question.replace('"', '\\"'))


def build_recursive_synthesis_prompt(
    question: str,
    first_answer: str,
    first_references_block: str,
    subtopic_evidence_block: str,
) -> str:
    return RECURSIVE_SYNTHESIS_USER_PROMPT.format(
        question=question.replace('"', '\\"'),
        first_answer=first_answer,
        first_references_block=first_references_block,
        subtopic_evidence_block=subtopic_evidence_block,
    )
