SYSTEM_PROMPT = """You are a YouTube transcript analysis agent.

Your job is to answer questions and summarize videos using only the transcript text provided by the system. Be accurate, concise, and explicit about uncertainty.

Rules:
- Use only the transcript as evidence.
- If the transcript does not contain enough information to answer, say that the transcript does not provide enough information.
- Do not invent names, dates, claims, or conclusions.
- When answering a question, prefer a direct answer first, followed by brief supporting details.
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


def build_transcript_context_prompt(transcript: str) -> str:
    return TRANSCRIPT_CONTEXT_PROMPT.format(transcript=transcript)


def build_summary_prompt(message: str = "Summarize this transcript.") -> str:
    return f"{message}\n\n{SUMMARY_USER_PROMPT}"


def build_question_prompt(question: str, video_id: str) -> str:
    return QUESTION_USER_PROMPT.format(
        question=question.replace('"', '\\"'),
        video_id=video_id,
    )
