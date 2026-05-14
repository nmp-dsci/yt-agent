# CLAUDE.md

## Project

You are working on a YouTube transcript agent prototype.

The intended system has two main agent paths:

1. A direct transcript LLM agent for Q&A and summarization.
2. A RAG agent for transcript retrieval and comparison.

Longer-term work includes multi-transcript support, trend tracking across related videos, evaluation datasets, accuracy scoring, and prompt optimization.

## Instructions for Claude

- Read `readme.md` and `AGENTS.md` before making changes.
- Keep implementations scoped to the requested task.
- Do not introduce unnecessary infrastructure.
- Document setup, run commands, and new dependencies in `readme.md`.
- Keep secrets out of source control. Use environment variables for API keys.
- Prefer simple, testable modules over tightly coupled agent flows.

## Suggested Module Boundaries

Use these boundaries when adding source code, unless the project has already established a different structure:

- `transcripts`: fetching, loading, cleaning, and normalizing transcript text.
- `agents`: prompt construction, Q&A behavior, summarization behavior.
- `rag`: chunking, embeddings, indexing, retrieval, and answer synthesis.
- `evals`: test questions, reference answers, scoring, and prompt experiments.

## Testing Expectations

- Add tests for new parsing, chunking, retrieval, and scoring logic.
- Mock external YouTube, LLM, and embedding provider calls.
- Prefer small fixtures over large raw transcript dumps.

## Style

- Use clear names for agent components and pipeline steps.
- Keep prompts versioned or easy to compare when prompt optimization begins.
- Avoid broad rewrites unless they are required to support the requested change.
