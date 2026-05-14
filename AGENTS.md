# AGENTS.md

## Project Context

This project is a YouTube transcript agent prototype.

Current goals from `readme.md`:

1. Read YouTube transcripts.
2. Build an LLM transcript agent that can answer questions and summarize a transcript.
3. Build RAG pipelines for comparison.
4. Build a second RAG-based agent.

Future goals:

- Support multiple transcripts in one agent workflow.
- Track trends across transcripts in the same field.
- Build an evaluation set and score answer accuracy.
- Optimize LLM system prompts to improve accuracy.

## Working Guidelines

- Keep changes small and easy to inspect.
- Prefer clear, boring implementations over premature abstractions.
- Do not add frameworks, package managers, databases, or external services unless the task requires them.
- When adding dependencies, document why they are needed and how to install them.
- If introducing API calls, keep secrets in environment variables and never commit keys or tokens.
- Keep transcript ingestion, agent logic, RAG retrieval, and evaluation code separated when those modules are added.
- Update `readme.md` when commands, setup steps, or project behavior changes.

## Expected Architecture Direction

When the codebase is created, prefer a structure similar to:

```text
src/
  transcripts/   # YouTube transcript loading, parsing, cleaning
  agents/        # Transcript Q&A and summarization agents
  rag/           # Chunking, embeddings, vector search, retrieval chains
  evals/         # Evaluation sets, scoring, prompt experiments
tests/
```

This is guidance, not a requirement. Follow the actual project structure once it exists.

## Quality Bar

- Add focused tests for transcript parsing, chunking, retrieval behavior, and evaluation scoring once code exists.
- Prefer deterministic tests for data processing.
- Mock external LLM, embedding, and YouTube calls in tests.
- Include example transcripts or fixtures only when licensing and privacy are clear.

## Agent Notes

- This directory may not be a git repository. Check before using git commands.
- Start by reading `readme.md` and this file.
- Preserve the project goal: compare direct transcript LLM workflows with RAG workflows.
- Avoid committing generated transcript data unless explicitly requested.
