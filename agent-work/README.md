# Agent Work

Use this directory to store markdown files that describe work for coding agents to implement.

## Structure

```text
agent-work/
  plans/       # High-level implementation plans and task breakdowns
  prps/        # Product Requirements Prompts or detailed implementation prompts
  specs/       # Behavior, API, data, and architecture specifications
  templates/   # Reusable markdown templates
  archive/     # Completed or superseded agent work documents
```

## Naming

Prefer dated, descriptive filenames:

```text
YYYY-MM-DD-short-topic.md
```

Examples:

```text
2026-05-14-transcript-ingestion-plan.md
2026-05-14-direct-transcript-agent-prp.md
2026-05-14-rag-chunking-spec.md
```

## Document Status

Add a status near the top of each document:

- `draft`: Still being shaped.
- `ready`: Ready for an agent to implement.
- `in-progress`: An agent is actively implementing it.
- `done`: Implemented and verified.
- `superseded`: Replaced by a newer document.

Move completed or superseded documents to `archive/` when they are no longer useful as active context.

## Agent Handoff Checklist

Before giving a document to a coding agent, make sure it includes:

- Clear goal and scope.
- Explicit non-goals.
- Expected files or modules to touch, if known.
- Acceptance criteria.
- Test or verification expectations.
- Any relevant constraints from `AGENTS.md`.
