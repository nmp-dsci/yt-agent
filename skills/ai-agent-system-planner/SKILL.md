---
name: ai-agent-system-planner
description: Use when asked to plan, design, architect, or specify an AI agent system, especially systems involving LangGraph, DeepSeek LLMs, RAG, tools, memory, evaluation, observability, or production rollout. Produces decision-complete engineering plans rather than directly implementing code.
---

# AI Agent System Planner

Use this skill to produce implementation-ready engineering plans for AI agent systems. The default target stack is LangGraph with DeepSeek LLM, while keeping the plan adaptable when the user or repo clearly uses another stack.

## Operating Mode

1. Explore existing repo context before asking questions.
2. Ask only for product intent, constraints, or tradeoffs that cannot be discovered locally.
3. Plan the system before implementation. The final output should be decision-complete enough for another engineer or agent to build without making architecture decisions.
4. Prefer concise, specific plans over broad agent theory.

## Required Discovery

Before drafting the final plan, identify:

- User workflow and primary audience.
- Inputs, outputs, and expected quality bar.
- Existing codebase structure, runtime, dependencies, and deployment assumptions.
- Agent responsibilities and non-goals.
- Data sources, tool/API boundaries, and privacy or security constraints.
- Evals needed to prove the agent works.

## Planning Workflow

### 1. Product Intent

Define:

- The agent's job-to-be-done.
- Who uses it and when.
- Success criteria and failure criteria.
- What is explicitly out of scope.

### 2. Architecture

Specify:

- Agent graph or workflow shape.
- State object and key fields.
- Nodes, tools, routers, and handoff points.
- Model calls and prompt responsibilities.
- Memory strategy, if needed.
- Human review or approval points, if needed.
- Error handling, retries, timeouts, and fallback behavior.

For LangGraph plans, default to:

- Python implementation.
- A typed graph state.
- Separate nodes for ingestion, reasoning, tool use, retrieval, answer generation, and evaluation when those responsibilities exist.
- Conditional edges for routing decisions.
- Tool functions isolated from prompt text and model-provider code.

### 3. Model and Provider Strategy

Default to DeepSeek when no other provider is requested.

Specify:

- Model role and expected capabilities.
- Provider adapter boundary.
- API key environment variable names.
- Temperature or determinism expectations.
- Streaming requirements, if any.
- Cost and latency constraints.

Keep provider-specific code behind a small adapter so the graph can swap DeepSeek for another chat model later.

### 4. Data and RAG

When RAG is relevant, specify:

- Source documents and ingestion path.
- Chunking strategy.
- Embedding provider and vector store choice.
- Metadata fields.
- Retrieval query construction.
- Reranking or filtering, if needed.
- Citation or grounding behavior.

For transcript systems, keep raw transcript loading, transcript cleaning, chunking, retrieval, and answer synthesis as separate responsibilities.

### 5. Evaluation and Observability

Every plan should include:

- Unit tests for deterministic logic.
- Integration tests with mocked model/tool calls.
- Golden-path scenario tests.
- Failure-mode tests.
- Evaluation dataset shape.
- Metrics for answer quality, groundedness, tool success, cost, and latency.
- Logging or tracing points for graph state transitions and external calls.

## Final Plan Format

Use this structure unless the user requests another format:

```markdown
# <Plan Title>

## Summary
- What will be built and why.
- Default stack and main constraints.

## Architecture
- Agent graph/workflow.
- State, nodes, tools, model calls, memory, and failure handling.

## Implementation Steps
- Ordered build steps with clear subsystem boundaries.
- Public interfaces, config, and environment variables.

## Test and Evaluation Plan
- Unit, integration, scenario, and eval coverage.
- Acceptance criteria.

## Assumptions
- Defaults chosen where the user did not specify.
- Known constraints or deferred work.
```

## Quality Bar

- Do not leave framework, model, state, or storage choices undecided unless the user explicitly wants options.
- Do not produce vague steps such as "add evals" without naming what will be evaluated.
- Do not mix direct implementation work into the plan unless the user has exited planning and asked for code.
- Keep plans compact, but include enough interface and behavior detail for implementation.
