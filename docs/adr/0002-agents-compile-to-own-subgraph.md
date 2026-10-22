# ADR-0002: Agents compile to graphflow's own subgraph, not LangChain's `create_agent`

**Status:** accepted
**Date:** 2026-09-22

## Context

Each graphflow agent compiles into its own LangGraph subgraph
(`graphflow/graph/agent_graph.py`: `prepare → model → [screen → approve] → tools → …
→ finalize`). LangChain 1.x ships `create_agent`, a maintained agent loop with a
middleware catalog (summarization, PII, model fallback, tool retry). Reusing it would
hand upkeep of the loop upstream. A spike on 2026-09-22 (`langchain` 1.4.2,
LangGraph 1.2.12) ran graphflow agents on `create_agent` with about 250 lines of glue
(model adapter bridge, tools with injected arguments, graphflow middleware,
`HumanInTheLoopMiddleware` for approvals). 285 of 327 tests passed. The 42 failures:

- 31 approval tests: LangChain's HITL raises on a wrong answer instead of asking
  again, uses a different payload shape, keeps the model's call on edits (adding a
  notice), and has no `check` / `describe` or approval handler.
- 4 structured-output tests: `response_format` works through tool/provider
  strategies, not graphflow's parse-then-retry-with-the-error flow.
- 1 `max_tool_iterations` test: no "force a final answer without tools"; LangChain's
  call-limit middleware ends the run instead.
- 6 glue gaps (tool events, `FromState`, `iteration`, async wrap hooks): plumbing,
  not fixed in the spike.

## Decision

We will keep compiling agents to graphflow's own subgraph.

## Consequences

**Easier:**

- Approval semantics (validation loop, `check` / `describe`, honest rejection
  sources), structured-output retries and `max_tool_iterations` stay as documented.
- One message format (graphflow dicts) end to end; no `langchain` dependency.

**Harder / we accept:**

- graphflow maintains its own agent loop, and LangChain's middleware catalog isn't
  directly usable.

## Alternatives considered

- **Compile to `create_agent`** — rejected: the approval logic would be re-hosted as
  a LangChain `after_model` middleware anyway (the code moves, it doesn't go away),
  at the cost of a second message format converted on every model call and a new
  dependency.
- **Adapter that runs LangChain `AgentMiddleware` inside graphflow agents** — not
  rejected; the cheaper route if the catalog is wanted later.
