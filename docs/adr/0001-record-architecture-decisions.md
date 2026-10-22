# ADR-0001: Record architecture decisions as ADRs

**Status:** accepted
**Date:** 2026-09-22

## Context

This project is built collaboratively by humans and AI coding agents. Agents start
every session with no memory of past discussions; humans forget them within months.
Decisions whose reasoning isn't written down get re-litigated — or worse, silently
reversed by an agent that sees only the code, not the constraint that shaped it.

## Decision

We will record every significant architectural decision as an ADR in `docs/adr/`,
using the template at `ai/templates/adr.md`, numbered sequentially. `AGENTS.md`
directs agents to consult ADRs before proposing structural changes.

## Consequences

**Easier:**

- Agents and new teammates learn *why* the system is shaped this way, cheaply.
- Rejected alternatives stay rejected unless the context genuinely changes.

**Harder / we accept:**

- Small writing cost at decision time — bounded by keeping ADRs under a page.
- The index must be maintained (`docs/adr/README.md`); the pre-merge checklist's
  knowledge-upkeep item covers this.

## Alternatives considered

- **Decision log in one big file** — rejected: merge conflicts, and agents can't load
  one decision without loading all of them.
- **Rely on git history / PR descriptions** — rejected: reasoning gets buried under
  mechanical detail and is effectively unsearchable at decision-granularity.
