# Architecture Decision Records

Short documents capturing structural decisions and *why* they were made, including
what was rejected. They stop settled debates from being re-litigated, by new
teammates and by AI agents alike.

## Rules

- One decision per record. Copy [ai/templates/adr.md](../../ai/templates/adr.md) to
  `NNNN-short-slug.md` with the next number.
- Write it **when the decision is made**, not retroactively.
- ADRs are immutable history: never edit a decision; supersede it with a new ADR and
  set the old one's status to `superseded by ADR-NNNN`.
- Agents: check this directory before proposing architectural changes
  (the context loading policy in `AGENTS.md`).

## Index

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions as ADRs | accepted |
| [0002](0002-agents-compile-to-own-subgraph.md) | Agents compile to graphflow's own subgraph, not `create_agent` | accepted |
| [0003](0003-adopt-template-keep-code-layout.md) | Adopt the AI-native template's knowledge layer, keep the code layout | accepted |
