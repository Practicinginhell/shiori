# ADR-0003: Adopt the AI-native template's knowledge layer, keep the code layout

**Status:** accepted
**Date:** 2026-09-22

## Context

The repository adopted the layout of
[ClaudeCodeTemplate](https://github.com/Practicinginhell/ClaudeCodeTemplate)
(commit `750f57c`): `AGENTS.md` as the canonical agent instructions with thin tool
adapters, `ai/` (context, prompts, templates, checklists), `.claude/`, `scripts/`,
`docs/adr/`, `tasks/`. The template also puts application code in `src/` and tests in
`tests/`, but calls `src/` a placeholder to restructure freely. Here, `src/` already
holds a legacy prototype that isn't part of the package, and the package lives at
`graphflow/`.

## Decision

We will adopt the template's knowledge and control layer in full and keep the code
where it is: the package in `graphflow/`, tests in `tests/`, runnable apps in
`examples/`, the legacy prototype in `src/`. The template's `src/` and `tests/`
placeholder READMEs are not added.

## Consequences

**Easier:**

- No import, packaging (`pyproject.toml`), CI or documentation churn.
- Agents learn the layout from the `AGENTS.md` repository map and
  `ai/context/architecture.md`, as the template intends.

**Harder / we accept:**

- `src/` means "legacy" here, not "the code", which surprises anyone expecting the
  template's convention; the repository map says so explicitly.

## Alternatives considered

- **Move the package to a `src/graphflow/` layout** — rejected: every import path,
  the packaging config, CI, and the docs would change for no functional gain, and the
  legacy prototype would have to move first.
