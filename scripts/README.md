# scripts/ — Uniform Command Verbs

One wrapper per verb, so agents and CI never guess the stack's commands
(["scripts to rule them all"](https://github.blog/engineering/scripts-to-rule-them-all/)).
`AGENTS.md` and CI both call these.

| Script | Verb | What it runs | Contract |
| --- | --- | --- | --- |
| `setup` | Install deps | `uv pip install -e ".[dev]"` (or `pip install` without uv) | Idempotent, non-interactive |
| `dev` | Run an app | `python examples/<name>/app.py` (default: the salon demo) | Foreground, logs to stdout |
| `test` | Test suite | `python -m pytest -q "$@"` | Forwards args; exit 0 = green |
| `lint` | Static analysis | `python -m ruff check graphflow tests examples` | Read-only; exit 0 = clean |
| `format` | Autofix in place | `ruff check --fix` (safe fixes only, no `ruff format`) | Idempotent |
| `build` | Package | `uv build` (or `pip wheel`) into `dist/` | Exit 0 = artifact exists |

## Rules

- Every script runs from anywhere (each `cd`s to the repo root).
- `PYTHON` picks the interpreter (default: `python` on PATH). For example,
  `PYTHON=/path/to/py310/bin/python ./scripts/test` checks the minimum version.
- Never interactive. `graphflow dev` (a REPL) is deliberately not wrapped.
- Exit codes are the API: `0` success, non-zero failure.
- Adding a verb? Add the wrapper here **and** a row to the Commands table in `AGENTS.md`.
