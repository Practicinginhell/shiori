# Test-Time Communication on graphflow

Implementation of **"Scaling Discovery through Test-Time Communication"**
(Park, Kontonis, Garg, Krishnamurthy, Papailiopoulos; arXiv:2609.21032, 2026) in
`graphflow.ttc`.

k identical agents with no predefined roles work the same task in one
container. They communicate only through a shared directory, and their
results are compared with the best of k independent agents (**team@k vs
best@k**) at the same per-agent budget.

## Paper → code

| Paper | Code |
|---|---|
| Communication prompt, Appendix A.2 (verbatim) | `graphflow.ttc.protocol.COMMUNICATION_PROMPT`, filled by `render_communication_prompt(n, workspace)` |
| Atomic slot ownership (`mkdir` race) | `SharedWorkspace.claim_slot` (atomic `os.mkdir`) |
| Approaches, findings, disconfirmations, score log, coordination | `SharedWorkspace.set_approach / append_finding / append_disconfirmation / log_score / append_coordination` |
| Score-log line `[slot $S HH:MM:SS] score=… family=…` | `SharedWorkspace.log_score` (UTC, `date -u +%H:%M:%S`) |
| Plateau: no strict improvement over 3 consecutive attempts | `SharedWorkspace.is_plateaued` |
| File locks on the shared graded artifact | `SharedWorkspace.lock(name)` |
| Submission ledger; best valid submission retained | `SubmissionLedger` (outside the agent container; submissions are scored from snapshots) |
| team@k: k agents, one container, private contexts, shared files | `run_team` / `arun_team`: a graphflow `Workflow` `prepare → parallel(agent_0..k-1) → collect` |
| best@k: k isolated single-agent trials | `run_independent` / `arun_independent` (same workflow, separate containers) |
| Agents keep working until time runs out | each `agent_i` node runs a graphflow `Agent` turn after turn until its `Budget` (seconds / turns / output tokens) is spent |
| Metrics against time and total output tokens | `RunResult.trajectory("time" | "tokens")`, `metrics.on_token_axis` |
| best@k = 1 − C(n−s,k)/C(n,k) | `metrics.best_at_k_exact`, `mean_best_at_k`, `matching_pool_size` |
| RHAE, Eq. 1–2 | `metrics.rhae_level_score`, `rhae_game`, `rhae` |
| Polyomino packing scorer, Appendix A.4 | `tasks.PolyominoPackingTask`, `tasks.polyomino.packing_quality` |
| Staged model and Proposition 1, Section 3.4 / Appendix B | `theory.team_success_lower_bound`, `best_success_upper_bound`, `exact_*`, `simulate_completion` (with herding `groups=g`) |

## Usage

```python
from graphflow.ttc import Budget, PolyominoPackingTask, run_independent, run_team

task = PolyominoPackingTask()                      # 70 cases, 100–10^4 pieces, 1–10 cells
budget = Budget(seconds=3 * 3600)                  # same per-agent budget for both settings
team = run_team(task, "openai:gpt-4.1", k=3, budget=budget, workdir="runs/team")
solo = run_independent(task, "openai:gpt-4.1", k=3, budget=budget, workdir="runs/solo")
print(team.best.score, solo.best.score)
print(team.trajectory("tokens"))
```

Any `Task` with an agent-accessible verifier works. Subclass `Task` or use
`CallableTask(name, instruction, scorer)`. Pass `listeners=[ExecutionTracer()]`
to trace every agent through graphflow's event stream.

`examples/06_test_time_communication/app.py demo` runs offline. It uses
scripted agents in the staged model from Figure 12, where each agent can
supply only some of the breakthroughs. Isolated agents stall at stage 1, while
the team keeps building on each other's verified results. The demo also
prints Proposition 1's exact probabilities next to its bounds.

## Differences from the paper's setup

These differences come from what the paper specifies and what is available
outside the authors' infrastructure.

- **Agent runtime.** The paper uses GitHub Copilot CLI agents with a shell in
  a Harbor container. Here each agent is a graphflow `Agent`. Its tools are
  the protocol's shell steps as structured calls (`claim_slot`, `log_score`,
  …), the task's own verifier (`submit`), and file access confined to the
  trial's container. After the verbatim prompt, a short harness note maps
  each shell step to its tool. `allow_shell=True` adds a bash tool so the
  prompt's commands can run literally. Enable it only inside an isolated
  container.
- **Turns.** CLI agents run autonomously. Here, when an agent ends a turn, the
  harness starts the next one with a short "continue (time remaining)"
  message, until the budget is spent.
- **Isolation.** Trials are separate directories on one machine, not
  containers. CPU and memory are not partitioned, and network access is not
  blocked.
- **Paths.** The paper does not give the shared-scratch paths, and the
  authors' repository was empty. The layout is `<container>/team/{slots,
  findings.md, disconfirmations.md, score_log.txt, coordination.md, work-S}`.
- **Single-agent prompt.** The paper does not give one. Independent agents
  receive the task instruction and a short tool note, without the
  communication prompt.
- **Polyomino packing.** The Frontier-CS hidden instances and I/O format are
  not public. Instances are generated with the paper's parameters from a
  seed, and the I/O format is documented in `AGENT.md`. The scorer follows
  Appendix A.4: reflections, 90° rotations and translations; one rectangle;
  no overlap; C/A averaged over cases; any invalid case rejects the
  submission; 2 s per case. The 256 MiB limit is enforced only where
  `ulimit -v` works (Linux, not macOS). Scores are therefore not directly
  comparable to the paper's 0.894 and 0.945.
- **Not included.**
  - ARC-AGI-3 needs the ARC game API and its per-level action budgets and
    team synchronization (Appendix A.3). Its metrics (exact best@k, RHAE) are
    included.
  - MNIST compression needs a GPU and a sealed test-set oracle (Appendix
    A.5).
  - Terminal-Bench 2.0.

  Each of these can be added as a `Task`.
