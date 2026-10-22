# Graphflow

**Graphflow** is a production-grade, high-level universal framework on top of **LangGraph**.

It provides a clean, declarative API for building AI agents, workflows, and multi-agent systems without repeatedly writing low-level LangGraph boilerplate, while maintaining LangGraph as the reliable execution engine underneath.

---

## Key Features

- **Intuitive Public API**: Build working agents in 5–15 lines of code (`Application`, `Agent`, `Workflow`, `State`, `Field`, `tool`).
- **Graphflow IR**: Pure intermediate representation strictly separating application definition from compilation and execution.
- **Typed State & Reducers**: Declarative reducers (`append`, `sum`, `merge_dict`) compiled into LangGraph TypedDict states with `Annotated` reducers, eliminating state-overwrite bugs.
- **LangGraph Compiler**: Compiles high-level workflows into native `StateGraph` and `Pregel` runnables.
- **Low-Level Escape Hatches**: Direct access to `to_langgraph()`, raw LangGraph nodes `(state, config)`, and the underlying compiled Pregel runnable (`app.compiled_graph`).
- **Orchestration Patterns**: Built-in Supervisor, Sequential pipeline, Parallel fan-out/fan-in, dynamic map-reduce via LangGraph `Send` API, and reflection loops.
- **Human-In-The-Loop (HITL)**: Native LangGraph `interrupt()` and `Command(resume=...)` abstraction. Workflow approval steps can route on rejection. Agents batch their tool calls that need approval into one request, answered per call with approve, edit (new arguments) or reject (feedback for the model), under per-tool policies (`interrupt_on`). `resume_all` answers parallel pauses at once.
- **Streaming Events**: Real-time, standardized stream events (`RunStarted`, `NodeStarted`, `ModelToken`, `ToolCalled`, `ToolCompleted`, `StateUpdated`, `HumanApprovalRequired`, `NodeCompleted`, and exactly one of `RunCompleted` / `RunInterrupted` / `RunFailed`).
- **Resilience**: `RetryPolicy` for nodes and tools (exponential backoff with jitter; transient errors only by default), enforced timeouts, `on_error` fallbacks, and error isolation for parallel branches.
- **Threads**: `app.thread(id, context=...)` is one conversation: `send`, `pending` (read from the checkpoint, so any worker can answer), `respond`, history and fork, with one run at a time per thread and redelivered messages skipped by `message_id`.
- **Run context**: `run(..., context={...})` carries who the user is (tenant, permissions) with each run, without saving it in the thread; tools receive it through `Annotated[..., FromContext(...)]` arguments the model can't see or set.
- **Agent middleware**: `before_model`, `after_model`, `wrap_model_call` and `wrap_tool_call` hooks (as in LangChain) for guardrails, caching or retries, plus a built-in `HistoryLimit` that bounds the prompt of long conversations; structured output that fails validation is sent back to the model with the error.
- **Validation**: Workflows are validated before compilation (dangling references, duplicate nodes, empty graphs), and declared state types are enforced at run boundaries or on every update, without requiring Pydantic models.
- **Plain-data IR**: `workflow.to_ir().to_dict()` is JSON-serializable; code is referenced through a registry, so graphs can be inspected, diffed and snapshot-tested without running them.
- **Agents as subgraphs**: every model call and tool round is checkpointed, so pausing for tool approval resumes exactly where it stopped.
- **Persistence & time travel**: Thread-isolated checkpoints (Memory, SQLite via `pip install graphflow[sqlite]`, or any LangGraph checkpointer such as `PostgresSaver`); `history`, `replay`, `fork` and `update_state` for browsing and rewriting a thread's past.
- **Observability**: Attach `ExecutionTracer` or `StructuredLogger` as `Application(listeners=[...])` to trace every run.
- **Offline Deterministic Testing**: Zero-cost, reproducible tests with `MockModel` and `TestHarness` requiring zero API keys or external infrastructure.
- **Test-time communication** (`graphflow.ttc`): the team@k vs best@k protocol from *Scaling Discovery through Test-Time Communication* (arXiv:2609.21032). See [docs/TTC.md](docs/TTC.md).
- **Developer CLI**: `graphflow init`, `run`, `test`, `graph`, `dev`, `inspect`.

---

## Quickstart

```python
from graphflow import Application, Agent, MockModel

agent = Agent(
    name="assistant",
    model=MockModel("Hello from Graphflow!"),
    instructions="You are a helpful assistant.",
)

app = Application("my-app")
app.register(agent)

result = app.run({"query": "Hello!"})
print(result["output"])
```

`run()` returns a `RunResult`: a dict of the final state that also carries
`result.status` (`"completed"` or `"interrupted"`) and `result.interrupts`
(pending approvals to `app.resume(thread_id, decision)`).

---

## Example Applications

Graphflow ships with 7 example applications in `examples/`:
1. `examples/01_chatbot`: Stateful conversational chatbot with memory across turns.
2. `examples/02_rag`: Retrieval-Augmented Generation pipeline.
3. `examples/03_research_agent`: Autonomous research agent with reflection/review cycle.
4. `examples/04_multi_agent`: Supervisor multi-agent pattern with specialized workers.
5. `examples/05_human_in_the_loop`: Proposal creation, pause for human verification/edits, and approved execution.
6. `examples/06_test_time_communication`: Communicating agent teams vs independent agents (team@k vs best@k), with an offline demo.
7. `examples/07_nail_salon`: A nail salon AI receptionist: salon data, SQLite store, a deterministic appointment engine, and a booking agent that confirms every change with the customer. See [its README](examples/07_nail_salon/README.md).
8. `examples/08_tanklight`: TankLight, a household water protection tracker for truck-served tanks in the North: a plain-Python rules core, a graphflow household pipeline, a simulated community with truck dispatch, and a TypeScript UI. See [its README](examples/08_tanklight/README.md).

---

## Development

Every task has one command in `scripts/`, and CI runs the same ones:

```bash
./scripts/setup      # install with the dev extras
./scripts/test       # the offline pytest suite (args go to pytest)
./scripts/lint       # ruff
./scripts/dev        # run an example (default: the nail salon demo)
./scripts/build      # sdist and wheel into dist/
```

CI runs lint and the tests on Python 3.10–3.13 (`.github/workflows/ci.yml`).

## Repository layout

| Path | Contents |
| --- | --- |
| `graphflow/`, `tests/`, `examples/` | The package, its tests, runnable apps |
| `docs/` | User guide, architecture, decision records (`docs/adr/`) |
| `scripts/` | The commands above |



water safety remoteness the reality of inut communities
four regions
How do we know the safety of the water