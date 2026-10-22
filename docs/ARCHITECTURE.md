# Graphflow Architecture Specification

**Graphflow** is a production-grade, high-level universal framework built on top of **LangGraph**. It eliminates low-level boilerplate—manual `StateGraph` definitions, reducer annotations, complex event filtering, and interrupt wiring—while maintaining LangGraph as the execution engine and providing zero-friction low-level escape hatches.

---

## 1. High-Level Architectural Model

```text
                           Developer / CLI
                                 │
                                 ▼
                     High-Level API (graphflow)
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
     Agents                  Workflows                  Tools
   (Autonomous,             (Declarative DAGs,       (Type-safe,
    Prompt/Model,            Loops, Branches,         Validation,
    Tools, Memory)           Parallel, HITL)          Async/Retry)
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 ▼
                      Graphflow Compiler / IR
                      (Intermediary Graph Spec)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         LangGraph StateGraph           Escape Hatches
         - Reducer annotations          (Access raw StateGraph,
         - Nodes & Command routing       inject custom nodes,
         - Send API for parallel         inspect Pregel graph)
         - Interrupt for HITL
                    │
                    ▼
          Unified Runtime Engine
         - Event Streaming (RunStarted, ToolCalled, ModelToken, etc.)
         - Checkpointing & Persistence (Memory, SQLite)
         - Observability & Tracing
         - Unified Error Handling & Retry Policies
                    │
                    ▼
        LangChain / Models / Tool APIs
        (OpenAI, Local/vLLM/Ollama, Mock, Custom)
```

---

## 2. Core Abstractions

| Abstraction | Module | Responsibility |
|---|---|---|
| `Application` | `graphflow.core.application` | Application definition container. Registers workflows, agents, and tools. Compiles into an immutable `CompiledApplication`. |
| `CompiledApplication` | `graphflow.core.application` | Executable runtime artifact. Exposes `.run()`, `.arun()`, `.stream()`, `.astream()`, `.resume()`, and `.visualize()`. Holds the underlying compiled Pregel graph as an escape hatch (`compiled_graph`). |
| `Agent` | `graphflow.core.agent` | Agent definition: model, instructions, tools, memory, structured output, middleware, tool approval. Compiles into its own LangGraph subgraph (see §4). |
| `Workflow` | `graphflow.core.workflow` | Fluent builder for workflows: `.then()`, `.branch()`, `.parallel()`, `.fan_out()`, `.loop()`, `.require_approval()`. |
| `State` / `Field` | `graphflow.core.state` | Typed, dict-like state container with declarative reducers (`"append"`, `"sum"`, `"merge_dict"`, or custom callables). Compiles into LangGraph TypedDict states with `Annotated` reducers. |
| `Tool` / `@tool` | `graphflow.tools.base` | Universal tool decorator and class supporting Pydantic input schemas, sync/async execution, retries, and timeouts. Bridges directly to LangChain `StructuredTool`. |
| `BaseModelAdapter` | `graphflow.models.base` | Provider-agnostic model interface with implementations for OpenAI (`OpenAIModel`), local endpoints (vLLM/Ollama), and deterministic zero-cost offline testing (`MockModel`). |
| `BaseCheckpointStore` | `graphflow.persistence.base` | Pluggable persistence layer (`InMemoryCheckpointStore`, `SqliteCheckpointStore`) enabling durable state, multi-turn conversational memory, and human-in-the-loop pause/resume. |
| `StreamEvent` | `graphflow.events` | Normalized stream events: `RunStarted`, `NodeStarted`, `ModelToken`, `ToolCalled`, `ToolCompleted`, `StateUpdated`, `HumanApprovalRequired`, `NodeCompleted`, and exactly one of `RunCompleted` / `RunInterrupted` / `RunFailed`. |
| `ApprovalDecision` | `graphflow.approvals` | Encapsulates human decision-making: approval, feedback, and state edits. Resumes execution via LangGraph `Command(resume=...)`. |

---

## 3. Graphflow IR (Intermediate Representation)

Graphflow strictly decouples workflow **definition** from **compilation and execution**:

```text
Workflow / Agent / State  ──lower──>  GraphIR (plain data) + Registry  ──>  Compiler  ──>  LangGraph StateGraph
```

The IR is **plain data**: `GraphIR.to_dict()` is JSON-serializable. Wherever a
workflow needs code or another live object (a node function, a condition, a
reducer, a type, a model, a tool), the IR holds a reference string such as
`"fn:prepare_tx"` and a `Registry` maps it to the object. `Workflow.to_program()`
returns both (`LoweredWorkflow(ir, registry)`); `Workflow.to_ir()` returns just
the IR for inspection, diffing or snapshot tests. `validate_graph(ir, registry)`
checks the structure and that every reference resolves.

### IR Components (`graphflow.ir`):
- `NodeIR`: id, `kind` (`function` | `raw` | `agent` | `subgraph`), function reference, retries, timeout, `on_error` reference, error isolation.
- `AgentIR` / `ToolIR`: agent definition as data (model, tools, structured output, middleware and approval handler as references; tool argument JSON schemas).
- `EdgeIR`: Deterministic directed edge between source and target nodes.
- `ConditionalEdgeIR`: Source node, routing condition reference, and route map.
- `ParallelBranchIR`: Source node, concurrent branch nodes, and fan-in join node.
- `DynamicFanOutIR`: Partition function reference, worker node, and aggregator node (compiled to LangGraph `Send`).
- `LoopIR`: Body node, continuation condition reference, exit node, `max_iterations`.
- `HITLIR`: Approval node with prompt, description, resume target and optional reject target.
- `StateSchemaIR` / `FieldIR`: field names, readable types (`type_repr`) and type references, defaults, reducers, `strict` and `validation` mode.

Builder code may hold live callables while a workflow is being defined; lowering
replaces them, so the compiler only ever sees plain data plus the registry.

### State validation

State schemas can be declared as `State` subclasses, TypedDicts, Pydantic models
or dicts; nobody is forced to write Pydantic models. Declared field types are
enforced with Pydantic `TypeAdapter`s under the hood (values are coerced the way
Pydantic's lax mode does), controlled per schema:

| `validation` | Checked |
|---|---|
| `"off"` | nothing |
| `"boundaries"` (default) | run input and human approval edits |
| `"updates"` | boundaries + every node's state update |

`strict = True` additionally rejects undeclared fields.

```python
class OrderState(State):
    order_id: str = ""
    items: list[str] = Field.reducer("append")

    class Config:
        validation = "updates"
        strict = True
```

---

## 4. LangGraph Compilation & Escape Hatches

### Compilation Pipeline
The `LangGraphCompiler` translates the `GraphIR`:
1. Dynamically constructs a `TypedDict` subclass where fields with reducers are typed as `typing.Annotated[type, reducer]`, preventing list-overwriting bugs in LangGraph.
2. Adds nodes wrapped with their retry policy (`RetryIR`, resolved to a `Retrier`), timeouts, and error handling. Retries happen inside the node wrapper, so `on_error` runs only after the last attempt fails; validation errors are never retried.
3. Maps parallel branches to LangGraph's concurrent super-steps.
4. Maps dynamic fan-outs (`fan_out`) to LangGraph's `Send` API.
5. Injects native LangGraph `interrupt(...)` in approval nodes. Like the agent's `approve` step, an
   approval node re-interrupts with an `error` when the answer's edits are invalid, instead of
   raising (a raised error would leave the bad answer attached to the interrupt).
6. Compiles each agent into its own subgraph (below) and each embedded workflow into a nested graph.
7. Connects loose terminal nodes to LangGraph `END`.

### Agents as subgraphs

```text
prepare -> model -+-> finalize -> END
            ^     +-> screen -+-> approve -> tools -+
            |     |           +-> tools ------------+
            |     +-> tools --------------------------+
            +-----------------------------------------+
```

Every model call and tool round is a separately checkpointed step. When a tool
needs human approval, the agent pauses in `approve` *before any tool runs*;
resuming continues there, without calling the model again or re-running tools.
The agent keeps its working messages in its own state and returns only its
final update to the workflow. Its tool and token events are attributed to the
workflow node that ran it.

All calls of a round that need approval form one interrupt (`action_requests`).
The `approve` node follows LangGraph's validation-loop pattern: an answer that
does not fit the pending calls (a decision the tool's policy does not allow,
edited arguments that fail the tool's schema, the wrong number of decisions)
re-interrupts with an `error` instead of failing the run. Edited arguments are
recorded per round and applied to the conversation history, so the model sees
the call that actually ran.

Before `approve`, `screen` runs each policy's `check` and records the refusals
as decisions (source `"check"`). Because that is saved in its own step before
the pause, the reviewer answers exactly the calls they were shown, even if a
refused call would pass on resume. `approve` puts only the remaining calls to
the reviewer, with each policy's `describe` output as the action's `summary`,
then runs `check` again on the approved and edited calls. Every decision
records its source (`reviewer`, `handler`, `check`), and the tool message the
model receives names it.

Model and tool calls pass through the agent's middleware stack
(`graphflow.middleware`): `before_model` hooks, then the `wrap_model_call`
chain (first middleware outermost), then `after_model` hooks; tool calls go
through the `wrap_tool_call` chain. The hooks run inside the checkpointed
`model` / `tools` / `finalize` steps, so a resume does not repeat them for
steps that already finished. The built-in `HistoryLimit` is a `before_model`
hook: it trims only the request's messages (never the checkpointed state),
always keeping the leading system messages and the current turn.

#### Why agents don't compile to LangChain's `create_agent`

Decided after a spike; the numbers and the rejected alternatives are in
[ADR-0002](adr/0002-agents-compile-to-own-subgraph.md).

### Run context and injected tool arguments

`run(..., context=...)` is passed to LangGraph as the run context
(`invoke(..., context=...)`). It reaches nested agent subgraphs (they inherit it)
and is not checkpointed, so every run and resume supplies it. Tools mark
arguments `Annotated[T, FromContext(key)]` or `FromState(key)`
(`graphflow.context`): the argument is excluded from the tool's model-facing
schema (`ToolIR.injected` records the mapping), and at the agent's tool step the
model's value for it is dropped and the real one filled in, before `Tool.execute`
runs. Middleware sees the context as `ModelRequest.context` / `ToolRequest.context`.

Setup errors (`ConfigurationError`, `CompilationError`, `StateValidationError`,
`graphflow.errors.SETUP_ERRORS`) pass through node wrappers unchanged: they are
never retried and never handed to `on_error`, since a fallback would hide the
mistake (for example a run that forgot its context).

Other errors that exhaust a node's retries (with no `on_error`) also keep their
type when they are graphflow errors (`ModelError`, `ToolExecutionError`,
`TimeoutError`, ...); anything else is wrapped in `ExecutionError` with the
original as `__cause__`. The node wrapper records the innermost node and its
attempt count on the error (`error.node`, `error.attempts`; outer nodes of a
nested workflow leave them alone). An agent turns an exception that leaves its
`wrap_model_call` chain into a `ModelError` carrying `status_code` and
`transient` (from `is_transient_error`), so retry policies keep retrying rate
limits and 5xx responses; middleware inside the chain still sees the
provider's exception.

### Threads

`CompiledApplication.thread(thread_id, context=...)` returns a `Thread`
(`graphflow/core/thread.py`), the per-conversation API. It keeps no conversation
data itself: `pending` comes from the checkpoint snapshot (`snapshot.interrupts`,
nested agent pauses included), so a new `Thread` object in another request
continues where the last one stopped. Each `CompiledApplication` holds one lock per
thread id, so runs on a thread are serialized within the process (async callers
poll the lock, never blocking the event loop). `send(..., message_id=...)` checks a
`MessageLog` (the LangGraph store if the application has one, else memory) and
records the id only after a successful run.

### Time travel

`CompiledApplication.history / replay / fork / update_state` wrap LangGraph's
`get_state_history` and `update_state`. Internal channels and nodes (`__gf_*`)
are hidden from the returned `Checkpoint`s. `update_state` validates the values
against the state schema and passes them through reducers; `overwrite=True`
wraps them in LangGraph's `Overwrite`.

### Escape Hatches
Advanced developers have three levels of escape hatches:
1. **Raw LangGraph Node**: Inject a node that directly expects `(state: dict, config: RunnableConfig)`:
   ```python
   workflow.add_raw_node("custom_node", my_langgraph_node_func)
   ```
2. **Access Uncompiled StateGraph**:
   ```python
   state_graph = workflow.to_langgraph()
   # Add low-level LangGraph conditional edges or modify nodes directly
   compiled = state_graph.compile()
   ```
3. **Access Underlying Pregel Runnable**:
   ```python
   compiled_app = app.compile()
   raw_runnable = compiled_app.compiled_graph
   # Use LangGraph internal inspection, time-travel, or attach subgraphs
   ```

---

## 5. Concurrency and State Consistency

LangGraph requires reducers when multiple concurrent nodes write to the same state key during a superstep. Graphflow guarantees consistency by:
1. Providing built-in reducers (`append`, `sum`, `merge_dict`) via `Field.reducer(...)`.
2. Automatically compiling reducers to `Annotated` types so parallel branches merge without `InvalidUpdateError`.
3. Isolating node execution so nodes return partial dictionaries without mutating shared state in-place.
