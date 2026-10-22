# Graphflow Quickstart Guide

This guide will help you build your first AI applications using **Graphflow** in minutes.

---

## 1. Installation

```bash
pip install -e .
```

To install with OpenAI provider support:
```bash
pip install -e ".[openai]"
```

---

## 2. Basic Agent in 10 Lines

```python
from graphflow import Application, Agent, MockModel

agent = Agent(
    name="assistant",
    model=MockModel("Hello! How can I assist you today?"),
    instructions="You are a helpful assistant.",
)

app = Application("hello-agent")
app.register(agent)

result = app.run({"query": "Hi!"})
print("Agent:", result["output"])
```

---

## 3. Defining Workflows with Typed State and Reducers

```python
from graphflow import Application, Workflow, State, Field

class DataState(State):
    input_text: str = ""
    processed_steps: list[str] = Field.reducer("append")
    summary: str = ""

def clean_step(state: dict) -> dict:
    return {"processed_steps": ["cleaned input text"]}

def format_step(state: dict) -> dict:
    return {
        "processed_steps": ["formatted into uppercase"],
        "summary": state["input_text"].upper()
    }

wf = Workflow("data_processor", state_schema=DataState)
wf.then(clean_step).then(format_step)

app = Application("data_app")
app.register(wf)

result = app.run({"input_text": "hello world"})
print("Steps:", result["processed_steps"])
print("Summary:", result["summary"])
```

---

## 4. Defining Tools

```python
from graphflow import tool

@tool
def calculate(expression: str) -> float:
    """Evaluates a mathematical expression."""
    return float(eval(expression))

@tool(retries=2, timeout=5.0)
def search_database(query: str) -> list[str]:
    """Searches documents matching the query."""
    return ["Doc A", "Doc B"]
```

---

## 5. Multi-Agent Supervisor Pattern

```python
from graphflow import Application, Agent, MockModel
from graphflow.multiagent import create_supervisor

researcher = Agent(name="researcher", model=MockModel("Research findings on quantum."))
coder = Agent(name="coder", model=MockModel("Code implementing quantum circuit."))

supervisor_model = MockModel([
    '{"next": "researcher", "instruction": "Gather facts"}',
    '{"next": "coder", "instruction": "Implement code"}',
    '{"next": "FINISH", "instruction": "Done"}',
])

wf = create_supervisor(
    name="quantum_team",
    model=supervisor_model,
    agents=[researcher, coder],
)

app = Application("quantum_app")
app.register(wf)

result = app.run({"query": "Simulate quantum circuit"})
print("Final Output:", result["output"])
```

---

## 6. Human-In-The-Loop (HITL)

```python
from graphflow import Application, Workflow, State, InMemoryCheckpointStore, ApprovalDecision, HumanApprovalRequired

class OrderState(State):
    amount: float = 0.0
    status: str = "pending"

def create_order(state: dict) -> dict:
    return {"amount": 5000.0}

def charge_card(state: dict) -> dict:
    return {"status": f"Charged ${state['amount']}"}

wf = Workflow("checkout", state_schema=OrderState)
wf.then(create_order)
wf.require_approval(before=charge_card, prompt="Authorize high-value charge?")
wf.then(charge_card)

app = Application("checkout_app", checkpointer=InMemoryCheckpointStore())
app.register(wf)

thread_id = "user_tx_123"

# Run until interrupt
for event in app.stream({"amount": 0}, thread_id=thread_id):
    if isinstance(event, HumanApprovalRequired):
        print(f"Paused for approval: {event.prompt}")

# Resume with human approval and modified amount
result = app.resume(
    thread_id=thread_id,
    decision=ApprovalDecision(approved=True, edits={"amount": 4500.0}),
)
print("Result:", result["status"])  # Charged $4500.0
```

### Approving an agent's tool calls: approve, edit or reject

Tools marked `requires_approval=True` (or listed in the agent's `interrupt_on`)
pause the agent before they run. All pending calls of a round arrive in one
request (`result.interrupts[0].actions`). Answer each call with a `ToolDecision`:
approve it, edit its arguments, or reject it with feedback for the model. An
answer that doesn't fit (a decision the tool doesn't allow, invalid edited
arguments) asks the reviewer again, with the reason in `payload["error"]`.

```python
from graphflow import Agent, Application, InMemoryCheckpointStore, MockModel, ModelResponse, ToolCall, ToolDecision, tool

@tool
def send_email(to: str, body: str) -> str:
    """Sends an email."""
    return f"sent to {to}"

model = MockModel([
    ModelResponse(tool_calls=[ToolCall(id="1", name="send_email", args={"to": "all@corp.com", "body": "hi"})]),
    "Done.",
])
agent = Agent(
    "mailer", model=model, tools=[send_email],
    interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit", "reject"]}},
)
app = Application("mail", checkpointer=InMemoryCheckpointStore()).register(agent)

paused = app.run({"query": "Email everyone"}, thread_id="t1")
print(paused.interrupts[0].actions[0]["args"])   # {'to': 'all@corp.com', 'body': 'hi'}

result = app.resume("t1", ToolDecision.edit({"to": "team@corp.com", "body": "hi"}))
print(result["output"])                          # Done.
# ToolDecision.approve(), ToolDecision.reject("feedback"), a list with one decision per
# pending call, or a bool (applied to every call) also work.
```

When parallel branches pause at the same time, answer them together with
`app.resume_all(thread_id, {interrupt_id: decision, ...})`.

**Checking and describing calls.** A policy can also have a `check` and a
`describe` function. Both receive a `ToolRequest` (the call's arguments, the run
context, the state) and may be async. `check` returns a reason to refuse, or
`None`. It runs before the reviewer is asked, so a call that can't succeed is
refused without bothering anyone, and again on their answer, in case the world
changed while they decided or they edited the arguments. `describe` writes the
summary the reviewer sees (`action["summary"]`), built from your data rather than
the model's wording. The model is told who refused a call: the reviewer, the
`approval_handler`, or the check with its reason. Keep both functions free of
side effects: they can run more than once (the approval step restarts on resume).

```python
from graphflow import Agent, Application, InMemoryCheckpointStore, MockModel, ModelResponse, ToolCall, tool

free = {"09:00", "10:00"}

@tool
def book(time: str) -> str:
    """Books a time."""
    free.discard(time)
    return f"booked {time}"

def check(request):
    time = request.tool_call.args["time"]
    return None if time in free else f"{time} is taken"

def describe(request):
    return f"Book {request.tool_call.args['time']} for {request.context['user']}"

model = MockModel([ModelResponse(tool_calls=[ToolCall(id="1", name="book", args={"time": "09:00"})]), "Booked."])
agent = Agent("desk", model=model, tools=[book], interrupt_on={"book": {"check": check, "describe": describe}})
thread = Application("desk", checkpointer=InMemoryCheckpointStore()).register(agent).thread("t", context={"user": "ann"})

thread.send({"query": "9am please"})
print(thread.pending[0].actions[0]["summary"])   # Book 09:00 for ann
free.discard("09:00")                            # someone else takes it meanwhile
thread.respond(True)
print(model.history[-1][-1]["content"])          # Tool call refused before running: 09:00 is taken
```

---

## 7. Agent Middleware

Middleware hooks run around an agent's model and tool calls. They follow
LangChain's agent middleware: `before_model` / `after_model` rewrite the
request or response; `wrap_model_call` / `wrap_tool_call` wrap the call itself
(retry it, cache it, block it). The first middleware is the outermost.

```python
from graphflow import Agent, AgentMiddleware, MockModel

class KeepRecentHistory(AgentMiddleware):
    """Sends only the system prompt and the last 6 messages to the model."""
    def before_model(self, request):
        system, rest = request.messages[:1], request.messages[1:]
        return request.override(messages=system + rest[-6:])

class BlockDangerousTools(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        if request.tool_call.name == "drop_table":
            return "Blocked by policy."      # the tool never runs; the model sees this result
        return handler(request)

agent = Agent("assistant", model=MockModel("Hi!"), middleware=[KeepRecentHistory(), BlockDangerousTools()])
print(agent.execute({"input": "Hello"})["output"])   # Hi!
```

For async runs, define `awrap_model_call` / `awrap_tool_call`; otherwise the
sync wrappers run in a worker thread.

With `structured_output=...`, an answer that fails validation is sent back to
the model with the error (`structured_output_retries`, default 1).

### Long conversations: `HistoryLimit`

A thread that lives for months (a chat channel) would send the model its whole
history every turn. `HistoryLimit` sends only the recent part: it keeps the
system messages and the current turn, then as much earlier history as fits
`max_messages` and/or `max_tokens`, cutting only where a user message starts so
a tool result never loses its call. The thread's `messages` keep everything.

```python
from graphflow import Agent, Application, Field, HistoryLimit, InMemoryCheckpointStore, MockModel, State, Workflow

class Chat(State):
    messages: list = Field.reducer("append")

model = MockModel(["one", "two", "three"])
agent = Agent("bot", model=model, middleware=[HistoryLimit(max_messages=3)])
thread = Application("chat", checkpointer=InMemoryCheckpointStore()).register(
    Workflow("chat", state_schema=Chat).then(agent)).thread("t")
for text in ("first", "second", "third"):
    thread.send(text)
print([m["content"] for m in model.history[-1]][1:])   # ['second', 'two', 'third']
print(len(thread.state["messages"]))                   # 6
# HistoryLimit(max_tokens=8000, count_tokens=my_tokenizer) for a token budget.
```

---

## 8. Retry Policies

`retry=RetryPolicy(...)` gives nodes and tools exponential backoff with
jitter. By default it retries only transient errors: connection errors,
timeouts, and HTTP 429 or 5xx. A bug such as a `KeyError` fails at once instead of
being retried. `retries=n` still retries any error `n` times.

```python
from graphflow import Application, RetryPolicy, Workflow

attempts = []

def call_api(state: dict) -> dict:
    attempts.append(1)
    if len(attempts) < 3:
        raise ConnectionError("temporary network failure")
    return {"output": "ok"}

wf = Workflow("resilient").then(
    call_api,
    retry=RetryPolicy(max_attempts=5, initial_interval=0.01, backoff_factor=2.0, max_interval=10.0),
)
print(Application("app").register(wf).run({})["output"], len(attempts))   # ok 3
# RetryPolicy(retry_on=RateLimitError) or retry_on=lambda e: ... chooses what to retry.
```

### Handling failures

Errors keep their type on the way out of a workflow, so you can react to each
kind. A failed model call is a `ModelError` (with `agent`, `status_code`,
`transient`, and the provider's exception as `__cause__`); retry policies retry
it only when it is transient. Graphflow's own errors, such as `ToolExecutionError`
and `TimeoutError`, pass through unchanged. Any other exception is wrapped in an
`ExecutionError` whose `__cause__` is the original. All of them are
`ExecutionError`s and say where they happened: `error.node` (the innermost node)
and `error.attempts`. Inside the agent, `wrap_model_call` middleware still sees
the provider's own exception, to retry or fall back.

```python
from graphflow import Agent, Application, MockModel, ModelError, Workflow

def provider_down(messages):
    raise ConnectionError("provider unreachable")

app = Application("support").register(Workflow("w").then(Agent("bot", model=MockModel(provider_down))))
try:
    app.run({"query": "hi"})
except ModelError as e:
    print(e.node, e.transient)   # bot True  -> e.g. reply "Sorry, please try again in a minute"
```

---

## 9. Time Travel: History, Replay and Fork

With a checkpointer, every step of a thread is saved. Browse the history,
re-run from any checkpoint, or fork: change a past state and continue from
there, keeping the original history.

```python
from graphflow import Application, Field, InMemoryCheckpointStore, State, Workflow

class Counter(State):
    n: int = 0
    log: list = Field.reducer("append")

def double(state): return {"n": state["n"] * 2, "log": ["double"]}
def inc(state): return {"n": state["n"] + 1, "log": ["inc"]}

app = Application("tt", checkpointer=InMemoryCheckpointStore()).register(
    Workflow("w", state_schema=Counter).then(double).then(inc)
)
app.run({"n": 3}, thread_id="t")                     # n = 3 * 2 + 1 = 7

history = app.history("t")                            # newest first
before_inc = next(c for c in history if c.next == ("inc",))
print(before_inc.values["n"])                         # 6

print(app.replay("t", before_inc.checkpoint_id)["n"])            # 7 (inc runs again)
print(app.fork("t", before_inc.checkpoint_id, {"n": 100})["n"])  # 101

app.update_state("t", {"log": ["note"]})              # passes through the append reducer
app.update_state("t", {"log": ["reset"]}, overwrite=True)  # replaces the value
```

`update_state` also lets a reviewer correct a paused thread's state before
resuming it.

---

## 10. Run Context: Who Is Asking

Data that comes with each run rather than living in the conversation (the user's
id, their tenant, their permissions) is the **run context**: `run(..., context=...)`.
Every node sees it (`get_context()`), agents inside workflows included, and it is
never saved in checkpoints, so pass it again on every `resume`.

Tools receive context values through arguments the model can neither see nor set:

```python
from typing import Annotated
from graphflow import Agent, Application, FromContext, MockModel, ModelResponse, ToolCall, tool

@tool
def my_orders(status: str, user_id: Annotated[str, FromContext("user_id")]) -> str:
    """Lists the current user's orders with a given status."""
    return f"orders of {user_id} with status {status}"

model = MockModel([ModelResponse(tool_calls=[ToolCall(id="1", name="my_orders", args={"status": "open"})]), "Done."])
app = Application("shop").register(Agent("support", model=model, tools=[my_orders]))
app.run({"query": "What are my open orders?"}, context={"user_id": "u-42"})
print(model.history[1][-1]["content"])   # orders of u-42 with status open
```

`user_id` is left out of the schema sent to the model; a value the model supplies
for it is ignored; a run that doesn't provide it fails with `ConfigurationError`
instead of guessing. `FromState("key")` fills an argument from the workflow state.

---

## 11. Threads: One Conversation as an Object

A chat app handles each incoming message on its own: a webhook call, a worker, a
request. `app.thread(...)` gives one conversation as an object that reads and
writes everything through the checkpointer, so any request can pick it up:

```python
from typing import Annotated
from graphflow import (Agent, Application, FromContext, InMemoryCheckpointStore, MockModel,
                       ModelResponse, ToolCall, tool)

@tool(requires_approval=True)
def cancel_order(order_id: str, user_id: Annotated[str, FromContext("user_id")]) -> str:
    """Cancels one of the current user's orders."""
    return f"cancelled {order_id} for {user_id}"

model = MockModel([ModelResponse(tool_calls=[ToolCall(id="1", name="cancel_order", args={"order_id": "A7"})]),
                   "Your order A7 is cancelled."])
app = Application("shop", checkpointer=InMemoryCheckpointStore()).register(
    Agent("support", model=model, tools=[cancel_order]))

# Request 1: the customer's message arrives (message_id: the channel's id for it)
thread = app.thread("user:u-42", context={"user_id": "u-42"})
thread.send("Please cancel order A7", message_id="msg-1")
print(thread.send("Please cancel order A7", message_id="msg-1").duplicate)   # True: a redelivery is skipped

# Request 2, maybe another worker: the pending decision is read from the checkpoint
thread = app.thread("user:u-42", context={"user_id": "u-42"})
print(thread.pending[0].actions[0]["args"])        # {'order_id': 'A7'}
print(thread.respond(True)["output"])             # Your order A7 is cancelled.
```

- `send` takes a dict of inputs, or text (added as a user message). It refuses to
  run while a decision is pending; answer with `respond` / `respond_all` first.
- Runs on one thread happen one at a time within a process; different threads run
  in parallel. With several worker processes, route each thread to one worker.
- Processed message ids are kept in the application's `store` when it has one
  (surviving restarts), otherwise in memory. Only successful runs are recorded, so
  a redelivery after a failure runs again.
- `thread.state`, `history`, `update_state`, `replay` and `fork` work on that
  conversation; every method has an `a...` async twin.

---

## 12. Command Line Interface (CLI)

Initialize a new project:
```bash
python -m graphflow.cli init my-project
```

Visualize the execution graph:
```bash
python -m graphflow.cli graph app.py --format ascii
python -m graphflow.cli graph app.py --format mermaid
```

Inspect internal nodes and IR:
```bash
python -m graphflow.cli inspect app.py
```

Run application:
```bash
python -m graphflow.cli run app.py --query "Explain LangGraph"
```

Start interactive developer REPL:
```bash
python -m graphflow.cli dev app.py
```
