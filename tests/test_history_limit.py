"""HistoryLimit: keeps a long conversation's prompt bounded.

Only what is sent to the model is trimmed; the thread's ``messages`` keep
everything. System messages and the current turn (the latest user message and
the agent's tool calls since) are always kept, and history is cut only where a
user message starts, so a tool result is never separated from its call.
"""

import asyncio

import pytest

from graphflow import (
    Agent,
    Application,
    ConfigurationError,
    Field,
    HistoryLimit,
    InMemoryCheckpointStore,
    MockModel,
    ModelRequest,
    ModelResponse,
    State,
    ToolCall,
    Workflow,
    tool,
)
from graphflow.middleware import estimate_tokens


class Chat(State):
    messages: list = Field.reducer("append")
    output: str = ""
    response: str = ""


def thread_for(agent):
    wf = Workflow("chat", state_schema=Chat).then(agent)
    return Application("a", checkpointer=InMemoryCheckpointStore()).register(wf).thread("t")


def roles(messages):
    return [m["role"] for m in messages]


def contents(messages):
    return [m["content"] for m in messages]


def test_keeps_the_system_prompt_and_the_latest_messages():
    model = MockModel(["a1", "a2", "a3"])
    thread = thread_for(Agent("bot", model=model, instructions="SYS", middleware=[HistoryLimit(max_messages=3)]))
    for text in ("u1", "u2", "u3"):
        thread.send(text)
    assert contents(model.history[-1]) == ["SYS", "u2", "a2", "u3"]  # 3 messages besides the system prompt
    assert len(thread.state["messages"]) == 6  # the thread keeps everything


def test_cuts_only_where_a_user_message_starts():
    request = ModelRequest(messages=[
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "t", "args": {}}]},
        {"role": "tool", "tool_call_id": "1", "name": "t", "content": "r1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ])
    trimmed = HistoryLimit(max_messages=4).before_model(request)
    assert contents(trimmed.messages) == ["SYS", "u2", "a2", "u3"]
    trimmed = HistoryLimit(max_messages=5).before_model(request)
    assert contents(trimmed.messages) == ["SYS", "u2", "a2", "u3"]  # starting at the tool result would orphan it
    assert HistoryLimit(max_messages=7).before_model(request) is None  # everything fits: unchanged


def test_the_current_turn_is_always_kept():
    @tool
    def lookup(q: str) -> str:
        """Looks something up."""
        return "result"

    model = MockModel(["old answer",
                       ModelResponse(tool_calls=[ToolCall(id="1", name="lookup", args={"q": "x"})]),
                       "new answer"])
    thread = thread_for(Agent("bot", model=model, tools=[lookup], middleware=[HistoryLimit(max_messages=1)]))
    thread.send("old question")
    assert thread.send("new question")["output"] == "new answer"
    final_prompt = model.history[-1]
    assert roles(final_prompt) == ["system", "user", "assistant", "tool"]
    assert final_prompt[1]["content"] == "new question" and final_prompt[-1]["content"] == "result"


def test_max_tokens_with_a_custom_counter():
    request = ModelRequest(messages=[
        {"role": "system", "content": "S"},
        {"role": "user", "content": "x" * 10},
        {"role": "assistant", "content": "y" * 10},
        {"role": "user", "content": "z"},
    ])
    limit = HistoryLimit(max_tokens=5, count_tokens=lambda m: len(m["content"]))
    assert contents(limit.before_model(request).messages) == ["S", "z"]
    roomy = HistoryLimit(max_tokens=100, count_tokens=lambda m: len(m["content"]))
    assert roomy.before_model(request) is None


def test_default_token_estimate_counts_text_and_tool_calls():
    big = {"role": "user", "content": "word " * 400}
    request = ModelRequest(messages=[{"role": "system", "content": "S"}, big, {"role": "user", "content": "now"}])
    assert contents(HistoryLimit(max_tokens=100).before_model(request).messages) == ["S", "now"]
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "search", "args": {"q": "x" * 400}}]}
    assert estimate_tokens(call) > estimate_tokens({"role": "assistant", "content": ""}) + 100


@pytest.mark.parametrize("kwargs", [{}, {"max_messages": 0}, {"max_tokens": -1}])
def test_needs_a_positive_limit(kwargs):
    with pytest.raises(ConfigurationError, match="HistoryLimit"):
        HistoryLimit(**kwargs)


def test_async_runs_are_trimmed_too():
    model = MockModel(["a1", "a2"])
    thread = thread_for(Agent("bot", model=model, middleware=[HistoryLimit(max_messages=1)]))

    async def go():
        await thread.asend("u1")
        await thread.asend("u2")

    asyncio.run(go())
    assert contents(model.history[-1])[1:] == ["u2"]


def test_nothing_is_dropped_when_everything_fits_even_if_history_starts_with_the_assistant():
    request = ModelRequest(messages=[
        {"role": "system", "content": "SYS"},
        {"role": "assistant", "content": "Welcome! How can I help?"},
        {"role": "user", "content": "hi"},
    ])
    assert HistoryLimit(max_messages=5).before_model(request) is None
