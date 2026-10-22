"""Example 1: Stateful Conversational Chatbot with Memory.

Demonstrates:
- Multi-turn conversation memory using checkpointer (thread_id)
- Automatic state reducer for appending messages
- Seamless interaction via Application.run()
"""

import os
import sys

# Ensure repository root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from graphflow import (
    Agent,
    Application,
    Field,
    InMemoryCheckpointStore,
    MockModel,
    State,
    Workflow,
)


class ChatState(State):
    """Chatbot conversation state."""
    messages: list[dict[str, str]] = Field.reducer("append")
    user_name: str = "User"
    response: str = ""


def build_chatbot_app(model=None) -> Application:
    # Use deterministic mock model if no live model provided
    chat_model = model or MockModel(
        responses=[
            "Hello! I am your AI assistant. How can I help you today?",
            "I remember you asked about LangGraph. It is a powerful orchestration framework!",
            "Certainly, we can explore multi-agent architectures next.",
        ]
    )

    assistant = Agent(
        name="assistant",
        model=chat_model,
        instructions="You are a friendly, helpful conversational assistant. Remember previous context.",
    )

    wf = Workflow("chat_workflow", state_schema=ChatState)
    wf.then(assistant)

    app = Application(
        name="stateful_chatbot",
        checkpointer=InMemoryCheckpointStore(),
    )
    app.register(wf)
    return app


if __name__ == "__main__":
    app = build_chatbot_app()
    thread_id = "conversation-alice-1"

    print("=== Turn 1 ===")
    res1 = app.run(
        {"messages": [{"role": "user", "content": "Hi, who are you?"}]},
        thread_id=thread_id,
    )
    print("Assistant:", res1.get("response"))

    print("\n=== Turn 2 (Context remembered via thread_id) ===")
    res2 = app.run(
        {"messages": [{"role": "user", "content": "Tell me about LangGraph."}]},
        thread_id=thread_id,
    )
    print("Assistant:", res2.get("response"))
    print("Total messages in history:", len(res2["messages"]))
