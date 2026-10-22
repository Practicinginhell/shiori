"""Example 2: Retrieval-Augmented Generation (RAG) Pipeline.

Demonstrates:
- Clean workflow connecting retrieval to generative answering
- Reducer-based document accumulation
- Structured input and grounded answer generation
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from graphflow import Agent, Application, Field, MockModel, State, Workflow


class RAGState(State):
    """RAG pipeline state schema."""
    query: str = ""
    retrieved_docs: list[str] = Field.reducer("append")
    answer: str = ""


# Sample in-memory knowledge base
KNOWLEDGE_BASE = [
    {"topic": "langgraph", "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs, modeling workflows as directed graphs."},
    {"topic": "graphflow", "content": "Graphflow is a high-level universal framework on top of LangGraph providing clean, declarative APIs and rich abstractions."},
    {"topic": "checkpoints", "content": "LangGraph persistence allows saving state at every super-step, supporting thread-level isolation, time-travel, and human-in-the-loop workflows."},
]


def retrieve_node(state: dict) -> dict:
    """Retrieve relevant documents based on query keywords."""
    query = state.get("query", "").lower()
    matches = [
        f"[{doc['topic'].upper()}]: {doc['content']}"
        for doc in KNOWLEDGE_BASE
        if doc["topic"] in query or any(word in doc["content"].lower() for word in query.split())
    ]

    if not matches:
        matches = [f"[{KNOWLEDGE_BASE[0]['topic'].upper()}]: {KNOWLEDGE_BASE[0]['content']}"]

    return {"retrieved_docs": matches}


def build_rag_app(model=None) -> Application:
    answer_model = model or MockModel(
        responses=[
            "Graphflow is a high-level universal framework that provides clean declarative APIs on top of LangGraph."
        ]
    )

    rag_agent = Agent(
        name="rag_synthesizer",
        model=answer_model,
        instructions="Synthesize a precise answer using exclusively the provided retrieved documents.",
    )

    def generate_node(state: dict) -> dict:
        docs_text = "\n".join(state.get("retrieved_docs", []))
        prompt = f"Query: {state.get('query')}\nContext:\n{docs_text}\nPlease answer the query based on context."
        resp = rag_agent.execute({"input": prompt})
        ans = resp.get("response", resp.get("output", ""))
        return {"answer": ans, "output": ans}

    wf = Workflow("rag_pipeline", state_schema=RAGState)
    wf.then(retrieve_node).then(generate_node)

    app = Application("rag_application")
    app.register(wf)
    return app


if __name__ == "__main__":
    app = build_rag_app()
    query = "What is Graphflow and how does it relate to LangGraph?"
    print(f"User Query: {query}\n")

    result = app.run({"query": query})
    print("Retrieved Documents:")
    for doc in result.get("retrieved_docs", []):
        print(" -", doc)

    print("\nSynthesized Answer:")
    print(result.get("answer"))
