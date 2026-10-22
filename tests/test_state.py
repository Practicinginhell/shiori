"""Unit tests for Graphflow State system and reducers."""

from graphflow import Field, State
from graphflow.core.state import create_state_schema


class CustomState(State):
    query: str = ""
    messages: list = Field.reducer("append")
    counter: int = Field.reducer("sum", default=0)
    metadata: dict = Field.reducer("merge_dict")
    status: str = "active"


def test_state_defaults_and_attributes():
    state = CustomState(query="test query")
    assert state.query == "test query"
    assert state.messages == []
    assert state.counter == 0
    assert state.metadata == {}
    assert state.status == "active"


def test_state_dict_interface():
    state = CustomState(query="test")
    assert state["query"] == "test"
    state["query"] = "updated"
    assert state["query"] == "updated"
    assert state.query == "updated"
    assert "query" in state


def test_state_schema_ir_generation():
    schema_ir = CustomState.get_schema_ir()
    assert schema_ir.name == "CustomState"
    assert "messages" in schema_ir.fields
    assert schema_ir.fields["messages"].reducer == "append"
    assert schema_ir.fields["counter"].reducer == "sum"
    assert schema_ir.fields["metadata"].reducer == "merge_dict"
    assert schema_ir.fields["status"].reducer == "replace"


def test_dynamic_state_schema():
    schema = create_state_schema({
        "items": Field.reducer("append"),
        "total": Field.reducer("sum", default=10),
    })
    assert "items" in schema.fields
    assert schema.fields["items"].reducer == "append"
    assert schema.fields["total"].default == 10
