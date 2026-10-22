"""Unit tests for Graphflow CLI."""

import os
import tempfile

from click.testing import CliRunner

from graphflow.cli.main import cli


def test_cli_init():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["init", "my-new-agent"])
        assert result.exit_code == 0
        assert os.path.exists("my-new-agent/app.py")
        assert os.path.exists("my-new-agent/README.md")


def test_cli_graph_and_inspect():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write('''from graphflow import Application, Workflow, State

class S(State):
    val: int = 0

def step(state: dict) -> dict:
    return {"val": 1}

wf = Workflow("test_wf", state_schema=S)
wf.then(step)

app = Application("inspect_app")
app.register(wf)
''')
        f.flush()
        temp_path = f.name

    try:
        # Test inspect
        res_inspect = runner.invoke(cli, ["inspect", temp_path])
        assert res_inspect.exit_code == 0
        assert "inspect_app" in res_inspect.output
        assert "step" in res_inspect.output

        # Test graph
        res_graph = runner.invoke(cli, ["graph", temp_path, "--format", "ascii"])
        assert res_graph.exit_code == 0
    finally:
        os.remove(temp_path)
