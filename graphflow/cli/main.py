"""Graphflow Command Line Interface."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import click

from graphflow import __version__
from graphflow.core.application import Application


def _load_app_from_file(file_path: str) -> Application:
    """Dynamically loads an Application instance from a Python file."""
    path = Path(file_path).resolve()
    if not path.exists():
        click.echo(f"Error: File '{file_path}' does not exist.", err=True)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("user_app_module", path)
    if not spec or not spec.loader:
        click.echo(f"Error: Could not load module from '{file_path}'.", err=True)
        sys.exit(1)

    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_app_module"] = mod
    spec.loader.exec_module(mod)

    # Look for an Application instance
    for attr in ("app", "application"):
        val = getattr(mod, attr, None)
        if isinstance(val, Application):
            return val

    # Search all module attributes
    for v in vars(mod).values():
        if isinstance(v, Application):
            return v

    click.echo(f"Error: No Application instance found in '{file_path}'.", err=True)
    sys.exit(1)


def _text_input(app: Application, text: str) -> dict[str, str]:
    """Builds run input for a text prompt using whichever of query/input the schema declares."""
    fields = app.compile().ir.state_schema.fields
    keys = [k for k in ("query", "input") if k in fields] or ["query", "input"]
    if not fields:  # default agent schema accepts both
        keys = ["query"]
    return {k: text for k in keys}


@click.group()
@click.version_option(version=__version__, prog_name="graphflow")
def cli() -> None:
    """Graphflow: Production-grade Universal Framework for LangGraph."""


@cli.command()
@click.argument("project_name")
def init(project_name: str) -> None:
    """Initialize a new Graphflow project."""
    target_dir = Path(project_name)
    if target_dir.exists():
        click.echo(f"Error: Directory '{project_name}' already exists.", err=True)
        sys.exit(1)

    target_dir.mkdir(parents=True)
    app_py = target_dir / "app.py"
    app_py.write_text(
        '''from graphflow import Application, Workflow, State, Field

class GreetingState(State):
    name: str = "World"
    greeting: str = ""

def say_hello(state: dict) -> dict:
    return {"greeting": f"Hello, {state.get('name', 'World')}!"}

wf = Workflow("greeting", state_schema=GreetingState)
wf.then(say_hello)

app = Application("hello-app")
app.register(wf)

if __name__ == "__main__":
    result = app.run({"name": "Developer"})
    print("Result:", result["greeting"])
'''
    )

    readme = target_dir / "README.md"
    readme.write_text(
        f"""# {project_name}

Built with [Graphflow](https://github.com/graphflow/graphflow).

## Run
```bash
python app.py
# or
graphflow run app.py
```
"""
    )

    click.echo(f"Initialized new Graphflow project in '{project_name}'!")
    click.echo(f"Run 'cd {project_name} && python app.py' to get started.")


@cli.command()
@click.argument("app_file", default="app.py")
@click.option("--query", "-q", default="Hello", help="Input query to pass to application")
@click.option("--thread-id", "-t", default=None, help="Thread ID for persistence")
def run(app_file: str, query: str, thread_id: str | None) -> None:
    """Run an application from a file."""
    app = _load_app_from_file(app_file)
    click.echo(f"Running '{app.name}'...")
    res = app.run(_text_input(app, query), thread_id=thread_id)
    click.echo("\n--- Execution Output ---")
    for k, v in res.items():
        click.echo(f"{k}: {v}")


@cli.command()
@click.argument("app_file", default="app.py")
@click.option("--format", "-f", default="ascii", type=click.Choice(["ascii", "mermaid"]))
def graph(app_file: str, format: str) -> None:
    """Visualize the application's execution graph."""
    app = _load_app_from_file(app_file)
    visualization = app.visualize(format=format)
    click.echo(visualization)


@cli.command()
@click.argument("app_file", default="app.py")
def inspect(app_file: str) -> None:
    """Inspect the internal IR and nodes of an application."""
    app = _load_app_from_file(app_file)
    compiled = app.compile()
    ir = compiled.ir

    click.echo(f"Application: {app.name}")
    click.echo(f"Workflow: {ir.name}")
    click.echo(f"Entry Point: {ir.entry_point}")
    click.echo("\nNodes:")
    for n_id, n in ir.nodes.items():
        click.echo(f"  - {n_id} ({n.kind})")
    click.echo("\nEdges:")
    for e in ir.edges:
        click.echo(f"  - {e.source} -> {e.target}")
    if ir.conditional_edges:
        click.echo("\nConditional Edges:")
        for c in ir.conditional_edges:
            click.echo(f"  - {c.source} -> {c.route_map}")
    if ir.parallel_branches:
        click.echo("\nParallel Branches:")
        for p in ir.parallel_branches:
            click.echo(f"  - {p.source} -> {p.branch_nodes} -> {p.fan_in or '(join)'}")
    if ir.loops:
        click.echo("\nLoops:")
        for loop in ir.loops:
            click.echo(f"  - {loop.body_node} (max {loop.max_iterations}) -> {loop.exit_node or 'END'}")
    if ir.hitl_nodes:
        click.echo("\nApproval Steps:")
        for h in ir.hitl_nodes.values():
            click.echo(f"  - {h.node_id} -> {h.resume_target} (reject: {h.reject_target or 'fail'})")


@cli.command()
@click.argument("test_args", nargs=-1)
def test(test_args: tuple[str, ...]) -> None:
    """Run tests using pytest."""
    cmd = [sys.executable, "-m", "pytest"] + list(test_args)
    res = subprocess.run(cmd, check=False)  # noqa: S603 - runs this interpreter's pytest
    sys.exit(res.returncode)


@cli.command()
@click.argument("app_file", default="app.py")
def dev(app_file: str) -> None:
    """Interactive development REPL for rapid application testing."""
    app = _load_app_from_file(app_file)
    click.echo(f"Starting Graphflow Dev REPL for '{app.name}'...")
    click.echo("Type your message, or 'exit' / 'quit' to stop.\n")

    thread_id = "dev-session-1"
    while True:
        try:
            user_input = click.prompt("User", prompt_suffix=" > ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            result = app.run(_text_input(app, user_input), thread_id=thread_id)
            out = result.get("response") or result.get("output") or result
            click.echo(f"\nAssistant > {out}\n")
        except (KeyboardInterrupt, EOFError):
            break
