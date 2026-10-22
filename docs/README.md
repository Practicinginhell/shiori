# docs/ — Human Documentation

Written for people first. The agent-facing knowledge is in [ai/](../ai/), and the
always-on rules are in [AGENTS.md](../AGENTS.md).

| Document | Contains |
| --- | --- |
| [QUICKSTART.md](QUICKSTART.md) | User guide: agents, workflows, tools, approvals, middleware, retries, time travel, run context, threads, CLI |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The design in depth: layers, IR, compilation, agent subgraphs, runtime |
| [TTC.md](TTC.md) | `graphflow.ttc`, the implementation of arXiv:2609.21032 |
| [adr/](adr/) | Architecture Decision Records: the *why* behind structural choices |
| [runbooks/](runbooks/) | Operational procedures (none yet) |

## Where a new document belongs

- Explains **how the system works** for agents making changes → `ai/context/`
- Records **a decision and its reasoning** → `adr/` (copy `ai/templates/adr.md`)
- Tells a human **how to operate something** → `runbooks/`
- Teaches **users** the framework → `QUICKSTART.md` (its code blocks must run)
