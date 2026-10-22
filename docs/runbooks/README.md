# Runbooks

Step-by-step operational procedures a person (or an agent, under supervision) follows
under pressure: deploying, rolling back, restoring data, rotating credentials,
handling an incident.

No runbooks exist yet — add one per procedure as the project grows, named for the
situation (`rollback-deploy.md`, `restore-database.md`, `rotate-api-keys.md`).

## Shape of a good runbook

1. **When to use this** — the symptom or situation, stated so it's matchable at 3 a.m.
2. **Preconditions** — access needed, state to confirm before acting.
3. **Steps** — numbered, copy-pasteable commands, each with its expected output so the
   operator knows immediately when something went sideways.
4. **Verification** — how to confirm the system is healthy afterward.
5. **Escalation** — who/what to invoke if the runbook doesn't resolve it.

Runbooks are executable documentation: after every real use, fix whatever step didn't
match reality.
