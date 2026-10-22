"""Chat message helpers.

Graphflow messages are plain dicts in the common chat format::

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [{"id": str, "name": str, "args": dict}],   # assistant only
     "tool_call_id": str, "name": str}                          # tool only
"""

from __future__ import annotations

from typing import Any

_ROLE_BY_LC_TYPE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def normalize_message(m: Any) -> dict[str, Any]:
    """Converts a dict, LangChain message or string into a Graphflow chat dict."""
    if isinstance(m, dict):
        msg = dict(m)
        if "role" not in msg:
            msg["role"] = _ROLE_BY_LC_TYPE.get(msg.pop("type", "user"), "user")
        msg.setdefault("content", "")
        return msg
    if hasattr(m, "content") and hasattr(m, "type"):
        msg = {"role": _ROLE_BY_LC_TYPE.get(m.type, "user"), "content": m.content}
        if getattr(m, "tool_calls", None):
            msg["tool_calls"] = [
                {"id": tc.get("id") or "", "name": tc["name"], "args": tc.get("args") or {}}
                for tc in m.tool_calls
            ]
        if getattr(m, "tool_call_id", None):
            msg["tool_call_id"] = m.tool_call_id
            msg["name"] = getattr(m, "name", None) or ""
        return msg
    return {"role": "user", "content": str(m)}


def strip_code_fences(text: str) -> str:
    """Removes a surrounding Markdown code fence (```json ... ```), if any."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
