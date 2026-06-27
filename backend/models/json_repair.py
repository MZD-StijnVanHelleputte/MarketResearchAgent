"""Best-effort recovery of a single string field from a not-quite-valid JSON LLM response.

LLM agents are asked to respond with a JSON envelope (e.g. {"domain": ..., "text": ...})
but occasionally wrap it in stray prose, truncate it, or mis-escape a quote. Silently
falling back to the raw response text leaks the JSON envelope into user-facing prose, so
callers should treat a `None` return as a real failure and use their own placeholder/
fallback content instead.
"""
import json
import re


def extract_json_field(raw: str, field: str) -> str | None:
    """Extract *field*'s string value from *raw*, tolerating common LLM JSON mistakes.

    Tries, in order: direct json.loads, json.loads on the substring between the first
    "{" and last "}" (strips wrapper prose), then a regex pull of just the field's
    value (survives one truncated/malformed sibling key). Returns None if all fail.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("```")
        ).strip()

    for candidate in (text, _braces_substring(text)):
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get(field), str):
            return data[field]

    match = re.search(rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if match:
        try:
            return json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            return match.group(1)

    return None


def _braces_substring(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
