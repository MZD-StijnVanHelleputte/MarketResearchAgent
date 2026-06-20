import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict  # already parsed from JSON string


@dataclass
class LLMResponse:
    content: str | None               # text response; None when tool_calls is non-empty
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # {prompt_tokens, completion_tokens, total_tokens}


@dataclass
class ParsedResponse:
    thought: str
    action: str | None
    action_input: dict | None
    final_answer: str | None


def parse_plan(raw: str) -> dict:
    """Extract and validate a plan JSON from raw LLM response text.

    Handles both bare JSON and JSON wrapped in a ```json ... ``` fence.
    """
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    plan = json.loads(text)

    for required_key in ("domain_activations", "tool_calls", "rationale"):
        if required_key not in plan:
            raise ValueError(f"Plan JSON missing required key: '{required_key}'")

    if not isinstance(plan["tool_calls"], list):
        raise ValueError("Plan 'tool_calls' must be a list")

    for i, tc in enumerate(plan["tool_calls"]):
        for tc_key in ("tool", "domain", "arguments"):
            if tc_key not in tc:
                raise ValueError(
                    f"tool_calls[{i}] missing required key: '{tc_key}'"
                )

    return plan


def parse_plan_list(raw: str) -> list[dict]:
    """Extract a JSON array of plans from raw LLM text.

    Accepts either a bare JSON array, a JSON object with a 'plans' key, or
    a code-fenced JSON block containing either of the above.
    """
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    parsed = json.loads(text)

    if isinstance(parsed, list):
        plans = parsed
    elif isinstance(parsed, dict) and "plans" in parsed:
        plans = parsed["plans"]
    else:
        raise ValueError("Expected a JSON array or {\"plans\": [...]} object")

    if not plans:
        raise ValueError("parse_plan_list: result is an empty list")

    return plans


def parse_react(raw: str) -> ParsedResponse:
    """Parse raw LLM output into ReAct fields (Thought / Action / Action Input / Final Answer)."""
    thought = ""
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None

    current_section: str | None = None
    buffer: list[str] = []

    def flush(section: str, lines: list[str]) -> None:
        nonlocal thought, action, action_input, final_answer
        content = "\n".join(lines).strip()
        if section == "thought":
            thought = content
        elif section == "action":
            action = content
        elif section == "action_input":
            try:
                action_input = json.loads(content)
            except json.JSONDecodeError:
                action_input = {"raw": content}
        elif section == "final_answer":
            final_answer = content

    for line in raw.splitlines():
        lower = line.lower()
        if lower.startswith("thought:"):
            if current_section:
                flush(current_section, buffer)
            current_section = "thought"
            buffer = [line[len("thought:"):].strip()]
        elif lower.startswith("action input:"):
            if current_section:
                flush(current_section, buffer)
            current_section = "action_input"
            buffer = [line[len("action input:"):].strip()]
        elif lower.startswith("action:"):
            if current_section:
                flush(current_section, buffer)
            current_section = "action"
            buffer = [line[len("action:"):].strip()]
        elif lower.startswith("final answer:"):
            if current_section:
                flush(current_section, buffer)
            current_section = "final_answer"
            buffer = [line[len("final answer:"):].strip()]
        else:
            buffer.append(line)

    if current_section:
        flush(current_section, buffer)

    return ParsedResponse(
        thought=thought,
        action=action,
        action_input=action_input,
        final_answer=final_answer,
    )
