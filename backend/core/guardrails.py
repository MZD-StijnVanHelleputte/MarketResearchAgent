"""Explicit safety enforcement: tool allowlist, injection detection, network kill-switch.

Cost and API-call counts are tracked live (see models/usage.py) but are NOT capped —
runs are never aborted for spend or call volume.
"""
import re

from config import settings


class GuardrailViolation(Exception):
    pass


class ToolNotAllowed(GuardrailViolation):
    pass


_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+|previous\s+|above\s+|prior\s+)?instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(a|an|if)\b", re.IGNORECASE),
    re.compile(r"forget\s+everything", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(your|all|previous|above|prior|the)\b", re.IGNORECASE),
    re.compile(r"new\s+(role|instructions|persona)\b", re.IGNORECASE),
    re.compile(r"\[system\]", re.IGNORECASE),
    re.compile(r"###\s*(system|instruction)", re.IGNORECASE),
    re.compile(r"your\s+(new|updated)\s+instructions\b", re.IGNORECASE),
    re.compile(r"system\s+prompt\b", re.IGNORECASE),
    re.compile(r"override\s+.*?\binstructions\b", re.IGNORECASE),
]


class Guardrails:
    def check_tool_allowed(self, tool_name: str) -> None:
        allowlist = settings.safety.tool_allowlist
        if allowlist and tool_name not in allowlist:
            raise ToolNotAllowed(
                f"Tool '{tool_name}' is not in the configured allowlist: {allowlist}"
            )

    def check_network_write(self) -> None:
        if not settings.safety.allow_network_writes:
            raise GuardrailViolation("ALLOW_NETWORK_WRITES is off.")

    def scan_for_injection(self, text: str) -> str | None:
        """Return a warning string if instruction-like content is found; None if clean."""
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return f"Possible prompt injection detected: '{match.group(0)}'"
        return None
