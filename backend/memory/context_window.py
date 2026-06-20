class ContextWindow:
    """Token budgeting, message history, and prompt pruning.
    Holds compact structured result objects (plan, collection manifest, signal summary),
    not raw sub-agent output."""

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._messages: list[dict] = []  # [{role, content, token_count}]

    def add(self, role: str, content: str) -> None:
        token_count = max(1, len(content) // 4)
        self._messages.append({"role": role, "content": content, "token_count": token_count})
        self.prune()

    def prune(self) -> None:
        while self._total_tokens() > self._budget and len(self._messages) > 1:
            self._messages.pop(0)

    def _total_tokens(self) -> int:
        return sum(m["token_count"] for m in self._messages)

    def to_prompt_messages(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def as_messages(self) -> list[dict]:
        return self.to_prompt_messages()

    @classmethod
    def from_state(cls, messages: list[dict], budget: int) -> "ContextWindow":
        cw = cls(budget)
        cw._messages = list(messages)
        return cw

    def to_state(self) -> list[dict]:
        return list(self._messages)
