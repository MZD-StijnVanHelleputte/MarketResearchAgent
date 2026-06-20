from abc import ABC, abstractmethod
from pydantic import BaseModel


class BaseTool(ABC):
    name: str
    description: str
    input_schema: type[BaseModel]
    # Set to "fmp" or "alpha_vantage" on tools that require a paid subscription.
    # The registry reads this to drop tools when the corresponding tier is "free".
    requires_premium: str | None = None

    @abstractmethod
    async def run(self, **kwargs) -> dict: ...
