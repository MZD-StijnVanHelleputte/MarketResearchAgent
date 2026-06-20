import dataclasses

from pydantic import BaseModel, Field

from services.forex_service import ForexService
from tools.base import BaseTool


class FxRatesInput(BaseModel):
    from_currency: str = Field(
        description=(
            "ISO 4217 base currency code, e.g. USD. "
            "Mining-relevant pairs: USD/AUD (Australia), USD/BRL (Brazil), "
            "USD/CLP (Chile copper), USD/ZAR (South Africa), USD/JPY (Komatsu HQ)."
        )
    )
    to_currency: str = Field(description="ISO 4217 quote currency code, e.g. AUD.")
    interval: str = Field(
        default="monthly",
        pattern=r"^(daily|weekly|monthly)$",
        description="daily, weekly, or monthly.",
    )


class FxRatesTool(BaseTool):
    name = "get_fx_rates"
    description = (
        "Get Alpha Vantage FX exchange-rate series (FX_DAILY, FX_WEEKLY, or FX_MONTHLY). "
        "Use to assess currency risk for mining projects and Komatsu equipment pricing. "
        "Key pairs: USD/AUD (Australian mines), USD/BRL (Brazil), USD/CLP (Chile copper), "
        "USD/ZAR (South Africa), USD/JPY (Komatsu). "
        "Note: FRED already covers JPY/USD (DEXJPUS) and CNY/USD (DEXCHUS); use this tool "
        "for AUD, BRL, CLP, ZAR, and other mining-country currencies. "
        "Returns from_currency, to_currency, interval, latest OHLC observation, all rows, and source."
    )
    input_schema = FxRatesInput

    def __init__(self, service: ForexService | None = None) -> None:
        self._service = service or ForexService()

    async def run(self, **kwargs) -> dict:
        inp = FxRatesInput(**kwargs)
        result = await self._service.get_fx_rates(inp.from_currency, inp.to_currency, inp.interval)
        return dataclasses.asdict(result)
