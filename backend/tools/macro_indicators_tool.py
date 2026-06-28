import dataclasses

from pydantic import BaseModel, Field

from services.macro_service import MacroService
from tools.base import BaseTool


class MacroIndicatorsInput(BaseModel):
    series_id: str
    limit: int = Field(default=10, ge=1, le=50)


class MacroIndicatorsTool(BaseTool):
    name = "get_macro_indicator"
    description = (
        "Fetch a macroeconomic time series from FRED (Federal Reserve Economic Data). "
        "Returns the most recent observations with title and units.\n"
        "Pass an exact FRED series_id. Pick from this verified catalogue of "
        "Komatsu-relevant indicators (or use search_fred_series to discover others):\n"
        "  Rates & credit: FEDFUNDS (Fed funds rate), DGS10 (10y Treasury), DGS2 (2y Treasury), "
        "MORTGAGE30US (30y mortgage), BAA10Y (Baa-Treasury spread)\n"
        "  Construction & housing: TTLCONS (total construction spending), TLNRESCONS (nonresidential "
        "construction), PRRESCONS (residential construction), HOUST (housing starts), PERMIT (building permits)\n"
        "  Industrial & mining output: INDPRO (industrial production), IPMAN (mfg production), "
        "IPMINE (mining production), IPG2122S (metal-ore mining), MCUMFN (mfg capacity utilization)\n"
        "  Capital goods orders: DGORDER (durable goods orders), NEWORDER (core capital-goods orders)\n"
        "  Commodities & energy: DCOILWTICO (WTI crude), DCOILBRENTEU (Brent crude), DHHNGSP (natural gas), "
        "PCOPPUSDM (copper), PIORECRUSDM (iron ore), PALUMUSDM (aluminum), PCOALAUUSDM (coal), "
        "PALLFNFINDEXM (all-commodity index), WPU101 (iron & steel PPI), PPIACO (PPI all commodities)\n"
        "  FX: DEXJPUS (JPY/USD), DEXCHUS (CNY/USD), DEXUSAL (USD/AUD), DEXUSEU (USD/EUR), "
        "DEXCAUS (CAD/USD), DEXBZUS (BRL/USD)\n"
        "  Output & prices: GDP (US GDP), GDPC1 (US real GDP), JPNRGDPEXP (Japan real GDP), "
        "CPIAUCSL (CPI), CPILFESL (core CPI), PCEPI (PCE price index)\n"
        "  Labor: PAYEMS (nonfarm payrolls), MANEMP (mfg employment), CES1021100001 (mining employment), "
        "UNRATE (unemployment rate)\n"
        "  Trade: BOPGSTB (trade balance), IMPCH (imports from China), EXPCH (exports to China)\n"
        "Use one of these series_ids. If you need a concept that is NOT listed, FIRST call "
        "search_fred_series to discover the real id, then pass that id here — never invent an id. "
        "FRED only covers standard (mostly US) macro series; for tariffs, trade-policy, geopolitical "
        "risk, or country-specific infrastructure spending use web_search/web_research instead."
    )
    input_schema = MacroIndicatorsInput

    def __init__(self, service: MacroService | None = None) -> None:
        self._service = service or MacroService()

    async def run(self, **kwargs) -> dict:
        inp = MacroIndicatorsInput(**kwargs)
        indicator = await self._service.get_indicator(inp.series_id, inp.limit)
        return dataclasses.asdict(indicator)
