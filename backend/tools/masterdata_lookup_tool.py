import asyncio
from pydantic import BaseModel
from tools.base import BaseTool
from services.masterdata_service import MasterDataService


class MasterdataLookupInput(BaseModel):
    entity_type: str
    region: str = ""
    keyword: str = ""


class MasterdataLookupTool(BaseTool):
    name = "masterdata_lookup"
    description = (
        "Query Komatsu's internal master-data registry. "
        "entity_type must be one of: distributors, competitors, operators, construction, "
        "others, equipment, commodities. "
        "Optionally filter by region (e.g. 'Asia-Pacific', 'Americas') or keyword "
        "(substring match on name, country, ticker, category, or product). "
        "Use entity_type='commodities' to retrieve valid commodity symbols before calling "
        "get_mining_metals_prices, get_energy_cost_prices, or get_broad_commodity_cycle. "
        "Use entity_type='operators' to identify known mining operators (Komatsu's customers) "
        "with their tickers, exchanges, and primary commodities; operators flagged is_private "
        "are not listed, so source their financials via other means. "
        "Use entity_type='construction' for construction & infrastructure contractors "
        "(Komatsu customers, from civil/marine works to residential developers) and "
        "entity_type='others' for niche industrial buyers (metals recyclers, steelmakers, "
        "pulp/paper producers) — both are also Komatsu customer segments. "
        "Use to identify known distributors, competitor tickers, customer entity names, "
        "and equipment model families before building external search queries."
    )
    input_schema = MasterdataLookupInput

    def __init__(self, service: MasterDataService | None = None) -> None:
        self._service = service or MasterDataService()

    async def run(self, **kwargs) -> dict:
        inp = MasterdataLookupInput(**kwargs)
        results = await asyncio.to_thread(
            self._service.lookup,
            inp.entity_type,
            inp.region,
            inp.keyword,
        )
        return {"entity_type": inp.entity_type, "results": results}
