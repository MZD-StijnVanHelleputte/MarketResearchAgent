"""Prompt builder for the Tree-of-Thought plan proposer."""
from __future__ import annotations

from retrieval.chroma_store import Chunk

_DOMAINS = [
    "competition", "distributors", "customers", "mining_projects",
    "commodities", "macro_geopolitics", "general_search",
]

# Collect-stage tool names — must match the registered tool `.name` values in
# tools/registry.py exactly, or grounding drops the call and routing KeyErrors.
_TOOLS = [
    "news_search", "news_top_headlines",
    "get_mining_metals_prices", "get_energy_cost_prices",
    "get_broad_commodity_cycle",
    "get_company_financials", "search_sec_filings", "get_mine_technical_report",
    "get_equity_price",
    "get_equity_history", "get_equity_financials", "get_fred_observations",
    "web_search", "masterdata_lookup", "get_macro_indicator",
]

_SYSTEM = """\
You are a strategic research planner for Komatsu's market intelligence system.
Given a research query and relevant background context, generate exactly {n} \
diverse execution plans.

Each plan specifies:
- Which of the 7 intelligence domains to activate
- Which data-gathering tools to call, with what arguments
- A rationale explaining what angle this plan investigates
- An estimated token cost (integer, rough estimate)

Available domains: competition, distributors, customers, mining_projects, \
commodities, macro_geopolitics, general_search

Domain note: "competition" covers equipment-maker rivals only (Caterpillar, Hitachi \
Construction Machinery, Sandvik, …) — companies that sell against Komatsu. "customers" \
covers mining operators only (BHP, Rio Tinto, Vale, …) — companies that buy equipment \
and are Komatsu's customers, not its rivals. Never route a mining operator's tool calls \
into "competition", or an equipment maker's into "customers".

Demand-side note: commodity CONSUMERS (EV/battery/auto OEMs and other large buyers — \
BYD, Volkswagen, Tesla, CATL, …) are neither rivals nor mining operators. When the \
research findings list demand-side companies, route their tool calls (`get_equity_price`, \
`get_equity_history`, `get_equity_financials`, `get_company_financials`, `news_search`, \
`web_search`) into "macro_geopolitics" as the demand-side angle — their commodity demand \
moves prices and belongs in the macro/demand picture, not "competition" or "customers".

Available tools: news_search, news_top_headlines, \
get_mining_metals_prices, get_energy_cost_prices, \
get_broad_commodity_cycle, \
get_company_financials, search_sec_filings, get_mine_technical_report, get_equity_price, \
get_equity_history, \
get_equity_financials, get_fred_observations, web_search, \
masterdata_lookup, get_macro_indicator

Tool usage guidance:
- `get_mine_technical_report` pulls the SEC S-K 1300 Technical Report Summary (mineral
  reserves, mine life, project economics) for one mining company ticker — use it for
  customers/mining_projects plans whenever a mine site's operator ticker is known, pairing
  it with `mine_name` to target a specific project for diversified miners.
- `get_mining_metals_prices`'s `symbol` argument must be one of COPPER, ALUMINUM,
  GOLD, SILVER, XAU, XAG — never a company/equity ticker. If a mining operator
  (BHP, Rio Tinto, Vale, …) is relevant, map it to the commodity it produces and use
  that commodity code here instead of the operator's ticker.
- `get_equity_price` is a same-day snapshot only — never rely on it alone for a
  trend question. Pair it with `get_equity_history` (OHLCV, supports up to 5y via
  the `period` argument) and `get_equity_financials` (annual or quarterly
  income-statement line items) whenever the query concerns market performance or
  financial trends over time, not just the current state.
- `get_fred_observations` supports arbitrary date ranges and frequency
  aggregation (daily/weekly/monthly/quarterly/annual) — use it for macro series
  instead of a single-point lookup.
- When the research findings list multiple stock tickers, plans should call
  ticker-scoped tools (`get_equity_price`, `get_equity_history`,
  `get_equity_financials`, `get_company_financials`) for EVERY ticker listed, not
  just one — each tracked competitor needs its own data point for a real
  comparison. Do not economize by picking a single representative ticker.
- Competitor analysis is not just numbers: for each competitor, pair the
  financial/equity calls with `news_search` (recent articles), `news_top_headlines`
  (breaking news), and `web_search` (official communications, IR pages) to capture
  recent announcements, strategy shifts, and product launches — a competitor with
  only stock/financial data and no news is an incomplete picture.
- Favor calling more tools over fewer when the query warrants it — the system
  budget allows up to 100 tool calls per run, so under-using available tools
  produces a weaker, thinner report.

DIVERSITY REQUIREMENT: Plans must differ on at least {min_dims} of the \
following dimensions:
1. Domain activation set (which domains are enabled)
2. Primary data source (financial APIs vs news vs web search vs filings)
3. Temporal focus (current snapshot vs historical trend vs forward-looking)
4. Entity focus (specific companies vs broad sector vs macro indicators)
5. Analysis angle (supply-side vs demand-side vs competitive vs macro)

Respond with ONLY a JSON object in this exact format (no other text):
{{
  "plans": [
    {{
      "plan_id": "plan_001",
      "domain_activations": {{"competition": true, "distributors": false, \
"customers": false, "mining_projects": false, "commodities": true, \
"macro_geopolitics": false, "general_search": false}},
      "entity_choices": {{"company": "Caterpillar", "commodity": "copper"}},
      "api_assignments": {{"financials": "fmp", "prices": "alpha_vantage"}},
      "tool_calls": [
        {{"tool": "get_company_financials", "domain": "competition", \
"arguments": {{"ticker": "CAT"}}}},
        {{"tool": "get_equity_history", "domain": "competition", \
"arguments": {{"ticker": "CAT", "period": "5y"}}}},
        {{"tool": "get_equity_financials", "domain": "competition", \
"arguments": {{"ticker": "CAT", "period": "annual"}}}},
        {{"tool": "get_mining_metals_prices", "domain": "commodities", \
"arguments": {{"symbol": "COPPER", "interval": "monthly"}}}}
      ],
      "estimated_token_cost": 1200,
      "rationale": "Focus on direct competitive financials and commodity input costs.",
      "depth": 1,
      "feasibility_score": 0.0,
      "quality_score": 0.0,
      "combined_score": 0.0,
      "gap_report": [],
      "receives_diversity_penalty": false,
      "is_survivor": false
    }}
  ]
}}
"""

_USER_TEMPLATE = """\
Research query: {query}

Background context from the industry knowledge base:
{context}
{research_section}
Generate exactly {n} diverse plans following the diversity requirement above.
Use the resolved entities and research findings above to make plans concrete \
(specific tickers, commodity symbols, named competitors, news query strings).
"""

_RESEARCH_SECTION_TEMPLATE = """\

Pre-planning research findings (use these to make tool calls specific):
- Competitors identified (equipment-maker rivals — route to "competition"): {competitors}
- Mining operators identified (Komatsu's customers — route to "customers"): {operators}
- Demand-side consumers (commodity buyers — route to "macro_geopolitics"): {demand_side_companies}
- Stock tickers (call ticker-scoped tools for EVERY one of these, not just one): {tickers}
- Commodities (with symbols): {commodities}
- Mine sites / regions: {regions}
- Recent news signals: {news_signals}
- Open questions (items still to resolve during collect): {open_questions}
"""


def propose_messages(
    query: str,
    context_chunks: list[Chunk],
    n: int,
    min_dims: int,
    research_context: "ResearchContext | None" = None,
) -> list[dict]:
    from core.tot.schemas import ResearchContext  # local import to avoid circularity

    context_text = "\n\n".join(
        f"[{c.domain}] {c.text}" for c in context_chunks[:5]
    ) or "(no background context available)"

    if research_context and any([
        research_context.competitors, research_context.operators, research_context.tickers,
        research_context.commodities, research_context.news_signals,
    ]):
        research_section = _RESEARCH_SECTION_TEMPLATE.format(
            competitors=", ".join(research_context.competitors) or "none identified",
            operators=", ".join(research_context.operators) or "none identified",
            demand_side_companies=", ".join(research_context.demand_side_companies) or "none identified",
            tickers=", ".join(research_context.tickers) or "none identified",
            commodities=", ".join(research_context.commodities) or "none identified",
            regions=", ".join(research_context.mine_sites + research_context.regions) or "none identified",
            news_signals="; ".join(research_context.news_signals[:3]) or "none",
            open_questions="; ".join(research_context.open_questions) or "none",
        )
    else:
        research_section = ""

    return [
        {
            "role": "system",
            "content": _SYSTEM.format(n=n, min_dims=min_dims),
        },
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                query=query,
                context=context_text,
                research_section=research_section,
                n=n,
            ),
        },
    ]
