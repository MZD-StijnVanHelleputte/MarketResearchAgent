"""Single source of truth for research domains and leaf-type toolsets.

A research plan is a tree: each DOMAIN (stem) holds LEAVES (units of research —
a company, a country, a commodity, a mine site, a topic). A leaf's TYPE
deterministically selects which tools collect its data (``LEAF_TOOLSETS``), so
collection is an execution phase rather than a per-call LLM guess. Each domain is
grounded in master data, so an entity (e.g. Caterpillar) is owned by exactly one
domain and overlap is impossible by construction — see
``services.masterdata_service.MasterDataService.resolve_entity``.

This module replaces the parallel hardcoded domain lists that used to live in
core/graph.py, config/settings.py, core/tot/schemas.py, tools/registry.py,
core/friendly_names.py and the per-domain agent classes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainSpec:
    """One research domain (a stem in the plan tree)."""

    key: str
    display_name: str
    # MasterDataService getter key (see MasterDataService._ENTITY_MAP), or None
    # when the domain's entities are surfaced by research rather than master data.
    masterdata_source: str | None
    # Leaf types this domain may hold; the first is the default. Drives both the
    # derived tool allowlist (DOMAIN_TOOLS) and leaf-type inference at plan time.
    leaf_types: tuple[str, ...]
    # Lower wins when a dataset could be attributed to more than one domain.
    ownership_priority: int
    # Whether synthesis decomposes the domain into per-entity subchapters.
    decomposable: bool
    # Persona handed to the generic collection agent (agents/__init__.py).
    role: str
    goal: str
    backstory: str

    @property
    def default_leaf_type(self) -> str:
        return self.leaf_types[0]


# ---------------------------------------------------------------------------
# Leaf type → deterministic default toolset
# ---------------------------------------------------------------------------
# Premium-gated tools may appear here; tools/registry.py's tier filter drops them
# at runtime when the relevant API tier is "free".
LEAF_TOOLSETS: dict[str, list[str]] = {
    "company": [
        "get_company_financials",
        "get_equity_price",
        "get_equity_history",
        "get_equity_financials",
        "get_income_statement",
        "get_balance_sheet",
        "get_cash_flow",
        "news_search",
        "news_top_headlines",
        "web_search",
        "search_sec_filings",
    ],
    "commodity": [
        "get_mining_metals_prices",
        "get_energy_cost_prices",
        "get_broad_commodity_cycle",
        "get_agricultural_commodity_prices",
    ],
    "distributor": [
        "masterdata_lookup",
        "news_search",
        "news_top_headlines",
        "web_search",
    ],
    "country": [
        "get_macro_indicator",
        "get_fred_observations",
        "search_fred_series",
        "get_fx_rates",
        "news_search",
        "web_search",
        "web_research",
    ],
    "mine_site": [
        "get_mine_technical_report",
        "search_sec_filings",
        "get_mining_metals_prices",
        "news_search",
        "web_search",
        "web_extract",
    ],
    "topic": [
        "web_search",
        "web_research",
        "web_extract",
        "web_map",
        "news_search",
        "news_top_headlines",
        "news_sources",
        "get_news_sentiment",
    ],
}


# ---------------------------------------------------------------------------
# Domain registry (9 domains)
# ---------------------------------------------------------------------------
DOMAINS: dict[str, DomainSpec] = {
    "commodities": DomainSpec(
        key="commodities",
        display_name="Commodities",
        masterdata_source="commodities",
        leaf_types=("commodity",),
        ownership_priority=1,
        decomposable=True,
        role="Commodities & Cycles Analyst",
        goal=(
            "Monitor commodity price trends — especially copper, gold, iron ore, and "
            "thermal coal — and interpret their implications for mining equipment demand "
            "cycles."
        ),
        backstory=(
            "You are an expert in commodity markets and their downstream effects on mining "
            "capex. You know that copper above $4/lb typically unlocks new mine development, "
            "that gold price volatility drives gold miner equipment deferrals, and that iron "
            "ore spreads affect the major diversified miners' equipment budgets. You "
            "translate commodity signals into concrete demand outlook statements for Komatsu."
        ),
    ),
    "competition": DomainSpec(
        key="competition",
        display_name="Competition",
        masterdata_source="competitors",
        leaf_types=("company",),
        ownership_priority=2,
        decomposable=True,
        role="Competitive Intelligence Analyst",
        goal=(
            "Analyse Caterpillar, Volvo CE, Liebherr, and Epiroc financials, product "
            "launches, and strategic moves to surface competitive risks and opportunities "
            "for Komatsu."
        ),
        backstory=(
            "You are a specialist in heavy equipment OEM competitive analysis. You have deep "
            "knowledge of Caterpillar's capex cycles, Volvo CE's electrification roadmap, "
            "Liebherr's niche product strategy, and Epiroc's autonomous mining push. You turn "
            "financial data and news signals into crisp competitive intelligence that a "
            "Komatsu strategy lead can act on. You cover equipment-maker rivals only — "
            "companies that sell against Komatsu, never its customers."
        ),
    ),
    "mining_operators": DomainSpec(
        key="mining_operators",
        display_name="Mining Operators",
        masterdata_source="operators",
        leaf_types=("company",),
        ownership_priority=3,
        decomposable=True,
        role="Mining Customer Demand Analyst",
        goal=(
            "Track equipment purchasing intentions, capex budgets, and fleet renewal cycles "
            "across Komatsu's mining-operator customer base (BHP, Rio Tinto, Vale, and "
            "peers)."
        ),
        backstory=(
            "You focus on the mining demand side of heavy equipment markets. You analyse "
            "mining company capex announcements, production guidance, and fleet utilisation "
            "to identify where Komatsu can capture incremental volume. You distinguish "
            "short-term project-driven demand from structural fleet replacement cycles. "
            "These companies buy Komatsu equipment — they are customers, never rivals."
        ),
    ),
    "construction_companies": DomainSpec(
        key="construction_companies",
        display_name="Construction Companies",
        masterdata_source="construction",
        leaf_types=("company",),
        ownership_priority=4,
        decomposable=True,
        role="Construction Customer Demand Analyst",
        goal=(
            "Track project pipelines, tender wins, and equipment capital plans across "
            "Komatsu's construction & infrastructure customers — from civil and marine "
            "works through to large residential developers."
        ),
        backstory=(
            "You focus on construction and infrastructure contractors as a Komatsu customer "
            "segment. You analyse construction tender pipelines, project wins, and capital "
            "plans to identify equipment demand. You distinguish project-driven demand from "
            "structural fleet replacement. These contractors buy Komatsu equipment — they "
            "are customers, never rivals."
        ),
    ),
    "specialized_customers": DomainSpec(
        key="specialized_customers",
        display_name="Specialized Customers",
        masterdata_source="others",
        leaf_types=("company",),
        ownership_priority=5,
        decomposable=True,
        role="Industrial Customer Demand Analyst",
        goal=(
            "Track equipment capital plans of niche industrial buyers — metals recyclers, "
            "steelmakers, smelters, and pulp/paper producers — that operate Komatsu "
            "equipment outside the core mining and construction segments."
        ),
        backstory=(
            "You focus on specialised industrial buyers of heavy equipment such as metals "
            "recyclers, steelmakers, and pulp/paper producers. You analyse their capital "
            "plans and operations to identify Komatsu demand. These firms buy Komatsu "
            "equipment — they are customers, never rivals, and are kept distinct from mining "
            "and construction customers."
        ),
    ),
    "distributors": DomainSpec(
        key="distributors",
        display_name="Distributors",
        masterdata_source="distributors",
        leaf_types=("distributor",),
        ownership_priority=6,
        decomposable=True,
        role="Dealer Network Analyst",
        goal=(
            "Monitor the health and performance of Komatsu's dealer network and track moves "
            "by competing OEMs to win or defend distribution relationships."
        ),
        backstory=(
            "You specialise in heavy equipment distribution channels. You track dealer "
            "consolidation trends, aftermarket revenue splits, and OEM incentive programmes "
            "across North America, Europe, and Asia-Pacific. You surface signals that "
            "indicate channel risk or opportunity for Komatsu's sales organisation."
        ),
    ),
    "mining_projects": DomainSpec(
        key="mining_projects",
        display_name="Mining Projects",
        masterdata_source=None,
        leaf_types=("mine_site",),
        ownership_priority=7,
        decomposable=True,
        role="Mining Projects Analyst",
        goal=(
            "Identify active, planned, and pipeline mining projects globally that represent "
            "equipment procurement opportunities or risks for Komatsu."
        ),
        backstory=(
            "You track the global mining project pipeline — from feasibility studies through "
            "construction to production ramp-up. You monitor SEC filings, news, and industry "
            "databases to spot new project announcements, expansion decisions, and mine "
            "closures. You quantify the equipment demand implications for Komatsu's mining "
            "division. Mine sites are projects, distinct from the operators that run them."
        ),
    ),
    "macroeconomics": DomainSpec(
        key="macroeconomics",
        display_name="Macroeconomics",
        masterdata_source=None,
        leaf_types=("country",),
        ownership_priority=8,
        decomposable=True,
        role="Macroeconomics Analyst",
        goal=(
            "Assess macroeconomic conditions by country/region — interest rates, industrial "
            "and mining production, construction spending, inflation, and FX — that affect "
            "Komatsu's global equipment markets."
        ),
        backstory=(
            "You specialise in macroeconomics for industrial companies. You track interest "
            "rate cycles (construction financing), infrastructure spending in key markets, "
            "industrial/mining production, and currency moves in major mining jurisdictions, "
            "connecting macro signals to Komatsu's order book and margin outlook. FRED "
            "covers standard (mostly US) macro series — rates, production, construction "
            "spending, commodity prices, FX. For country-specific infrastructure spending "
            "and policy context, use web_search and web_research rather than FRED."
        ),
    ),
    "general_search": DomainSpec(
        key="general_search",
        display_name="General Market Search",
        masterdata_source=None,
        leaf_types=("topic", "company"),
        ownership_priority=9,
        decomposable=True,
        role="Open-Web Intelligence Analyst",
        goal=(
            "Capture market intelligence signals from the open web that fall outside "
            "structured data feeds — product announcements, executive commentary, analyst "
            "opinions, emerging trends, and the demand-side consumer companies (EV, "
            "battery, auto OEMs) whose appetite drives commodity demand."
        ),
        backstory=(
            "You are a broad-based intelligence researcher skilled at extracting signal from "
            "noise on the open web. You identify analyst reports, industry conference "
            "summaries, executive interviews, and technology trend articles that could "
            "affect Komatsu's strategic position. You also cover third-party demand-side "
            "consumers (BYD, Volkswagen, Tesla, CATL, …) as the downstream demand angle on "
            "commodities — they are neither Komatsu's rivals nor its customers. You are "
            "selective — you surface only items with clear relevance and discard noise."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Derived helpers (read by everything that used to hardcode the domain list)
# ---------------------------------------------------------------------------

def domain_keys() -> list[str]:
    """All domain keys, in ownership-priority order."""
    return [s.key for s in sorted(DOMAINS.values(), key=lambda s: s.ownership_priority)]


def get_spec(domain: str) -> DomainSpec | None:
    return DOMAINS.get(domain)


def display_name(domain: str) -> str:
    spec = DOMAINS.get(domain)
    return spec.display_name if spec else domain.replace("_", " ").title()


def ownership_order() -> list[str]:
    """Domain keys ordered by ownership priority (earlier entries win ties)."""
    return domain_keys()


def decomposable_domains() -> set[str]:
    return {s.key for s in DOMAINS.values() if s.decomposable}


def masterdata_source(domain: str) -> str | None:
    spec = DOMAINS.get(domain)
    return spec.masterdata_source if spec else None


def leaf_tools(leaf_type: str) -> list[str]:
    return list(LEAF_TOOLSETS.get(leaf_type, []))


def domain_tools(domain: str) -> list[str]:
    """Union of the toolsets for every leaf type the domain may hold, order-preserving."""
    spec = DOMAINS.get(domain)
    if spec is None:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for lt in spec.leaf_types:
        for tool in LEAF_TOOLSETS.get(lt, []):
            if tool not in seen:
                seen.add(tool)
                out.append(tool)
    return out
