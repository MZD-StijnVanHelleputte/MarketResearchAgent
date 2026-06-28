from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal

from core.domains import decomposable_domains as _registry_decomposable_domains

# Single source of truth for the Mistral model fallback (used when LLM__MODEL is unset).
MISTRAL_DEFAULT_MODEL = "mistral-medium-latest"

# Anchor on-disk storage to the backend package dir, not the process's cwd — the same
# pattern services/masterdata_service.py uses for data/. Without this, sqlite_path's
# default of "./outputs/..." silently resolves to a different file depending on
# whether uvicorn was launched from backend/ or the repo root.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class LLMSettings(BaseSettings):
    provider: Literal["mistral"] = "mistral"
    model: str = MISTRAL_DEFAULT_MODEL
    propose_temperature: float = 0.9
    work_temperature: float = 0.2
    max_tokens: int = 8192
    # PlanProposer asks for `branching_factor` (7) deeply-nested plans in one
    # completion — the global max_tokens above is too tight for that and was
    # truncating the JSON mid-string. Override just for that call.
    propose_max_tokens: int = 8192
    # Token pricing for the live cost counter (USD per 1M tokens). Mistral Medium
    # defaults; override via LLM__INPUT_PRICE_PER_1M / LLM__OUTPUT_PRICE_PER_1M.
    input_price_per_1m: float = 1.275
    output_price_per_1m: float = 6.375
    # Per-call HTTP read timeout for the Mistral API (httpx default is 60 s, which is
    # too short for PlanProposer generating 7 plans at high temperature + 4096 tokens).
    llm_timeout_s: int = 300
    # Process-wide cap on concurrent CrewAI kickoff() calls against this API key.
    # Per-domain semaphores (synthesis.max_parallel_subdomains) only bound
    # concurrency within one domain; with 7 domains running in parallel that can
    # still mean dozens of simultaneous requests without this global cap.
    max_concurrent_calls: int = 7


class ToTSettings(BaseSettings):
    branching_factor: int = 7
    depth: int = 2
    survivors: int = 3
    min_diversity_dims: int = 3
    feasibility_weight: float = 0.60
    quality_weight: float = 0.40
    diversity_penalty: float = 0.25
    diversity_overlap_threshold: float = 0.80


class ReActSettings(BaseSettings):
    max_iterations: int = 15
    collect_max_retries: int = 3  # silent collect→backtrack retries before Gate 2 is shown
    confidence_threshold: float = 0.75
    hard_time_limit_s: int = 3600   # 60-min safety net; soft timeout handles UX
    soft_timeout_s: int = 900       # 15-min soft timeout → prompts user to continue, repeats every soft_timeout_s
    per_tool_latency_budget_s: int = 10
    run_token_budget: int = 200_000
    # 1 initial attempt + 4 LLM-adapted repair retries = 5 total attempts per tool call
    tool_repair_max_attempts: int = 4
    max_parallel_tool_calls: int = 5  # concurrency cap for tool-execution loops
    # Ceiling on a single CrewAI synthesis call (crew.kickoff -> litellm -> Mistral).
    # Without this, a stalled completion call blocks indefinitely; since it's run via
    # asyncio.to_thread, an unbounded hang here would otherwise tie up a worker thread
    # forever instead of falling back to plain-text formatting.
    crew_llm_timeout_s: int = 90
    # Tool circuit breaker: a tool is blocked for the rest of the run after this many
    # logical-call failures, or immediately on the first rate-limit (HTTP 429) — a
    # throttled API key won't recover mid-run, so retrying it just wastes minutes.
    tool_failure_threshold: int = 5
    # Stall watchdog: how often to check for inactivity, and how long with zero
    # tool/LLM/progress activity counts as a stall. The idle threshold sits above the
    # longest legitimately-silent single call (Tavily /research ~180 s) to avoid false
    # alarms during a slow-but-healthy call.
    stall_check_interval_s: int = 60
    stall_idle_threshold_s: int = 210


class RetrievalSettings(BaseSettings):
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 5
    reranker_enabled: bool = True
    reranker_model: str = ""
    embedding_model: str = ""
    staleness_window_days: dict[str, int] = Field(default_factory=dict)
    corpus_refresh: str = "quarterly"
    named_entity_confidence_discount: float = 0.7


class StoreSettings(BaseSettings):
    sqlite_path: str = str(_BACKEND_ROOT / "outputs" / "intel.db")
    # Outside the project dir on purpose: SQLite WAL mode (required by the
    # LangGraph checkpointer) needs mmap'd file locking that iCloud/Drive-synced
    # folders don't support, so a path under ./outputs raises "disk I/O error".
    checkpoint_path: str = "~/.komatsu_capstone/checkpoints.db"
    chroma_path: str = str(_BACKEND_ROOT / "outputs" / "chroma")
    wipe_session_stores_on_chat: bool = True
    episodic_enabled: bool = False
    episodic_min_quality_score: float = 0.75
    chroma_collected_prefix: str = "collected"      # collection = f"{prefix}_{run_id}"
    chroma_episodic_collection: str = "episodic_memory"
    chroma_knowledge_collection: str = "industry_knowledge"
    vector_backend: Literal["chroma", "azure_search"] = "chroma"


class DomainConfig(BaseSettings):
    enabled: bool = True
    weight: float = 1.0


class GateSettings(BaseSettings):
    gate_1_enabled: bool = True
    gate_2_enabled: bool = True
    gate_3_enabled: bool = True
    auto_approve_gates: bool = False
    gate_timeout_s: int = 600
    gate_clarification_enabled: bool = True


class ReportSettings(BaseSettings):
    exec_summary_min_words: int = 400
    exec_summary_max_words: int = 500
    report_format: str = "pdf"
    output_dir: str = str(_BACKEND_ROOT / "outputs")
    # Numeric table datasets with more rows than this render as a chart instead.
    report_table_max_rows: int = 10


class SynthesisSettings(BaseSettings):
    """Controls the 3-tier hierarchical synthesis.

    Tier 1 = per-entity (subdomain) analysis, Tier 2 = domain rollup,
    Tier 3 = executive summary (built from the rollups). When disabled the
    pipeline reverts to the legacy 2-tier flow (one chapter per domain).
    """
    hierarchical_enabled: bool = True
    # Domains decomposed into per-entity subchapters. Defaults come from the domain
    # registry (core/domains.py); general_search has no fixed master-data entity
    # list, so its "entities" are LLM-derived themes instead (see core/subdomains.py).
    decomposable_domains: set[str] = Field(
        default_factory=lambda: set(_registry_decomposable_domains())
    )
    max_subdomains_per_domain: int = 25
    max_parallel_subdomains: int = 7  # concurrent Tier-1 LLM calls within a domain
    max_parallel_chapters: int = 4  # concurrent Tier-2 domain-chapter syntheses
    subdomain_min_words: int = 150
    subdomain_max_words: int = 250
    rollup_min_words: int = 250
    rollup_max_words: int = 400
    # Thematic decomposition (general_search): skip the LLM
    # theme-extraction call entirely when the chapter has less evidence than this.
    theme_extraction_min_evidence_words: int = 300
    theme_extraction_max_evidence_chars: int = 6000
    theme_extraction_temperature: float = 0.0
    # Pre-finalization completeness gate: scan synthesized chapters/subchapters
    # for fallback placeholders (e.g. "Limited evidence collected for X") or
    # empty evidence, and attempt one bounded remediation pass before the
    # report is assembled. See core/completeness.py.
    completeness_gate_enabled: bool = True
    completeness_max_remediation_rounds: int = 1


class SafetySettings(BaseSettings):
    tool_allowlist: list[str] = Field(default_factory=list)
    max_spend_per_run_usd: float = 5.00
    max_api_calls_per_run: int = 100
    allow_network_writes: bool = False


class UnderstandSettings(BaseSettings):
    """Controls the pre-planning research loop and plan-merger step."""
    research_enabled: bool = True
    research_max_tool_calls: int = 10
    research_timeout_s: int = 300
    # If the consolidated plan still has gap_report entries after merging,
    # retry grounding up to this many extra rounds before presenting the
    # plan at Gate 1. Bounded by round count (not time) since most gaps are
    # structural (no substitute tool exists) and won't close with more
    # attempts; the existing soft timeout still governs overall duration.
    gap_remediation_max_rounds: int = 3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    tot: ToTSettings = Field(default_factory=ToTSettings)
    react: ReActSettings = Field(default_factory=ReActSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    stores: StoreSettings = Field(default_factory=StoreSettings)
    gates: GateSettings = Field(default_factory=GateSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    synthesis: SynthesisSettings = Field(default_factory=SynthesisSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    understand: UnderstandSettings = Field(default_factory=UnderstandSettings)
    max_parallel_subagents: int = 5
    cors_allowed_origins: list[str] = Field(
        default=["http://localhost:5000", "http://localhost:5001"]
    )

    # API tier flags — set to "premium" in .env to unlock premium-gated tools
    fmp_tier: Literal["free", "premium"] = "free"
    alpha_vantage_tier: Literal["free", "premium"] = "free"
    # NewsAPI.org free dev plan only returns articles from the last month;
    # paid plans unlock the full historical archive (~5 years).
    newsapi_tier: Literal["free", "premium"] = "free"
    newsapi_free_max_lookback_days: int = 30
    newsapi_premium_max_lookback_days: int = 1825

    # Secrets (env-only)
    mistral_api_key: str = ""
    alpha_vantage_api_key: str = ""
    fmp_api_key: str = ""
    newsapi_api_key: str = ""
    sec_edgar_api_key: str = ""

    # NewsAPI client config
    newsapi_base_url: str = "https://newsapi.org"
    newsapi_timeout_s: int = 10
    newsapi_max_retries: int = 3
    newsapi_rate_limit_per_min: int = 100

    # Alpha Vantage client config
    alpha_vantage_base_url: str = "https://www.alphavantage.co"
    alpha_vantage_timeout_s: int = 10
    alpha_vantage_max_retries: int = 3
    alpha_vantage_rate_limit_per_min: int = 5  # free-tier cap

    # FMP (Financial Modeling Prep) client config — /stable/ API (the /api/v3/* routes
    # are legacy and return HTTP 403 for keys created after 2025-08-31).
    fmp_base_url: str = "https://financialmodelingprep.com"
    fmp_timeout_s: int = 15
    fmp_max_retries: int = 1  # fail fast on rate limits so we fall back to yfinance
    fmp_rate_limit_per_min: int = 10
    fmp_backoff_cap_s: float = 2.0  # cap retry backoff so a 429 doesn't stall the request

    # EDGAR client config (sec_edgar_api_key = contact email for User-Agent)
    edgar_base_url: str = "https://efts.sec.gov"
    edgar_timeout_s: int = 15
    edgar_max_retries: int = 2
    edgar_rate_limit_per_min: int = 10
    edgar_exhibit_excerpt_max_chars: int = 15000

    # Tavily web search config
    tavily_api_key: str = ""
    tavily_base_url: str = "https://api.tavily.com"
    tavily_timeout_s: int = 15
    tavily_research_timeout_s: int = 180  # /research takes 30–120 s
    tavily_rate_limit_per_min: int = 60

    # FRED (Federal Reserve Economic Data) config — free key at fred.stlouisfed.org
    fred_api_key: str = ""
    fred_base_url: str = "https://api.stlouisfed.org/fred"
    fred_timeout_s: int = 15
    fred_max_retries: int = 3
    fred_rate_limit_per_min: int = 120

    # Master data paths (relative to data/)
    commodity_tickers_path: str = "commodities/commodity_tickers.csv"


settings = Settings()
