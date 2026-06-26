from pydantic import BaseModel, Field, field_validator


class CandidatePlan(BaseModel):
    plan_id: str
    domain_activations: dict[str, bool]
    entity_choices: dict[str, str] = Field(default_factory=dict)
    api_assignments: dict[str, str] = Field(default_factory=dict)

    @field_validator("entity_choices", "api_assignments", mode="before")
    @classmethod
    def _coerce_str_dict(cls, v: object) -> object:
        """LLMs sometimes return list values; join them into a single string."""
        if not isinstance(v, dict):
            return v
        return {
            k: (", ".join(str(i) for i in val) if isinstance(val, list) else str(val))
            for k, val in v.items()
        }
    tool_calls: list[dict] = Field(default_factory=list)
    estimated_token_cost: int = 0
    rationale: str = ""
    depth: int = 1
    feasibility_score: float = 0.0
    quality_score: float = 0.0
    combined_score: float = 0.0
    gap_report: list[str] = Field(default_factory=list)
    receives_diversity_penalty: bool = False
    is_survivor: bool = False


class ResearchContext(BaseModel):
    """Entities and signals discovered before plan proposal (Research loop output)."""
    competitors: list[str] = Field(default_factory=list)  # equipment-maker rivals, e.g. "Caterpillar Inc. (CAT)"
    operators: list[str] = Field(default_factory=list)    # mining companies / Komatsu customers, e.g. "BHP Group (BHP)"
    demand_side_companies: list[str] = Field(default_factory=list)  # commodity consumers driving demand, e.g. "BYD (1211.HK)", "Volkswagen (VOW3.DE)"
    tickers: list[str] = Field(default_factory=list)      # "CAT", "VOLV-B.ST"
    commodities: list[str] = Field(default_factory=list)  # "Gold (GC=F)"
    mine_sites: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    news_signals: list[str] = Field(default_factory=list) # top recent headlines
    open_questions: list[str] = Field(default_factory=list)
    tool_calls_used: int = 0


class PlannedToolCall(BaseModel):
    """A concrete, parameter-resolved tool call that will be made during Collect."""
    tool: str
    params: dict = Field(default_factory=dict)
    domain: str
    rationale: str = ""


class ConsolidatedPlan(BaseModel):
    """Single merged plan produced from TOT_SURVIVORS, shown at Gate 1."""
    plan_id: str
    source_plan_ids: list[str] = Field(default_factory=list)
    domains_active: list[str] = Field(default_factory=list)
    entity_manifest: dict = Field(default_factory=dict)
    planned_tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    research_findings: str = ""
    rationale: str = ""
    gap_report: str = ""
    feasibility_score: float = 0.0
    quality_score: float = 0.0
