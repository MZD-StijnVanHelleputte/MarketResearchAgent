from pydantic import BaseModel


class CandidatePlan(BaseModel):
    plan_id: str
    feasibility_score: float
    quality_score: float
    rationale: str
    gap_report: str


class PlanReview(BaseModel):
    """Deprecated: kept for backward compatibility. Gate 1 now sends ConsolidatedPlan."""
    run_id: str
    plans: list[CandidatePlan]


class PlannedToolCall(BaseModel):
    tool: str
    params: dict
    domain: str
    rationale: str = ""


class ConsolidatedPlan(BaseModel):
    """Single merged plan sent to Gate 1 (replaces the 3-option PlanReview)."""
    plan_id: str
    source_plan_ids: list[str] = []
    domains_active: list[str] = []
    entity_manifest: dict = {}
    planned_tool_calls: list[PlannedToolCall] = []
    research_findings: str = ""
    rationale: str = ""
    gap_report: str = ""
    feasibility_score: float = 0.0
    quality_score: float = 0.0


class ConsolidatedPlanReview(BaseModel):
    """Gate 1 payload: the consolidated plan plus the run identifier."""
    run_id: str
    plan: ConsolidatedPlan


class DomainConfidence(BaseModel):
    domain: str
    confidence: float
    status: str


class ConfidenceSummary(BaseModel):
    run_id: str
    domains: list[DomainConfidence]


class BriefReview(BaseModel):
    run_id: str
    sections: list[dict]
    executive_summary: str


class ClarificationRequest(BaseModel):
    run_id: str
    missing_fields: list[str]


class ClarificationAnswer(BaseModel):
    equipment_models: list[str] = []
    operators: list[str] = []
    competitor_tickers: list[str] = []


class GateDecisionResponse(BaseModel):
    run_id: str
    gate: int
    decision: str
    next_status: str | None = None
