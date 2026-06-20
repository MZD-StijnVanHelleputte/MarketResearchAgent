"""Dashboard metrics: aggregate token/cost/run statistics from the SQLite run log."""
from fastapi import APIRouter
from pydantic import BaseModel

from memory.sqlite_store import SqliteStore

router = APIRouter(tags=["metrics"])


class RunMetric(BaseModel):
    run_id: str
    query: str
    started_at: str
    duration_seconds: int
    total_tokens: int
    cost_usd: float
    tool_calls: int
    status: str


class DashboardSummary(BaseModel):
    total_runs: int
    total_tokens: int
    total_cost_usd: float
    avg_duration_seconds: float
    recent_runs: list[RunMetric]


@router.get("/metrics", response_model=DashboardSummary)
async def get_metrics(recent: int = 20) -> DashboardSummary:
    runs = await SqliteStore().list_runs(limit=200)

    total_tokens = sum(r["total_tokens"] for r in runs)
    total_cost = sum(r["cumulative_cost_usd"] for r in runs)
    durations = [r["duration_seconds"] for r in runs]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    recent_runs = [
        RunMetric(
            run_id=r["run_id"],
            query=r.get("query") or "",
            started_at=r.get("created_at") or "",
            duration_seconds=r["duration_seconds"],
            total_tokens=r["total_tokens"],
            cost_usd=round(r["cumulative_cost_usd"], 6),
            tool_calls=r["api_call_count"],
            status=r.get("status") or "",
        )
        for r in runs[:recent]
    ]

    return DashboardSummary(
        total_runs=len(runs),
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 6),
        avg_duration_seconds=round(avg_duration, 1),
        recent_runs=recent_runs,
    )
