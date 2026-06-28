from fastapi import APIRouter, HTTPException
from memory.sqlite_store import SqliteStore

router = APIRouter(tags=["logs"])


@router.get("/runs/{run_id}/logs")
async def get_run_logs(run_id: str) -> list[dict]:
    store = SqliteStore()
    row = await store.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return await store.get_step_events(run_id, limit=None, order="desc")
