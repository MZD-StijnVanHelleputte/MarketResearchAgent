from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings
from memory.sqlite_store import SqliteStore

router = APIRouter(tags=["reports"])


@router.get("/runs/{run_id}/report")
async def download_report(run_id: str) -> FileResponse:
    store = SqliteStore()
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") not in ("done",):
        raise HTTPException(
            status_code=409,
            detail=f"Report not ready: run status is '{run.get('status')}'",
        )
    pdf_path = Path(settings.report.output_dir) / f"{run_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not yet generated for this run")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"komatsu_report_{run_id[:8]}.pdf",
    )
