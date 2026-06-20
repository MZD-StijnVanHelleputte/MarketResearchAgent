from fastapi import APIRouter, HTTPException

from api.schemas.session import Session, SessionList
from memory.sqlite_store import SqliteStore

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=SessionList)
async def list_sessions() -> SessionList:
    rows = await SqliteStore().list_sessions()
    sessions = [
        Session(session_id=r["session_id"], created_at=r["created_at"], run_count=r["run_count"])
        for r in rows
    ]
    return SessionList(sessions=sessions, total=len(sessions))


@router.get("/{session_id}", response_model=Session)
async def get_session(session_id: str) -> Session:
    row = await SqliteStore().get_session(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return Session(
        session_id=row["session_id"],
        created_at=row["created_at"],
        run_count=row["run_count"],
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    await SqliteStore().wipe_session(session_id)
    return {"deleted": session_id}
