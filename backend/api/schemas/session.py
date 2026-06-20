from pydantic import BaseModel


class Session(BaseModel):
    session_id: str
    created_at: str
    run_count: int


class SessionList(BaseModel):
    sessions: list[Session]
    total: int
