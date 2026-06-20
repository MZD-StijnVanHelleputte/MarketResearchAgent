from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    status: str
