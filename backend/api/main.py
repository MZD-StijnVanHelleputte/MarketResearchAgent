import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, gates, health, knowledge, logs, metrics, preferences, reports, sessions, tests
from config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from retrieval import Retriever
    from services.masterdata_service import MasterDataService
    from state_bus.server import mcp_server  # noqa: F401 — registers the MCP server instance
    from core.graph import init_checkpointer

    await init_checkpointer()
    logger.info("LangGraph checkpointer backed by SQLite (durable across restarts).")

    svc = MasterDataService()
    retriever = Retriever()
    try:
        count = await asyncio.to_thread(svc.load_industry_knowledge, retriever)
        if count:
            logger.info("Seeded industry knowledge: %d chunks written.", count)
    except Exception as exc:
        logger.warning("Industry knowledge seeding failed (non-fatal): %s", exc)

    logger.info("FastMCP planning state bus ready.")
    yield


app = FastAPI(title="Komatsu Market Intelligence API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(gates.router)
app.include_router(knowledge.router)
app.include_router(logs.router)
app.include_router(metrics.router)
app.include_router(preferences.router)
app.include_router(reports.router)
app.include_router(sessions.router)
app.include_router(tests.router)
