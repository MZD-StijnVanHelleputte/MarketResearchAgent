from fastapi import APIRouter

from memory.sqlite_store import SqliteStore

router = APIRouter(prefix="/preferences", tags=["preferences"])


@router.get("")
async def get_preferences() -> dict:
    return {"preferences": await SqliteStore().get_all_preferences()}


@router.put("")
async def update_preferences(body: dict) -> dict:
    store = SqliteStore()
    for key, value in body.items():
        await store.set_preference(key, value)
    return {"preferences": await store.get_all_preferences()}
