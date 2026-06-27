import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_figures (
    run_id       TEXT NOT NULL,
    domain       TEXT NOT NULL,
    key          TEXT NOT NULL,
    value        TEXT NOT NULL,
    collected_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS preferences (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query      TEXT,
    status     TEXT NOT NULL,
    stage      TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    brief      TEXT,
    sources    TEXT,
    error      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS step_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    level      TEXT    NOT NULL DEFAULT 'info',
    stage      TEXT,
    domain     TEXT,
    event_type TEXT    NOT NULL,
    label      TEXT    NOT NULL,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_step_events_run ON step_events(run_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Must match _NODE_MESSAGES["collect"] in api/routers/chat.py — logged once per
# collect_node entry (including ReAct backtrack re-entries), marking where the
# *current* collection pass begins.
_COLLECT_ITERATION_START_LABEL = "Collecting intelligence from data sources…"


def _compute_collect_progress(events: list[dict], current_stage: str | None) -> dict:
    """Derive live collection-progress fields from the run's step_events, so the
    frontend can show a determinate completion bar instead of a spinner.

    tool_calls_total comes from each domain agent's one-time "domain_filter" event
    (detail["matched"] = calls assigned to that domain); tool_calls_completed counts
    "tool_call"/"tool_error" events. A retried call logs an extra "tool_error" before
    its eventual "tool_call", so completed can briefly exceed total — an accepted,
    cosmetic overcount, not worth suppressing.

    ReAct backtracking can re-invoke collect_node within the same run, re-logging
    domain_filter/tool_call/tool_error from scratch each time — without scoping to
    the latest collect_node entry, sums would accumulate across iterations and
    produce a meaningless (and sometimes completed > total) ratio.
    """
    boundary = 0
    for e in events:
        if (
            e.get("event_type") == "progress"
            and e.get("stage") == "collect"
            and e.get("label") == _COLLECT_ITERATION_START_LABEL
        ):
            boundary = e.get("id", boundary)
    events = [e for e in events if e.get("id", 0) >= boundary]

    total = 0
    completed = 0
    current_label: str | None = None
    current_domain: str | None = None
    for e in events:
        event_type = e.get("event_type")
        if event_type == "domain_filter":
            try:
                detail = json.loads(e["detail"]) if e.get("detail") else {}
            except (json.JSONDecodeError, TypeError):
                detail = {}
            total += detail.get("matched") or 0
        elif event_type in ("tool_call", "tool_error") and e.get("stage") == "collect":
            completed += 1
        if (
            event_type == "progress"
            and e.get("stage") == "collect"
            and (e.get("label") or "").startswith("Fetching ")
        ):
            current_label = e.get("label")
            current_domain = e.get("domain") or None

    if current_stage != "collect" or (total > 0 and completed >= total):
        current_label = None
        current_domain = None

    return {
        "tool_calls_total": total,
        "tool_calls_completed": completed,
        "current_tool_label": current_label,
        "current_domain": current_domain,
    }


def _duration_seconds(created_at: str | None, updated_at: str | None) -> int:
    """Whole-second elapsed time between two ISO timestamps (0 on parse failure)."""
    try:
        start = datetime.fromisoformat(created_at)
        end = datetime.fromisoformat(updated_at)
        return max(0, int((end - start).total_seconds()))
    except (TypeError, ValueError):
        return 0


class SqliteStore:
    """Session structured store (wiped per chat) and persistent user preferences.
    This is the only module that touches SQLite."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path or settings.stores.sqlite_path
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with sqlite3.connect(self._path) as conn:
            conn.executescript(_SCHEMA)
            # Idempotent migrations
            try:
                conn.execute("ALTER TABLE runs ADD COLUMN initial_state TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE runs ADD COLUMN gate_data TEXT")
            except sqlite3.OperationalError:
                pass
            for col_def in (
                "exec_summary TEXT",
                "warnings TEXT",
                "injection_flags TEXT",
                "merge_log TEXT",
                "cumulative_cost_usd REAL",
                "api_call_count INTEGER",
                "total_tokens INTEGER",
                "plans TEXT",
                "status_message TEXT",
                "activity_log TEXT",
                "paused_at REAL",
            ):
                try:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    # ── Runs ──────────────────────────────────────────────────────────────────

    async def upsert_run(
        self,
        run_id: str,
        session_id: str,
        query: str | None,
        status: str,
        stage: str,
        confidence: float = 0.0,
        brief: str | None = None,
        sources: list | None = None,
        error: str | None = None,
        initial_state: dict | None = None,
        gate_data: dict | None = None,
        exec_summary: str | None = None,
        warnings: list | None = None,
        injection_flags: list | None = None,
        merge_log: list | None = None,
        cumulative_cost_usd: float | None = None,
        api_call_count: int | None = None,
        total_tokens: int | None = None,
        plans: list | None = None,
        status_message: str | None = None,
        activity_log: list[str] | None = None,
        paused_at: float | None = None,
    ) -> None:
        now = _now()
        sources_json       = json.dumps(sources) if sources is not None else None
        initial_state_json = json.dumps(initial_state) if initial_state is not None else None
        gate_data_json     = json.dumps(gate_data) if gate_data is not None else None
        warnings_json      = json.dumps(warnings) if warnings is not None else None
        inj_flags_json     = json.dumps(injection_flags) if injection_flags is not None else None
        merge_log_json     = json.dumps(merge_log) if merge_log is not None else None
        plans_json         = json.dumps(plans) if plans is not None else None
        activity_log_json  = json.dumps(activity_log) if activity_log is not None else None
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                INSERT INTO runs (run_id, session_id, query, status, stage, confidence,
                                  brief, sources, error, initial_state, gate_data,
                                  exec_summary, warnings, injection_flags, merge_log,
                                  cumulative_cost_usd, api_call_count, total_tokens, plans,
                                  status_message, activity_log, paused_at,
                                  created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status              = excluded.status,
                    stage               = excluded.stage,
                    confidence          = excluded.confidence,
                    brief               = excluded.brief,
                    sources             = excluded.sources,
                    error               = excluded.error,
                    initial_state       = COALESCE(runs.initial_state, excluded.initial_state),
                    gate_data           = excluded.gate_data,
                    exec_summary        = excluded.exec_summary,
                    warnings            = excluded.warnings,
                    injection_flags     = excluded.injection_flags,
                    merge_log           = excluded.merge_log,
                    cumulative_cost_usd = excluded.cumulative_cost_usd,
                    api_call_count      = excluded.api_call_count,
                    total_tokens        = excluded.total_tokens,
                    plans               = COALESCE(excluded.plans, runs.plans),
                    status_message      = COALESCE(excluded.status_message, runs.status_message),
                    activity_log        = COALESCE(excluded.activity_log, runs.activity_log),
                    paused_at           = COALESCE(excluded.paused_at, runs.paused_at),
                    updated_at          = excluded.updated_at
                """,
                (run_id, session_id, query, status, stage, confidence,
                 brief, sources_json, error, initial_state_json, gate_data_json,
                 exec_summary, warnings_json, inj_flags_json, merge_log_json,
                 cumulative_cost_usd, api_call_count, total_tokens, plans_json,
                 status_message, activity_log_json, paused_at, now, now),
            )
            await db.commit()

    async def get_run(self, run_id: str) -> dict | None:
        async with aiosqlite.connect(self._path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        if result.get("sources"):
            result["sources"] = json.loads(result["sources"])
        else:
            result["sources"] = []
        if result.get("initial_state"):
            result["initial_state"] = json.loads(result["initial_state"])
        else:
            result["initial_state"] = None
        if result.get("gate_data"):
            result["gate_data"] = json.loads(result["gate_data"])
        else:
            result["gate_data"] = None
        result["exec_summary"] = result.get("exec_summary") or ""
        result["status_message"] = result.get("status_message") or ""
        for field in ("warnings", "injection_flags", "merge_log", "plans"):
            raw = result.get(field)
            result[field] = json.loads(raw) if raw else []
        # The live activity feed shown in chat is built from step_events
        # (granular, business-friendly progress messages), not the legacy
        # activity_log column (which only ever held 5 static stage strings).
        # limit=1000: a single collect stage with ~50 tool calls across 7 domains can
        # log 200+ events (domain_filter + per-call progress + tool_call/error +
        # retries) on its own, before counting understand/synthesize stage events —
        # truncating here would silently undercount tool_calls_total/completed below.
        events = await self.get_step_events(run_id, limit=1000)
        result["activity_log"] = [e["label"] for e in events if e["event_type"] == "progress"]
        result.update(_compute_collect_progress(events, result.get("stage")))
        return result

    async def list_runs(self, limit: int = 100) -> list[dict]:
        """Return recent runs with their live metrics, newest first.

        Used by the dashboard metrics endpoint; duration is created_at→updated_at.
        """
        async with aiosqlite.connect(self._path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT run_id, query, status, created_at, updated_at, "
                "       cumulative_cost_usd, api_call_count, total_tokens "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
        out: list[dict] = []
        for r in rows:
            row = dict(r)
            row["cumulative_cost_usd"] = row.get("cumulative_cost_usd") or 0.0
            row["api_call_count"] = row.get("api_call_count") or 0
            row["total_tokens"] = row.get("total_tokens") or 0
            row["duration_seconds"] = _duration_seconds(row.get("created_at"), row.get("updated_at"))
            out.append(row)
        return out

    # ── Sessions ──────────────────────────────────────────────────────────────

    async def list_sessions(self) -> list[dict]:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT session_id, MIN(created_at) AS created_at, COUNT(*) AS run_count "
                "FROM runs GROUP BY session_id ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {"session_id": r[0], "created_at": r[1], "run_count": r[2]}
            for r in rows
        ]

    async def get_session(self, session_id: str) -> dict | None:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT session_id, MIN(created_at) AS created_at, COUNT(*) AS run_count "
                "FROM runs WHERE session_id = ? GROUP BY session_id",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return {"session_id": row[0], "created_at": row[1], "run_count": row[2]}

    # ── Step events ───────────────────────────────────────────────────────────

    async def log_step_event(
        self,
        run_id: str,
        ts: str,
        level: str,
        stage: str,
        domain: str,
        event_type: str,
        label: str,
        detail: Any,
    ) -> None:
        detail_json = json.dumps(detail) if detail is not None else None
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO step_events (run_id,ts,level,stage,domain,event_type,label,detail) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (run_id, ts, level, stage or "", domain or "", event_type, label, detail_json),
            )
            await db.commit()

    async def get_step_events(self, run_id: str, limit: int = 500) -> list[dict]:
        async with aiosqlite.connect(self._path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id,ts,level,stage,domain,event_type,label,detail "
                "FROM step_events WHERE run_id=? ORDER BY id LIMIT ?",
                (run_id, limit),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def wipe_session(self, session_id: str) -> None:
        """Delete session_figures for all runs belonging to the session."""
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "DELETE FROM session_figures WHERE run_id IN "
                "(SELECT run_id FROM runs WHERE session_id = ?)",
                (session_id,),
            )
            await db.commit()

    # ── Figures ───────────────────────────────────────────────────────────────

    async def write_figure(
        self, run_id: str, domain: str, key: str, value: Any
    ) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO session_figures (run_id, domain, key, value, collected_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, domain, key, json.dumps(value), _now()),
            )
            await db.commit()

    async def query_figures(self, run_id: str) -> list[dict]:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT domain, key, value, collected_at FROM session_figures "
                "WHERE run_id = ? ORDER BY collected_at",
                (run_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {
                "domain": r[0],
                "key": r[1],
                "value": json.loads(r[2]),
                "collected_at": r[3],
            }
            for r in rows
        ]

    # ── Preferences ───────────────────────────────────────────────────────────

    async def set_preference(self, key: str, value: Any) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, json.dumps(value), _now()),
            )
            await db.commit()

    async def get_preference(self, key: str) -> Any | None:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def get_all_preferences(self) -> dict:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT key, value FROM preferences") as cursor:
                rows = await cursor.fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}
