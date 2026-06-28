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


def _planned_calls(plan: dict | None) -> list[dict]:
    if not isinstance(plan, dict):
        return []
    calls = plan.get("planned_tool_calls") or plan.get("tool_calls") or []
    return calls if isinstance(calls, list) else []


def _planned_call_ids(plan: dict | None) -> dict[str, dict]:
    """Return the authoritative logical call IDs for a plan.

    The domain agent enumerates calls after filtering by domain, so the stable ID is
    f"{plan_id}:{domain}:{idx-within-that-domain}".
    """
    plan_id = (plan or {}).get("plan_id", "")
    by_domain: dict[str, list[dict]] = {}
    for call in _planned_calls(plan):
        domain = call.get("domain") or "general_search"
        by_domain.setdefault(domain, []).append(call)

    out: dict[str, dict] = {}
    for domain, calls in by_domain.items():
        for idx, call in enumerate(calls):
            out[f"{plan_id}:{domain}:{idx}"] = call
    return out


def _call_status_map(
    events: list[dict],
    allowed_call_ids: set[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Map each logical tool call_id to its final (status, reason) from step_events.

    status is "success" or "failed"; reason is the error string for failures (""
    otherwise). A "tool_call" event marks a call_id succeeded, "tool_failed_final"
    marks it failed; transient mid-retry "tool_error" events are ignored. The latest
    terminal write for a given call_id wins, so a no-data final failure can override
    an earlier transport-level tool_call event for the same logical call.
    """
    status: dict[str, tuple[str, str]] = {}
    for e in events:
        event_type = e.get("event_type")
        if event_type not in ("tool_call", "tool_failed_final") or e.get("stage") != "collect":
            continue
        try:
            detail = json.loads(e["detail"]) if e.get("detail") else {}
        except (json.JSONDecodeError, TypeError):
            detail = {}
        call_id = detail.get("call_id")
        if not call_id:
            continue
        if allowed_call_ids is not None and call_id not in allowed_call_ids:
            continue
        if event_type == "tool_call":
            status[call_id] = ("success", "")
        else:
            status[call_id] = ("failed", str(detail.get("error") or ""))
    return status


def _call_result_meta(
    events: list[dict],
    allowed_call_ids: set[str] | None = None,
) -> dict[str, tuple[str, int]]:
    """Map each succeeded tool call_id to the coarse (data_type, count) of what it
    returned, read from the "tool_call" event detail (see core/tool_router). Used to
    label each tool-call node in the collection-plan tree at Gate 2. Latest write
    wins, matching _call_status_map."""
    meta: dict[str, tuple[str, int]] = {}
    for e in events:
        if e.get("event_type") != "tool_call" or e.get("stage") != "collect":
            continue
        try:
            detail = json.loads(e["detail"]) if e.get("detail") else {}
        except (json.JSONDecodeError, TypeError):
            detail = {}
        call_id = detail.get("call_id")
        if not call_id:
            continue
        if allowed_call_ids is not None and call_id not in allowed_call_ids:
            continue
        meta[call_id] = (detail.get("data_type") or "data", detail.get("count") or 0)
    return meta


def _compute_collect_progress(
    events: list[dict],
    current_stage: str | None,
    plan: dict | None = None,
) -> dict:
    """Derive live collection-progress fields from the run's step_events, so the
    frontend can show a determinate completion bar instead of a spinner.

    tool_calls_total comes from the approved plan, never from live execution
    events. Completion is tracked per logical call via detail["call_id"] (stable
    across all of a call's retry attempts) and filtered to that plan's call IDs.
    Transient "tool_error" events from mid-retry attempts are not counted, so a
    call that fails twice then succeeds is still only counted once.
    """
    planned = _planned_call_ids(plan)
    allowed_ids = set(planned)
    total = len(planned)
    current_label: str | None = None
    current_domain: str | None = None
    for e in events:
        event_type = e.get("event_type")
        if (
            event_type == "progress"
            and e.get("stage") == "collect"
            and (e.get("label") or "").startswith("Fetching ")
        ):
            current_label = e.get("label")
            current_domain = e.get("domain") or None

    call_status = _call_status_map(events, allowed_ids)
    completed = len(call_status)
    failed = sum(1 for st, _ in call_status.values() if st == "failed")
    succeeded = completed - failed

    if current_stage != "collect" or (total > 0 and completed >= total):
        current_label = None
        current_domain = None

    return {
        "tool_calls_total": total,
        "tool_calls_completed": completed,
        "tool_calls_succeeded": succeeded,
        "tool_calls_failed": failed,
        "current_tool_label": current_label,
        "current_domain": current_domain,
    }


def _leaf_for_call(call: dict, leaves: list[dict]) -> dict | None:
    """Best-effort attribution of a planned tool call to one of its domain's leaves.

    Priority: (1) a ticker/symbol param equal to the leaf's params, then (2) a leaf
    label appearing in any string param (e.g. a search query). Returns None when no
    leaf claims the call (it lands in the domain's "General" bucket)."""
    params = call.get("params") or {}
    call_tickers = {
        str(params[k]).strip().lower()
        for k in ("ticker", "symbol")
        if params.get(k)
    }
    if call_tickers:
        for leaf in leaves:
            lp = leaf.get("params") or {}
            leaf_vals = {
                str(lp[k]).strip().lower() for k in ("ticker", "symbol") if lp.get(k)
            }
            if leaf_vals & call_tickers:
                return leaf
    # Fall back to a leaf label mentioned in any free-text param.
    blob = " ".join(str(v) for v in params.values()).lower()
    if blob:
        for leaf in leaves:
            label = str(leaf.get("label") or "").strip().lower()
            if len(label) >= 3 and label in blob:
                return leaf
    return None


def build_collection_plan(plan: dict | None, events: list[dict]) -> dict | None:
    """Annotate the consolidated plan's tool calls with live execution status, as a
    tree the frontend renders directly: plan → domains → leaves → tool calls.

    Each planned tool call is grouped under the leaf it serves (see _leaf_for_call)
    and labelled pending/succeeded/failed by matching its reconstructed call_id
    (f"{plan_id}:{domain}:{idx}", idx = position within the domain-filtered calls,
    exactly as base_domain_agent enumerates them) against the run's step_events.
    Succeeded calls also carry the coarse data_type/count of what they returned.

    Renders during planning too: a preliminary plan that has leaves but no tool
    calls yet shows the domain→leaf skeleton (every node pending), so the tree
    visibly builds up across Understand before filling in during collection.
    Returns None only when there is nothing (no calls and no leaves) to show.
    """
    if not isinstance(plan, dict):
        return None
    calls = _planned_calls(plan)
    leaves_all = plan.get("leaves") or []
    if not calls and not leaves_all:
        return None

    from core.domains import display_name, ownership_order
    from tools.registry import tool_display_name

    plan_id = plan.get("plan_id", "")
    allowed_ids = set(_planned_call_ids(plan))
    status_map = _call_status_map(events, allowed_ids)
    result_meta = _call_result_meta(events, allowed_ids)

    def _empty() -> dict:
        return {"total": 0, "succeeded": 0, "failed": 0, "pending": 0}

    def _bump(counter: dict, status: str) -> None:
        counter["total"] += 1
        counter[status] += 1

    # Group calls by domain in plan order; the index within a domain's list is the
    # same idx base_domain_agent uses to build call_id.
    by_domain: dict[str, list[dict]] = {}
    for call in calls:
        domain = call.get("domain") or "general_search"
        by_domain.setdefault(domain, []).append(call)

    # Domains come from the planned calls *and* the plan's leaves, so the skeleton is
    # visible during planning (leaves present, no calls yet). Order by ownership
    # priority, with any unknown domains trailing.
    leaf_domains = {lf.get("domain") for lf in leaves_all if lf.get("domain")}
    order = ownership_order()
    domain_keys = sorted(
        set(by_domain) | leaf_domains,
        key=lambda d: order.index(d) if d in order else len(order),
    )

    plan_counts = _empty()
    domain_nodes: list[dict] = []
    for domain in domain_keys:
        leaves_in_domain = [lf for lf in leaves_all if lf.get("domain") == domain]
        # Preserve leaf order from the plan; "General" holds unattributable calls.
        leaf_nodes: dict[str, dict] = {}
        leaf_order: list[str] = []

        def _leaf_node(key: str, label: str, leaf_type: str) -> dict:
            if key not in leaf_nodes:
                leaf_nodes[key] = {
                    "key": key, "label": label, "leaf_type": leaf_type,
                    "tool_calls": [], **_empty(),
                }
                leaf_order.append(key)
            return leaf_nodes[key]

        # Pre-create every planned leaf so it stays visible even before any tool call
        # is attributed to it (and so a leaf shown during planning doesn't vanish once
        # tool calls are assigned in the merged plan).
        for lf in leaves_in_domain:
            _leaf_node(
                lf.get("key") or lf.get("label") or "?",
                lf.get("label") or lf.get("key") or "?",
                lf.get("leaf_type") or "",
            )

        domain_counts = _empty()
        for idx, call in enumerate(by_domain.get(domain, [])):
            call_id = f"{plan_id}:{domain}:{idx}"
            st, reason = status_map.get(call_id, ("pending", ""))
            status = "succeeded" if st == "success" else st  # "success"→"succeeded"
            leaf = _leaf_for_call(call, leaves_in_domain)
            if leaf is not None:
                node = _leaf_node(
                    leaf.get("key") or leaf.get("label") or "?",
                    leaf.get("label") or leaf.get("key") or "?",
                    leaf.get("leaf_type") or "",
                )
            else:
                node = _leaf_node("__general__", "General", "")
            tool = call.get("tool", "")
            tc = {
                "tool": tool,
                "display": tool_display_name(tool),
                "status": status,
                "reason": reason or None,
            }
            if status == "succeeded":
                data_type, count = result_meta.get(call_id, ("data", 0))
                tc["data_type"] = data_type
                tc["count"] = count
            node["tool_calls"].append(tc)
            _bump(node, status)
            _bump(domain_counts, status)
            _bump(plan_counts, status)

        domain_nodes.append({
            "domain": domain,
            "display": display_name(domain),
            "leaves": [leaf_nodes[k] for k in leaf_order],
            **domain_counts,
        })

    return {**plan_counts, "domains": domain_nodes}


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
        # Live stem→leaf collection plan with per-tool-call status, for the frontend
        # tree that replaces the flat sources panel during collection / at Gate 2.
        plan = (result.get("plans") or [None])[0]
        result.update(_compute_collect_progress(events, result.get("stage"), plan))
        result["collection_plan"] = build_collection_plan(plan, events)
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

    async def get_step_events(
        self, run_id: str, limit: int | None = 500, order: str = "asc"
    ) -> list[dict]:
        """order='asc' (default) is chronological, required by callers that derive
        progress/collection-plan state from event sequence. order='desc' is for
        display (e.g. the Testing page log viewer), which wants newest-first and
        no limit so no events are silently dropped."""
        direction = "DESC" if order == "desc" else "ASC"
        sql = f"SELECT id,ts,level,stage,domain,event_type,label,detail FROM step_events WHERE run_id=? ORDER BY id {direction}"
        params: tuple = (run_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (run_id, limit)
        async with aiosqlite.connect(self._path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_last_activity_ts(self, run_id: str) -> float | None:
        """Epoch seconds of the most recent step_event for a run, or None if the run
        has logged nothing yet. step_events are written on every tool call/error and
        progress milestone, so this is the run's 'last sign of life' — used by the
        stall watchdog. ts is stored as an ISO-8601 UTC string, which sorts
        chronologically, so MAX(ts) gives the latest."""
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT MAX(ts) FROM step_events WHERE run_id=?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if not row or not row[0]:
            return None
        try:
            return datetime.fromisoformat(row[0]).timestamp()
        except ValueError:
            return None

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
