# AGENTS.md

This file describes the structure, conventions, and build plan of the **Komatsu Market
Intelligence Agent** for any agent (human or AI) working on it. Read it before writing or
modifying any code.

The system is an agentic market-intelligence assistant: it monitors Komatsu's competitive
landscape across the globe, synthesising signals from **seven intelligence domains**
into an on-demand brief or recurring newsletter for a sales manager or strategy lead.

It is built around one reasoning spine — **Understand → Collect → Synthesize** — with a
**Tree-of-Thought planner** inside Understand, a **ReAct-style backtracking** controller over
all three stages, **RAG retrieval** at three distinct points, **parallel multi-agent
collection**, and **human-review gates** between stages.

---

## 1. Architecture at a glance

The system is decomposed into a **light frontend** (interaction only) and a **backend** that
does all the heavy lifting. They are separate processes that communicate **only over HTTP**.

```
komatsu-intel/
├── frontend/                 # light client — interaction only, no business logic
│   └── KomatsuIntel.Frontend/   # ASP.NET Core Blazor WebAssembly app
│       ├── Program.cs           # app entry point, DI, HttpClient base address
│       ├── Services/
│       │   └── ApiClient.cs     # typed HttpClient wrapper over the backend HTTP API
│       ├── Components/          # shared Razor components (gate widgets, charts, nav)
│       ├── Pages/
│       │   ├── FrontPage.razor  # newspaper-style daily brief; resets 00:00 each morning
│       │   ├── Agent.razor      # conversational interface — ask custom questions / request reports
│       │   ├── Archive.razor    # historical reports + visual episodic-memory browser
│       │   ├── Dashboard.razor  # usage monitoring — tool calls, token consumption, run history
│       │   └── Testing.razor    # test-case runner to validate agent behaviour
│       └── KomatsuIntel.Frontend.csproj
│
├── backend/
│   ├── core/                 # orchestration brain: LangGraph graph, ToT planner, ReAct loop
│   ├── agents/               # CrewAI specialised sub-agents (domain, critic, synthesis)
│   ├── tools/                # thin tool definitions (LLM function-calling interface)
│   ├── services/             # business logic
│   ├── clients/              # raw external API communication
│   ├── retrieval/            # RAG: chunking, embeddings, ChromaDB, reranking
│   ├── memory/               # short-term context + SQLite structured/session store
│   ├── mcp/                  # FastMCP server: shared planning state + episodic memory
│   ├── reports/              # report assembly + PDF generation
│   ├── api/                  # FastAPI interface layer
│   ├── prompts/              # prompt templates + tool JSON schemas
│   ├── models/               # LLM client + response parser
│   ├── config/               # settings.py (the single tuning surface) + logging
│   ├── data/                 # version-controlled master data
│   ├── tests/                # unit/ + integration/
│   └── main.py               # backend entry point
│
├── .env.example              # all required environment variables
└── README.md                 # deployment / infrastructure
```

### Framework mapping

| Concern | Framework | Where |
| --- | --- | --- |
| Stateful orchestration, stage gates, backtracking | **LangGraph** | `core/graph.py` |
| Thought generation (depth-1 plan proposals) | **LangChain (LCEL)** | `core/tot/` |
| Specialised sub-agents (domain, critic, synthesis) | **CrewAI** | `agents/` |
| Shared planning state + episodic memory bus | **FastMCP** | `mcp/` |
| HTTP interface | **FastAPI** | `api/` |
| Vector store / semantic retrieval | **ChromaDB** (Azure AI Search optional) | `retrieval/` |
| Structured + session store | **SQLite** | `memory/` |
| Schemas, settings, validation | **Pydantic** | everywhere |
| Light interaction UI | **Blazor WebAssembly** (ASP.NET Core) | `frontend/` |

---

## 2. Reasoning architecture (the spine)

Everything in `core/` exists to run one loop. It maps directly onto the capstone success
metrics.

```
        ┌──────────────────────────────────────────────────────────────┐
        │  UNDERSTAND  (Tree of Thought, depth 2)                        │
        │   1. ideal-plan propose  → 7 unconstrained plans (high temp)   │
        │   2. reality grounding   → rewrite each against real tools     │
        │   3. score + rank        → feasibility 60% / quality 40%       │
        │   4. select top 3 survivors                                    │
        │   ── GATE 1: human approves plans ──                           │
        └──────────────────────────────────────────────────────────────┘
                              │  (3 plans dispatched in parallel)
        ┌──────────────────────────────────────────────────────────────┐
        │  COLLECT  (parallel domain sub-agents)                         │
        │   per tool call: run → verify shape → store                    │
        │   structured → SQLite     unstructured → ChromaDB              │
        │   update plan after every call                                 │
        │   ── GATE 2: confidence summary; redirect / re-run ──          │
        └──────────────────────────────────────────────────────────────┘
                              │
        ┌──────────────────────────────────────────────────────────────┐
        │  SYNTHESIZE  (per-chapter)                                     │
        │   retrieve knowledge (ChromaDB) + figures (SQLite)             │
        │   cross-domain correlation checks                              │
        │   400–500 word executive summary placed first                 │
        │   merge the 3 parallel chapter sets → assembled PDF            │
        │   ── GATE 3: human reviews brief; flag sections ──             │
        └──────────────────────────────────────────────────────────────┘
```

**Backtracking (ReAct controller).** The LangGraph top-level loop monitors all three stages.
If Synthesize finds a signal under-evidenced, it re-enters Understand (reframe) or drops back
to Collect (more data). Stopping condition: convergence above `CONFIDENCE_THRESHOLD` **or**
`HARD_TIME_LIMIT_S`, after which a partial brief is delivered with flagged gaps. The preserved
depth-1 plans allow re-entry without restarting from scratch.

**Tree of Thought (inside Understand).**
- *Depth 1 — unconstrained ideal plans.* One high-temperature LCEL propose call generates
  `TOT_BRANCHING_FACTOR` (default 7) complete candidate plans, each a structured JSON object
  (plan id, domain-activation decisions, entity-resolution choices, API assignments, estimated
  token cost, one-paragraph rationale). The generator must vary plans on ≥ `TOT_MIN_DIVERSITY_DIMS`
  dimensions (domain scope, geographic focus, time horizon).
- *Depth 2 — reality grounding.* The CrewAI critic rewrites each plan against tools that
  actually exist (the API clients, ChromaDB, SQLite, the gate constraints), substituting or
  dropping anything unavailable and appending a gap report.
- *Evaluate & prune.* All plans scored on **feasibility** (`FEASIBILITY_WEIGHT`, default 0.60)
  and **quality** (`QUALITY_WEIGHT`, default 0.40). Top `TOT_SURVIVORS` (default 3) retained;
  the rest pruned **but preserved in MCP state** for the recovery path (see §8).

The ToT mechanism *replaces* the single-pass plan step at the start of Understand. The Collect
and Synthesize stages are otherwise unchanged; they simply run `TOT_SURVIVORS` times in parallel
rather than once, and the controller merges the chapter sets (resolving contradictions in favour
of the highest-feasibility plan's figures).

### Capstone concept coverage

| Capstone concept | Where it lives |
| --- | --- |
| Tool calling | `tools/` + `prompts/tool_schemas.py` + `core/tool_router.py` |
| Reasoning / ReAct | `core/graph.py` (three-stage loop + backtracking) |
| Knowledge & memory (RAG) | `retrieval/` + `memory/` + `mcp/` (episodic) |
| Further reasoning (ToT) | `core/tot/` + critic in `agents/` |
| Multi-agent coordination | `agents/` (parallel domain sub-agents) + `core/` controller |
| Safety / guardrails | §10 + `core/guardrails.py` |

---

## 3. Layer reference

### `frontend/`
A **thin** client. It contains **no reasoning, retrieval, or business logic** — every action
is an HTTP call to `api/`. Implementation is a **Blazor WebAssembly** app (ASP.NET Core .NET
8+): `ApiClient.cs` is a typed `HttpClient` wrapper and all UI state is local to each page.
The frontend never talks to ChromaDB, SQLite, the MCP server, or any external API directly.

The app has **five pages**:

| Page | Route | Purpose |
| --- | --- | --- |
| **Front Page** | `/` | Newspaper-style daily intelligence brief automatically assembled each morning at 00:00; read-only, resets on a daily schedule. |
| **Agent** | `/agent` | ChatGPT-style conversational interface — users ask custom questions, request on-demand reports, and step through the three review gates (Plan → Confidence → Brief). |
| **Archive** | `/archive` | Searchable library of all completed intelligence reports, also serving as a visual browser of episodic memory (past successful execution plans and their outcomes). |
| **Dashboard** | `/dashboard` | Live and historical monitoring of agent runs: tool calls made, tokens consumed per stage, API cost, run durations, and gate decisions. |
| **Testing** | `/testing` | Curated test cases that can be executed on demand to verify end-to-end agent behaviour (tool calling, ToT planning, RAG retrieval, gate flow). Results displayed inline with pass/fail status. |

Each page is a routable Razor component under `Pages/`; shared widgets (gate panels, charts,
report cards) live in `Components/`.

### `core/`
The orchestration brain. Contains the LangGraph state graph (`graph.py`), the Tree-of-Thought
planner (`tot/`), the ReAct backtracking controller, the tool router (`tool_router.py`), and
the controller that dispatches survivors in parallel and merges synthesis output. This is the
only layer that coordinates across all others. Nothing outside `core/` imports from `core/`
except `api/` (to invoke the agent) and `main.py` (to start it).

### `agents/`
CrewAI agent definitions — the specialised workers `core/` dispatches:
- **Domain sub-agents** (one per active intelligence domain) — run targeted Collect tasks in parallel.
- **Grounding & critic agent** — depth-2 reality grounding, feasibility/quality scoring, diversity penalty.
- **Synthesis sub-agents** — per-chapter writing and cross-domain correlation.

Agents import `tools/`, `retrieval/`, `memory/`, `models/`, and `prompts/`. An agent has no
knowledge of the LangGraph graph or the API. `core/` owns *when* agents run; `agents/` owns
*what* each one does.

### `tools/`
Thin tool definitions — the function-calling interface between the LLM and the system. Each
file defines a tool name, description, Pydantic input schema, and a `run()` method that
delegates **immediately** to a service. Tools contain no business logic, HTTP calls, or data
access. All tool files sit flat — no subfolders.

`tools/registry.py` owns three per-stage tool lists. The **stage owns its toolset** — a tool
file has no knowledge of which stages use it, and the same tool object can appear in more than
one list:

| List | Stage | Contents |
| --- | --- | --- |
| `UNDERSTAND_TOOLS` | Understand | `IndustryKnowledgeTool` (search industry books/articles to frame the question), `EpisodicMemoryTool` (search past successful reports + execution plans as examples) |
| `COLLECT_TOOLS` | Collect | all external API tools (FMP, yfinance, EDGAR, NewsAPI, web search), SQLite write, ChromaDB write (to `collected_{run_id}`) |
| `SYNTHESIZE_TOOLS` | Synthesize | `IndustryKnowledgeTool` (interpret signals against domain knowledge), `EpisodicMemoryTool` (reference how past reports drew conclusions), SQLite query, web search; retrieval from `collected_{run_id}` is built-in (not a tool call) |

The two RAG tools (`IndustryKnowledgeTool`, `EpisodicMemoryTool`) are explicit LLM-callable tools — the agent decides when to invoke them. Retrieval from the collected-data store during Synthesize is automatic and happens inside the node, not via a tool call.

`tool_router.py` selects the correct list based on the active stage and passes it as the
`tools` parameter on each LLM call and each CrewAI `Task`, keeping irrelevant schemas out of
context.

### `services/`
Business logic. Services are called by tools (and by agents via tools); they orchestrate
caching, validation, transformation, error handling, and coordination across multiple clients.
A service knows nothing about the agent, the LLM, or any tool schema. Master-data loading from
`data/` happens **only** in `masterdata_service.py`.

### `clients/`
Raw external communication, one client per external API/SDK (SEC EDGAR, Alpha Vantage, yfinance,
Financial Modeling Prep, NewsAPI). Each handles auth, base URL, timeout, retries, and HTTP
status errors; returns raw dicts/dataclasses; raises on failure. No business logic. All extend
`base_http_client.py` for shared retry/timeout/rate-limit behaviour.

### `retrieval/`
The RAG layer. Owns all three ChromaDB collections, embeddings, chunking, and reranking.
Exposes a single `Retriever` interface that accepts a `collection` parameter — callers never
reference ChromaDB directly. The three collections are:

- `collected_{run_id}` — written by collect tools, read automatically during Synthesize.
- `episodic_memory` — written by `mcp/` after Gate 3 approval, read via `EpisodicMemoryTool`.
- `industry_knowledge` — seeded from `data/knowledge/` at startup, read via `IndustryKnowledgeTool`.

`retrieval/` depends on `models/` (embeddings) and is the **only** module that touches ChromaDB.
Swapping in Azure AI Search means re-implementing this interface and nothing else.

### `memory/`
Short-term + structured state. `context_window.py` does token budgeting, message history, and
prompt pruning so the coordinator holds compact structured result objects (a plan, a collection
manifest, a signal summary) rather than raw sub-agent output. `sqlite_store.py` owns the
**SQLite** structured/session store: collected figures (commodity prices, financial ratios,
capex), resolved master data, and persisted user preferences. Memory is the only module that
touches SQLite.

### `mcp/`
A **FastMCP** server that acts as the shared-state bus for the Tree of Thought. It exposes, as
MCP resources/tools, the planning state (all candidate plans at both depths, scores, the
coverage checklist, gap reports) and **episodic memory** (past successful execution plans). The
controller and every CrewAI agent read/write through this single context, so planning state is
decoupled from any one process and survives backtracking. Persistence is delegated to `memory/`.

### `reports/`
Report assembly and PDF generation. Takes the merged chapter sets plus the executive summary
and produces the final brief. Pure formatting — no retrieval or reasoning. Generated artefacts
go to `outputs/` (not version-controlled).

### `api/`
The FastAPI interface. Routers are thin: validate input, call `core/`, return a schema. The
human-review gates are exposed as resumable endpoints backed by the LangGraph checkpointer.

```
api/
├── main.py            # app init, lifespan, middleware
├── dependencies.py    # shared DI: get_agent(), get_checkpointer(), auth
├── routers/
│   ├── chat.py        # POST /chat, POST /chat/stream  (start a run)
│   ├── gates.py       # POST /runs/{id}/gates/{gate}/approve | /redirect  (resume)
│   ├── preferences.py # GET/PUT /preferences  (master data + persistent prefs)
│   ├── reports.py     # GET /runs/{id}/report  (download PDF)
│   ├── sessions.py    # GET/DELETE /sessions/{id}
│   └── health.py      # GET /health
└── schemas/
    ├── chat.py        # ChatRequest, ChatResponse, Message
    ├── gate.py        # PlanReview, ConfidenceSummary, BriefReview
    └── session.py     # Session, SessionList
```

### `models/`
The LLM interface. `llm_client.py` is a wrapper over the Mistral chat/embeddings API (the only
supported provider). `response_parser.py` parses raw output into structured types (thought, action,
action input, final answer) and validates ToT plan JSON. Nothing outside `core/`, `agents/`, and
`retrieval/` (embeddings) interacts with the LLM directly.

### `config/`
`settings.py` — the **single tuning surface** (the "config.py"); a Pydantic `BaseSettings` class
that loads and validates every tunable variable and secret (§9). `logging.py` configures
structured logging. Every module imports from `config/`; `config/` imports nothing else.

### `data/`
Version-controlled master data, organised by domain entity — the source of truth for static
reference data. Loaded at startup via `masterdata_service.py`, never imported directly. Paths are
referenced through `settings.py`, never hardcoded.

```
data/
├── equipment/          # Komatsu equipment models, classes, specs
├── sites/              # mine sites, regions, geography metadata
├── competitors/        # Caterpillar, Volvo CE, Liebherr, Epiroc — names, product codes, tickers
├── distributors/       # Komatsu + competitor dealer networks
└── knowledge/          # industry books, articles, technical reports seeded into ChromaDB industry_knowledge
    ├── mining/         # mining industry reports, commodity cycle analyses
    ├── construction/   # construction market reports
    └── heavy_equipment/ # OEM strategy, product lifecycle, procurement behaviour
```

Files in `data/knowledge/` are version-controlled (PDF, Markdown, or plain text). They are
ingested into ChromaDB `industry_knowledge` at startup via `masterdata_service.py` using the
same chunker/embedder pipeline in `retrieval/`. The collection is only re-ingested if the
content hash has changed (staleness check against `CORPUS_REFRESH` cadence).

### `tests/`
Mirrors source under `unit/` and `integration/`. Unit tests mock at the **client boundary**.
Integration tests run against real services and require env vars.

---

## 4. Dependency rules

Allowed import directions are strictly one-way. The frontend is a separate process and reaches
the backend **only over HTTP**.

```
frontend/ ──HTTP──▶ api/ ──▶ core/ ──▶ agents/ ──▶ tools/ ──▶ services/ ──▶ clients/
                                  │                                  ↘ memory/
                                  │                                  ↘ retrieval/
                                  │                                  ↘ data/ (via masterdata_service)
                                  ├──▶ mcp/ ──▶ memory/
                                  ├──▶ reports/
                                  ↘ models/   ↗ (also used by agents/, retrieval/)
                                  ↘ prompts/  ↗
config/  ◀── everything
```

**Violations to flag immediately:**
- A `client` importing from a `service`, `tool`, or `agent`
- A `service` importing from `core/`, `agents/`, or `api/`
- A `tool` containing an `httpx`/`requests` call or business logic
- Any module other than `retrieval/` touching ChromaDB (this covers all three collections: `collected_*`, `episodic_memory`, `industry_knowledge`)
- Any module other than `memory/` touching SQLite
- Any module other than `masterdata_service.py` reading files from `data/`
- The `frontend/` importing any backend module or calling an external API directly
- Anything outside `config/` reading `os.environ` or defining env vars

---

## 5. Naming conventions

| Layer | Suffix | Example |
| --- | --- | --- |
| Tool | `_tool.py` | `commodity_price_tool.py` |
| Service | `_service.py` | `commodity_service.py` |
| Client | `_client.py` | `alpha_vantage_client.py` |
| Agent | `_agent.py` | `competition_agent.py` |
| Router | none | `chat.py`, `gates.py` |
| Schema | none | `chat.py` inside `schemas/` |

Classes follow the same pattern: `CommodityPriceTool`, `CommodityService`,
`AlphaVantageClient`, `CompetitionAgent`.

---

## 6. The seven intelligence domains

Domains are configured in `settings.py` (`DOMAINS`), each with an enable flag and a weighting.
Each active domain maps to a CrewAI sub-agent in `agents/`.

| Domain | Scope | Primary tools |
| --- | --- | --- |
| Competition | Rivals (Caterpillar, Volvo CE, Liebherr, Epiroc): financials, launches, strategy | FMP, yfinance, SEC EDGAR, NewsAPI |
| Distributors | Komatsu + competitor dealer network health, events, coverage | NewsAPI, web search |
| Customers | Mining operators, construction firms, railways, armies, port authorities | NewsAPI, web search |
| Mining projects | Active/pipeline projects, capex announcements, investment cycles | SEC EDGAR, NewsAPI, web search |
| Commodities | Gold, silver, copper, oil — demand-cycle indicators | Alpha Vantage, yfinance |
| Macro & geopolitics | Trade policy, infrastructure spend, energy transition | NewsAPI, web search |
| General search | Open-web signals not covered by structured feeds | web search |

---

## 7. Adding a new external data source

Follow this sequence exactly — do not skip layers:

1. **Create a client** in `clients/` (extend `base_http_client.py`) for the raw API calls.
2. **Create a service** in `services/` for business logic and error handling.
3. **Create a tool** in `tools/` with a Pydantic input schema and a `run()` that calls the service.
4. **Register the tool** in `tools/registry.py`: add the tool instance to `UNDERSTAND_TOOLS`,
   `COLLECT_TOOLS`, and/or `SYNTHESIZE_TOOLS` — whichever stages need it. The same instance
   may appear in multiple lists.
5. **Add the tool schema** to `prompts/tool_schemas.py` so the LLM can call it.
6. **Wire it to a domain agent** in `agents/` if it belongs to a specific domain.
7. **Add env vars** to `config/settings.py` and `.env.example`; add cost/rate caps to `SAFETY`.
8. **Write unit tests** mocking at the client boundary.

### Adding new master data

1. **Create a subfolder** in `data/` named after the entity (e.g. `data/ports/`).
2. **Add files** as JSON or CSV — version-controlled, treat them as code.
3. **Extend `masterdata_service.py`** to load and expose the dataset.
4. **Add the path** to `config/settings.py` — never hardcode.
5. **Write unit tests** against the loader using the actual files (no mocking).

---

## 8. Memory & retrieval architecture

### Store taxonomy

The system uses **three distinct ChromaDB collections** serving separate purposes. They must not be conflated; each is accessed by a dedicated path.

| Store | Backend | Collection name | Module | Lifetime | Holds |
| --- | --- | --- | --- | --- | --- |
| Short-term context | in-process | — | `memory/context_window.py` | per turn | compact result objects, message history |
| Session structured | SQLite | — | `memory/sqlite_store.py` | **wiped per chat** | collected figures, manifests |
| **Collected unstructured** (internal long-term memory) | ChromaDB | `collected_{run_id}` | `retrieval/` | **wiped per chat** | news, filings, announcements gathered during Collect; the "factory store" that Synthesize reads from |
| User preferences / master data | SQLite | — | `memory/sqlite_store.py` | persistent | resolved entities, gate decisions, domain weights |
| **Industry knowledge base** | ChromaDB | `industry_knowledge` | `retrieval/` | persistent, refreshed quarterly | industry books, articles, technical reports seeded from `data/knowledge/`; provides the interpretive frame for both planning and synthesis; accessed via `IndustryKnowledgeTool` |
| **Episodic memory** | ChromaDB | `episodic_memory` | `retrieval/` | persistent | past successful research reports + their execution plans; written by `mcp/` after a report is approved; accessed via `EpisodicMemoryTool` |

Short-term memory holds **references**, not raw data: full content is written to the long-term
stores immediately and accessed by reference, keeping the coordinator's context stable regardless
of how many sub-agents ran. Episodic memory is enabled only once enough high-quality example
plans exist (`EPISODIC_ENABLED`).

### Retrieval is required

Live APIs solve **freshness** (prices, earnings, project news). They do **not** solve
**interpretation** — recognising what a fresh signal means for Komatsu depends on a stable body
of domain knowledge too large for any prompt and too specialised for the base model. Semantic
retrieval at inference time keeps context small while surfacing the interpretive frame each step
needs. A non-RAG design would fail precisely at the cross-domain reasoning that separates this
agent from a news aggregator.

### Three retrieval points

The three RAG systems each have a distinct purpose and activation path:

1. **Understand — query framing and planning** — the LLM calls `IndustryKnowledgeTool` with its
   current sub-question to surface domain knowledge that frames what data is needed and why.
   It also calls `EpisodicMemoryTool` to retrieve past successful execution plans as few-shot
   examples for planning. Both are **explicit LLM tool calls** — the agent decides when to invoke
   them; they are not injected automatically.

2. **Collect — write path (not a retrieval point)** — every piece of unstructured data gathered
   (news, filings, web content) is immediately chunked and written to ChromaDB `collected_{run_id}`
   by the collect tools. This is internal to the collect node; the LLM does not call a retrieval
   tool here.

3. **Synthesize — per-chapter interpretation and grounding** — each chapter node *automatically*
   retrieves its supporting chunks from `collected_{run_id}` and figures from SQLite (built-in,
   not a tool call). The synthesis agent may also call `IndustryKnowledgeTool` (to understand how
   to interpret what the data means for Komatsu) and `EpisodicMemoryTool` (to reference how past
   reports drew conclusions on similar signals).

All retrieval queries are generated from the agent's **current sub-question**, not the user's
original prompt, so retrieval is grounded in what must be resolved right now.

### Retrieval design choices

- **Chunking.** Domain-corpus documents are **semantically chunked** to preserve structure.
  Documents retrieved during Collect are chunked at `CHUNK_SIZE` (600 tokens) with
  `CHUNK_OVERLAP` (100). Prior-run summaries are stored as **one chunk per domain per run**.
- **Reranking.** Each query returns `RETRIEVAL_TOP_K` (5) candidates passed through a lightweight
  **cross-encoder reranker** (`RERANKER_MODEL`) scored against the specific sub-question; tangential
  chunks are dropped before entering context.
- **Staleness guard.** Every chunk carries a timestamp. At synthesis, prior-run results older
  than the per-domain `STALENESS_WINDOW_DAYS` are re-queried via live API or surfaced with a
  caveat; the corpus is versioned and refreshed quarterly; named-entity claims are weighted below
  API data (`NAMED_ENTITY_CONFIDENCE_DISCOUNT`). Worst-case staleness is routed to Gate 3, not
  silently embedded.

### Multi-agent coordination & the recovery path

`TOT_SURVIVORS` plans run in parallel; the controller merges their chapter sets. A
**diversity risk** exists: if the seven depth-1 plans converge, parallel execution adds no
coverage.
- *Prevention.* The propose prompt enforces variation on ≥ `TOT_MIN_DIVERSITY_DIMS` dimensions;
  the critic applies a diversity penalty to any plan structurally identical to a higher-ranked one.
- *Recovery.* After synthesis, the controller checks chapter-set overlap. If it exceeds
  `DIVERSITY_OVERLAP_THRESHOLD` (0.80), it pulls the highest-scoring structurally distinct plan
  from the four pruned plans held in MCP state and appends its output as a supplementary section —
  genuine breadth without a full re-run.

---

## 9. Configuration — `config/settings.py` (the single tuning surface)

Every tunable lives here, validated by Pydantic `BaseSettings`. **No other module reads
`os.environ` or hardcodes a parameter.** Secrets come from the environment (`.env`); behaviour
parameters have documented defaults. Group with nested models for clarity.

### LLM
| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `mistral` | Only `mistral` is supported |
| `LLM_MODEL` | `mistral-medium-latest` | Mistral model name |
| `LLM_PROPOSE_TEMPERATURE` | `0.9` | High temp for diverse depth-1 plans |
| `LLM_WORK_TEMPERATURE` | `0.2` | Low temp for grounding, synthesis |
| `LLM_MAX_TOKENS` | `4096` | Per-call output cap |
| `LLM_INPUT_PRICE_PER_1M` | `1.275` | USD / 1M input tokens (live cost counter) |
| `LLM_OUTPUT_PRICE_PER_1M` | `6.375` | USD / 1M output tokens (live cost counter) |

### Tree of Thought
| Variable | Default | Purpose |
| --- | --- | --- |
| `TOT_BRANCHING_FACTOR` | `7` | Depth-1 plans generated |
| `TOT_DEPTH` | `2` | Tree depth (propose + grounding) |
| `TOT_SURVIVORS` | `3` | Plans dispatched to Collect in parallel |
| `TOT_MIN_DIVERSITY_DIMS` | `3` | Dimensions each plan must vary on |
| `FEASIBILITY_WEIGHT` | `0.60` | Score weight — executability |
| `QUALITY_WEIGHT` | `0.40` | Score weight — query coverage |
| `DIVERSITY_PENALTY` | `0.25` | Critic penalty for structural duplicates |
| `DIVERSITY_OVERLAP_THRESHOLD` | `0.80` | Trigger for the recovery path |

### ReAct loop & feasibility budgets
| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_REACT_ITERATIONS` | `8` | Hard cap on reformulation/backtrack cycles |
| `CONFIDENCE_THRESHOLD` | `0.75` | Convergence stopping condition |
| `HARD_TIME_LIMIT_S` | `180` | Wall-clock cap → partial brief with flagged gaps |
| `PER_TOOL_LATENCY_BUDGET_S` | `10` | Feasibility check per data point |
| `RUN_TOKEN_BUDGET` | `200000` | Per-run token budget |

### Retrieval
| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` | `600` | Tokens per Collect-time chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap |
| `RETRIEVAL_TOP_K` | `5` | Candidates per query before rerank |
| `RERANKER_ENABLED` | `true` | Toggle cross-encoder rerank |
| `RERANKER_MODEL` | — | Cross-encoder model id |
| `EMBEDDING_MODEL` | — | Embedding model id |
| `STALENESS_WINDOW_DAYS` | `{domain: days}` | Per-domain re-query window |
| `CORPUS_REFRESH` | `quarterly` | Domain-corpus refresh cadence |
| `NAMED_ENTITY_CONFIDENCE_DISCOUNT` | `0.7` | Down-weight vs API data |

### Stores
| Variable | Default | Purpose |
| --- | --- | --- |
| `SQLITE_PATH` | `./outputs/intel.db` | Structured/session store |
| `CHROMA_PATH` | `./outputs/chroma` | Root path for all ChromaDB collections |
| `CHROMA_COLLECTED_COLLECTION` | `collected_{run_id}` | Per-run unstructured collected data (factory store) |
| `CHROMA_EPISODIC_COLLECTION` | `episodic_memory` | Persistent episodic memory (past reports + plans) |
| `CHROMA_KNOWLEDGE_COLLECTION` | `industry_knowledge` | Persistent industry knowledge base |
| `WIPE_SESSION_STORES_ON_CHAT` | `true` | Clear session SQLite + collected ChromaDB per chat |
| `EPISODIC_ENABLED` | `false` | Enable once enough quality examples exist |
| `EPISODIC_MIN_QUALITY_SCORE` | `0.75` | Minimum combined ToT score for a plan to be stored in episodic memory |
| `VECTOR_BACKEND` | `chroma` | `chroma` \| `azure_search` (swap behind `retrieval/`) |

### Domains
| Variable | Default | Purpose |
| --- | --- | --- |
| `DOMAINS` | 7 domains, all enabled | Per-domain `{enabled, weight}` |
| `MAX_PARALLEL_SUBAGENTS` | `3` | Bound on concurrent Collect agents |

### Human-in-the-loop
| Variable | Default | Purpose |
| --- | --- | --- |
| `GATE_1_ENABLED` / `GATE_2_ENABLED` / `GATE_3_ENABLED` | `true` | Toggle each review gate |
| `AUTO_APPROVE_GATES` | `false` | Dev-only: skip human review |
| `GATE_TIMEOUT_S` | `600` | Auto-pause expiry per gate |

### Report
| Variable | Default | Purpose |
| --- | --- | --- |
| `EXEC_SUMMARY_MIN_WORDS` / `EXEC_SUMMARY_MAX_WORDS` | `400` / `500` | Summary length band |
| `REPORT_FORMAT` | `pdf` | Output format |
| `OUTPUT_DIR` | `./outputs` | Generated artefacts (not version-controlled) |

### External clients
Per client: `*_API_KEY` (env-only), `*_BASE_URL`, `*_TIMEOUT_S`, `*_MAX_RETRIES`,
`*_RATE_LIMIT_PER_MIN`.

### Safety
| Variable | Default | Purpose |
| --- | --- | --- |
| `TOOL_ALLOWLIST` | all registered | Tools the agent may call |
| `MAX_SPEND_PER_RUN_USD` | `1.00` | Hard cost cap on paid APIs |
| `MAX_API_CALLS_PER_RUN` | `100` | Hard call cap |
| `ALLOW_NETWORK_WRITES` | `false` | Master kill-switch for any side-effecting call |

---

## 10. Safety & guardrails

The agent reads untrusted web content and calls paid, rate-limited APIs, so guardrails are
explicit and enforced in `core/guardrails.py`, not left to prompt wording.

**Potential unintended actions and mitigations:**
- *Runaway cost / API abuse.* Every run is bounded by `MAX_SPEND_PER_RUN_USD`,
  `MAX_API_CALLS_PER_RUN`, and per-client rate limits in `base_http_client.py`. Exceeding a cap
  aborts the run with a partial brief rather than continuing.
- *Prompt injection from retrieved content.* Web pages, filings, and news are **data, not
  instructions**. Retrieved text is never executed as commands; any instruction-like content
  found in a source is surfaced to the user, never acted on. Tool calls are restricted to
  `TOOL_ALLOWLIST`.
- *No side effects.* The agent is read-only by design. It performs no trades, transfers,
  purchases, or writes to external systems; `ALLOW_NETWORK_WRITES` defaults off as a kill-switch.
- *Hallucinated figures.* All time-sensitive numbers route through live API calls; the staleness
  guard and `NAMED_ENTITY_CONFIDENCE_DISCOUNT` down-weight stale or model-supplied claims; low-
  confidence domains are flagged rather than asserted.
- *Silent bad assumptions.* Master-data clarification validates entities (equipment models, mine
  sites, competitor names, tickers) before the first run and asks a clarifying question with a
  suggested correction rather than proceeding on a bad guess.
- *Unsupervised execution.* The three review gates (LangGraph interrupts) keep a human in the
  loop at every stage; `AUTO_APPROVE_GATES` is dev-only and off in any deployed configuration.

---

## 11. Human-in-the-loop gates

Gates are LangGraph **interrupts** backed by the checkpointer, so a run pauses, persists its
state, and resumes when the frontend posts a decision. Each gate maps to one API endpoint and
one frontend view.

- **Gate 1 — after Understand.** Frontend shows the top-`TOT_SURVIVORS` plans with scores and gap
  reports. User approves, edits scope, or rejects. Catches misaligned plans before any data is collected.
- **Gate 2 — after Collect.** Frontend shows a per-domain confidence summary (strong vs incomplete).
  User can redirect or re-run a domain before synthesis.
- **Gate 3 — after Synthesize.** Frontend shows the assembled brief. User flags sections for removal
  or deeper coverage on the next run, and downloads the PDF.

**Master-data clarification** runs before the first Gate 1: the agent prompts for entity-level
inputs, validates them, and stores them as persistent preferences in SQLite. **Persistent
preferences** (gate decisions, domain weighting, entity resolution) are retrieved at the start of
later runs, progressively shaping scope without manual reconfiguration.

---

## 12. Build roadmap (step by step)

Each phase is independently runnable and maps to a capstone checkpoint/metric. Build in order;
do not start a phase until the previous one passes its tests.

| Phase | Goal | Delivers | Checkpoint |
| --- | --- | --- | --- |
| 0 | Skeleton | folders, `config/settings.py`, `models/llm_client.py`, `api/health`, `main.py` | — |
| 1 | Tool calling | one `client → service → tool` slice (Alpha Vantage commodity price), registry, schema | 1.1 |
| 2 | Reasoning loop | LangGraph three-stage graph (Understand→Collect→Synthesize) + ReAct backtracking, single-plan | 1.2 |
| 3 | Memory | `memory/` short-term context + SQLite session/preferences store | 1.2 |
| 4 | Retrieval (RAG) | `retrieval/` ChromaDB, chunking, reranker, the three retrieval points, staleness guard | 3.1 |
| 5 | Tree of Thought | `core/tot/` propose + CrewAI critic grounding + scoring/pruning, `mcp/` state | 4.1 |
| 6 | Multi-agent | `agents/` parallel domain sub-agents; controller dispatch + merge; recovery path | 4.1 |
| 7 | Gates + frontend | LangGraph interrupts, `api/gates.py`, Agent page with gate flow; Front Page daily brief; Archive + episodic-memory browser; Dashboard metrics; Testing harness; master-data clarification | 1.2 |
| 8 | Safety | `core/guardrails.py`, cost/call caps, injection handling, allowlist | Safety |
| 9 | Reporting | `reports/` chapter merge + 400–500 word exec summary + PDF assembly | Deliverables |

Phases 1–3 already deliver a working ReAct tool-calling agent with memory; ToT and multi-agent
parallelism layer on top without rewriting the spine.

---

## 13. Testing conventions

- `tests/` mirrors the source tree under `unit/` and `integration/`.
- **Unit tests** mock at the **client boundary** — never mock services or tools internally.
- **Integration tests** run against real services and require env vars; mark them so they can be
  skipped in CI without keys.
- **Master-data loaders** are tested against the actual files in `data/` — no mocking.
- **ToT and retrieval** get golden-file tests: a fixed prompt must produce plans/chunks within
  documented score and overlap bounds, so config changes have measurable effects.

---

## 14. What this file is not

This file documents structure, conventions, and the build plan. It does **not** document:
- Individual function signatures (see docstrings)
- API endpoint contracts (see `api/schemas/`)
- Deployment or infrastructure (see `README.md`)
- Secret values (see `.env.example` and your environment)
