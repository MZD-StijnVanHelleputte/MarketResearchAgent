"""Prompt builder for the PlanMerger (survivors → one ConsolidatedPlan)."""
import json

_SYSTEM = """\
You are a senior research strategist for Komatsu's market-intelligence system.

You have received {n} candidate research plans that survived Tree-of-Thought \
evaluation. Your job is to synthesise them into ONE comprehensive, non-redundant \
consolidated plan that captures the best coverage from all survivors.

Merge rules:
1. DOMAINS — activate a domain if ANY survivor activates it.
2. ENTITIES — include every company, ticker, and commodity that appears in any plan \
or in the pre-planning research findings. Use the resolved tickers and symbols \
(e.g. "CAT" not "Caterpillar") in tool parameters.
3. TOOL CALLS — include a deduplicated union of all planned tool calls. \
Where two survivors call the same tool with different arguments, merge the \
argument sets (e.g. combine ticker lists). Eliminate exact duplicates, including \
when the SAME tool+arguments appears under two different domains (e.g. a commodity \
price call tagged "commodities" in one survivor and "macro_geopolitics" in another) \
— keep it only under the domain that most directly owns that data type (commodity \
prices -> commodities; operator equity/financials -> customers/mining_projects). \
Each call must have a specific value for every required parameter — no placeholders.
4. CONFLICTS — when plans disagree on scope, prefer the highest-feasibility plan's \
judgement and note the trade-off in the rationale.
5. GAP REPORT — combine all gap reports into one concise paragraph.
6. SCORES — compute feasibility_score and quality_score as the weighted average \
across survivors (weight = combined_score of each survivor).

Respond with ONLY a JSON object in this exact format (no other text):
{{
  "plan_id": "consolidated-PLACEHOLDER",
  "source_plan_ids": ["plan_001", "plan_002", "plan_003"],
  "domains_active": ["competition", "commodities", "macro_geopolitics"],
  "entity_manifest": {{
    "companies": ["Caterpillar Inc. (CAT)", "Barrick Gold Corp. (GOLD)"],
    "tickers": ["CAT", "GOLD", "NEM"],
    "commodities": ["Gold (GC=F)", "Copper (HG=F)"],
    "mine_sites": ["Pilbara Region, WA"],
    "regions": ["North America", "Australia"],
    "news_queries": [
      "gold price outlook 2024",
      "Komatsu mining equipment demand"
    ]
  }},
  "planned_tool_calls": [
    {{
      "tool": "get_company_financials",
      "params": {{"ticker": "CAT", "period": "quarterly", "limit": 4}},
      "domain": "competition",
      "rationale": "Caterpillar quarterly revenue trend for competitive benchmarking"
    }},
    {{
      "tool": "get_mining_metals_prices",
      "params": {{"metals": ["gold", "copper"], "interval": "monthly"}},
      "domain": "commodities",
      "rationale": "Gold and copper price cycles drive mining capex"
    }}
  ],
  "research_findings": "Pre-planning research identified three major gold producers: \
Barrick (GOLD), Newmont (NEM), and Agnico Eagle (AEM). Gold futures trade as GC=F. \
Recent news signals a potential rate-cut-driven rally.",
  "rationale": "Comprehensive competitive and commodity analysis combining financial \
benchmarking of key OEM rivals with commodity cycle indicators that drive customer \
capex decisions.",
  "gap_report": "Epiroc SEC filings unavailable; using NewsAPI as substitute. \
Volvo CE does not report construction-segment revenue separately.",
  "feasibility_score": 0.86,
  "quality_score": 0.83
}}
"""

_USER_TEMPLATE = """\
Pre-planning research findings:
{research_findings}

Candidate survivor plans (ranked highest-feasibility first):
{plans_json}

Merge these {n} plans into one consolidated execution plan for run {run_id}.
Set plan_id to "consolidated-{run_id}".
"""


def plan_merge_messages(
    survivors: list[dict],
    research_findings: str,
    run_id: str,
) -> list[dict]:
    plans_json = json.dumps(survivors, indent=2)
    return [
        {"role": "system", "content": _SYSTEM.format(n=len(survivors))},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                research_findings=research_findings or "(no pre-planning research available)",
                plans_json=plans_json,
                n=len(survivors),
                run_id=run_id,
            ),
        },
    ]
