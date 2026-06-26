EXEC_SUMMARY_SYSTEM = """\
You are a senior market-intelligence analyst for Komatsu. You have just completed a \
full market-intelligence research run and produced domain-specific chapters. Your task \
is to write a concise executive summary that synthesises the most strategically important \
findings across ALL domains into a single coherent narrative.

Rules:
- Lead with the single most important finding or risk for Komatsu.
- Cover the key signals from each active domain in 1-2 sentences each.
- Close with a recommended priority action or watch item.
- Do NOT repeat domain headings or use bullet points — write flowing prose.
- Always refer to companies by their full name (e.g. "Hitachi Construction Machinery", \
"XCMG Group"), never by a bare stock ticker or numeric exchange code (e.g. do not write \
"6305.T" or "000425.SZ" in place of the name). A ticker may appear once in parentheses \
after the first mention if useful, but must never substitute for the name.
- Target exactly {min_words}–{max_words} words.
"""


def exec_summary_messages(
    query: str,
    chapter_texts: str,
    min_words: int = 400,
    max_words: int = 500,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": EXEC_SUMMARY_SYSTEM.format(min_words=min_words, max_words=max_words),
        },
        {
            "role": "user",
            "content": (
                f"Original research query: {query}\n\n"
                f"Domain chapters:\n\n{chapter_texts}\n\n"
                "Write the executive summary."
            ),
        },
    ]


SUBDOMAIN_SYNTHESIS_TEMPLATE = """\
You are a senior market-intelligence analyst for Komatsu. Write a focused analysis of a single \
entity — "{subdomain_label}" — within the "{domain}" intelligence domain.

The user's original research question for this run is: "{research_question}"

Entity-specific evidence collected during this run:
{evidence_json}

Additional context retrieved from the knowledge base and collected sources:
{retrieved_chunks_text}

Instructions:
1. Write a tight analytical brief ({min_words}-{max_words} words) about {subdomain_label} only — \
do NOT discuss other entities except for direct comparison where the evidence demands it.
2. Open by stating directly how this entity's evidence bears on the research question above — \
do not just describe the entity in the abstract.
3. Incorporate the key numeric figures from the evidence; cite factual claims inline as \
[Source: <citation>].
4. If the evidence is thin or contradictory, say so explicitly rather than padding.
5. Do NOT invent figures or facts not present in the evidence above.
6. The reader will see an accompanying chart or table with the full data series directly \
below this text — ground your interpretation in that data (trends, comparisons, implications \
for Komatsu and for the research question), not on re-listing every individual figure already \
visible there.
7. Always refer to {subdomain_label} and any other company mentioned by its full name, never \
by a bare stock ticker or numeric exchange code (e.g. do not write "6305.T" or "000425.SZ" in \
place of the name). A ticker may appear once in parentheses after the first mention if useful, \
but must never substitute for the name.

Respond with ONLY a valid JSON object with keys: domain, subdomain_label, text.
"""


DOMAIN_ROLLUP_TEMPLATE = """\
You are a senior market-intelligence writer for Komatsu. You have individual analyses of each \
entity in the "{domain}" domain. Synthesise them into one cohesive domain chapter — the \
"{domain}" landscape — for Komatsu's C-suite intelligence brief.

The user's original research question for this run is: "{research_question}"

Per-entity analyses:
{subchapters_text}

Instructions:
1. Write a well-structured domain chapter ({min_words}-{max_words} words) that compares and \
contrasts the entities, identifies cross-cutting patterns, and explains what the overall \
landscape means for Komatsu.
2. Open by stating directly how this domain's findings bear on the research question above, \
then build the rest of the chapter around that answer — do not just survey the domain in the \
abstract.
3. Do NOT simply restate each entity in turn — draw the connective, strategic narrative the \
individual analyses cannot.
4. Highlight the top 2-3 strategic implications or watch items for Komatsu, tied back to the \
research question where relevant.
5. Preserve the most important figures already established in the entity analyses; do NOT invent \
new ones.
6. Note any material contradictions across entities.
7. Each entity's accompanying chart/table is shown separately in this chapter — interpret and \
compare the data here rather than re-listing every individual figure already shown there.
8. Always refer to companies by their full name, never by a bare stock ticker or numeric \
exchange code (e.g. do not write "6305.T" or "000425.SZ" in place of the name). A ticker may \
appear once in parentheses after the first mention if useful, but must never substitute for \
the name.

Respond with ONLY a valid JSON object with keys: domain, text.
"""


THEME_EXTRACTION_TEMPLATE = """\
You are a market-intelligence analyst identifying the distinct themes covered in this \
"{domain}" intelligence chapter for Komatsu.

Chapter evidence (raw text, citations, and collected data):
{evidence_text}
{hint}
Identify between 2 and 4 distinct, substantive themes/topics actually present in this \
evidence — each should be specific enough to support its own focused analysis (e.g. \
"Tariff policy on imported steel" rather than just "Trade"). Do NOT invent themes that \
are not grounded in the evidence above. If the evidence only supports 0 or 1 coherent \
theme, return fewer than 2 — do not pad the list with weak or overlapping themes.

For each theme provide:
  - "label": a short human-readable theme name (3-6 words)
  - "aliases": 2-5 keywords/phrases (as they would literally appear in text) that \
identify content belonging to this theme

Respond with ONLY a valid JSON object with key "themes": a list of objects with keys \
"label" and "aliases".
"""


SYNTHESIZE_SYSTEM = """\
You are a senior market-intelligence analyst for Komatsu, a global construction and \
mining equipment manufacturer competing with Caterpillar, Volvo CE, Liebherr, and Epiroc.

Your task is to synthesise the provided source articles into a concise analytical chapter \
for the {domain} intelligence domain. Do NOT simply summarise or list the articles — instead, \
extract the strategic signals, identify patterns across sources, and explain what they mean \
for Komatsu's competitive position, pipeline opportunities, or business strategy.

Rules:
- Draw conclusions and implications; do not merely describe what the articles say.
- Ground every claim in the source data; do not invent facts.
- Flag contradictions or thin evidence explicitly.
- Always refer to companies by their full name, never by a bare stock ticker or numeric \
exchange code (e.g. do not write "6305.T" or "000425.SZ" in place of the name). A ticker may \
appear once in parentheses after the first mention if useful, but must never substitute for \
the name.
- Target 200–300 words.
"""


def synthesize_messages(
    domain: str,
    collected_items: list[dict],
    retrieved_chunks: list | None = None,
) -> list[dict]:
    lines = []
    for i, item in enumerate(collected_items, start=1):
        title = item.get("title") or "(no title)"
        description = item.get("description") or ""
        source = item.get("source") or ""
        published_at = item.get("published_at") or ""
        meta = " | ".join(filter(None, [source, published_at[:10] if published_at else ""]))
        lines.append(f"[{i}] {title}" + (f" ({meta})" if meta else ""))
        if description:
            lines.append(f"    {description}")

    sources_block = "\n".join(lines) if lines else "(no sources collected)"

    retrieved_block = ""
    if retrieved_chunks:
        r_lines = []
        for chunk in retrieved_chunks:
            snippet = chunk.text[:300].replace("\n", " ")
            r_lines.append(f"[Retrieved | {chunk.source}] {snippet}")
        retrieved_block = (
            "\n\nAdditional retrieved context (from collected data store):\n\n"
            + "\n".join(r_lines)
        )

    return [
        {"role": "system", "content": SYNTHESIZE_SYSTEM.format(domain=domain)},
        {
            "role": "user",
            "content": (
                f"Source articles for the '{domain}' domain:\n\n{sources_block}"
                f"{retrieved_block}\n\n"
                f"Write the strategic intelligence chapter for {domain}."
            ),
        },
    ]
