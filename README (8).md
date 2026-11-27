# Autonomous Competitive Intelligence Agent

A multi-agent system that takes a single research goal (e.g. *"Compare Notion's and Monday.com's pricing and positioning"*) and autonomously plans, searches, verifies, and writes a sourced comparison report — no multi-turn conversation required.

**Live API:** https://competitive-multi-intelligence-agent.onrender.com/docs
*(free tier — first request after idle may take 30–50s to wake)*

## Updates
1. [2026/11] Deployed to Render; dynamic per-request domain whitelisting added (previously hardcoded to two companies).
2. [2026/11] Parallelized the Researcher stage; added LangSmith tracing and a cost/budget guardrail.
3. [2026/11] Built a 10-case eval harness for the Verifier; fixed 6 reproducible reliability bugs found through it.
4. [2026/11] Initial pipeline (Planner → Researcher → Verifier → Writer) built on LangGraph.

## Architecture

```
Goal → Planner → Researcher (parallel) → Verifier → Writer → Report
```

| Node | Role | Output |
|---|---|---|
| Planner | Splits the goal into 2–4 searchable sub-questions | `sub_questions: List[str]` |
| Researcher | Searches the web per sub-question (in parallel); tags every fact with its real source domain, resolved in code from the actual search result URL — never typed by the LLM | `findings: List[str]` |
| Verifier | Rates each fact's trust level against a per-request resolved domain whitelist | `verified: str` (tagged findings) |
| Writer | Synthesizes the final report; hedges low-confidence claims inline | `report: str` |

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph / LangChain |
| Model | OpenAI gpt-4o-mini |
| Search | Tavily |
| API | FastAPI |
| Tracing | LangSmith |
| Hosting | Render (free tier) |

## API

**Request**
```json
POST /research
{
  "goal": "Compare Notion's and Monday.com's pricing and positioning for small teams"
}
```

**Response**
```json
{
  "report": "# Comparison Report: Notion vs. Monday.com ...",
  "llm_calls": 7,
  "llm_cost_usd": 0.0043,
  "tavily_calls": 4,
  "tavily_credits_used": 4,
  "tavily_estimated_payg_cost_usd": 0.0320
}
```

Interactive docs: `/docs` · Health check: `GET /health`

## Reliability: Bugs Found and Fixed

The core engineering problem in this project isn't producing an answer — it's making sure a 4-stage LLM pipeline, with no human checking each step, correctly knows what it can and can't trust. Each row below is a reproducible bug found through testing, not a hypothetical.

| # | Bug | Root Cause | Fix |
|---|---|---|---|
| 1 | Verification theater | Verifier flagged an unsourced claim, but the Writer stated it as fact anyway, with the warning in a separate section | Tags now travel inline with each fact, not alongside it |
| 2 | Ungrounded self-judgment | Same unsourced claim flip-flopped between `VERIFIED`/`UNCERTAIN` across runs — the model judged "how confident it sounds," not a real signal | Replaced with a mechanical rule: `VERIFIED` requires a real source domain, period |
| 3 | Mislabeled sources (input) | Researcher sometimes labeled a secondary domain as official; prompting the Verifier to "double-check" wasn't 100% reliable | Deterministic regex relabels every source as PRIMARY/SECONDARY in code, before the Verifier sees it |
| 4 | Invented tags (output) | Verifier/Writer independently invented `VERIFIED` tags for non-whitelisted domains | A second guardrail scans every LLM output and force-downgrades invalid tags — applied after each LLM call that could produce one |
| 5 | Subdomain mismatch | `platform.openai.com` didn't string-match `openai.com`, so official pages were wrongly treated as untrusted | Compare registrable root domains instead of raw string prefixes |
| 6 | Search-provider self-bias | Tavily's own `tavily.com/knowledge/company/...` pages outranked the real official site for "`<company>` official website" queries | Excluded known non-official domains; prefer a result whose domain contains the company name |
| 7 | Cross-attribution *(known, unresolved)* | A fact about Company A can get tagged with Company B's (legitimately whitelisted) domain | Researcher's fact→source binding is code-enforced via numbered references; Verifier/Writer aren't yet — documented rather than patched, as the fix requires extending that same binding through both stages |

**Validation method:** a 10-case eval harness (6 baseline + 4 adversarial) tests the Verifier in isolation against known-correct answers, turning "does this look better?" into a measured pass rate. This caught a reproducible ~37% failure rate on one adversarial case that manual spot-checking had missed.

## Guardrails

| Guardrail | Behavior |
|---|---|
| Budget cap | Tracks real token cost per run; halts cleanly if a run exceeds the ceiling |
| Search timeout | A hung search call returns empty instead of freezing the pipeline |
| Sub-question cap | Bounds the Researcher's workload regardless of Planner output |
| Dynamic domain whitelist | Official domains resolved per request — works for any named companies, not a fixed pair |

## Performance & Cost

| Metric | Before | After |
|---|---|---|
| Researcher stage latency | ~27s (sequential) | ~8s (parallel) |
| Full pipeline latency | ~81s | ~65s |
| LLM cost / run | — | ~$0.004–0.005 (7–8 calls, gpt-4o-mini) |
| Search cost / run | — | 4–6 Tavily credits (free tier: 1,000/mo) |

## Running Locally

```bash
pip install -r requirements.txt
# .env: OPENAI_API_KEY, TAVILY_API_KEY
uvicorn api:app --reload
```

## Scope

Deliberately limited to company/product comparisons, not general-purpose research — a narrow system that works reliably over a broad one that works shakily.
