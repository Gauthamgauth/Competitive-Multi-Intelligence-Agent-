"""
Phase 2: Multi-agent graph.
Planner -> Researcher -> Verifier -> Writer, coordinated via LangGraph StateGraph.
Each node has ONE job. This is what makes the system debuggable and extensible.
"""

import os
import re
import time
import threading
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langsmith import traceable

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearch(max_results=5)

# NOTE: OFFICIAL_DOMAINS is no longer a hardcoded constant - it's resolved dynamically
# per request (see resolve_official_domains below), since this agent can research ANY
# companies named in the goal, not just Notion/Monday.com.

# --- Phase 3 guardrails: containment, not correctness ---
MAX_SUB_QUESTIONS = 4        # planner can't blow up the researcher's workload
SEARCH_TIMEOUT_SECONDS = 15  # a hung search call can't freeze the whole run
MAX_BUDGET_USD = 0.50        # hard stop if a single run gets too expensive

# gpt-4o-mini pricing (per 1M tokens) - update if pricing changes
PRICE_PER_1M_INPUT = 0.150
PRICE_PER_1M_OUTPUT = 0.600

_executor = ThreadPoolExecutor(max_workers=4)


class BudgetExceededError(Exception):
    """Raised when a single run's estimated cost exceeds MAX_BUDGET_USD."""
    pass


class CostTracker:
    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd
        self.total_cost = 0.0
        self.call_count = 0
        self._lock = threading.Lock()  # parallel researcher calls hit this concurrently

    def track(self, response) -> None:
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT
        cost += (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT
        with self._lock:
            self.total_cost += cost
            self.call_count += 1
            over_budget = self.total_cost > self.budget_usd
            current_total, current_count = self.total_cost, self.call_count
        if over_budget:
            raise BudgetExceededError(
                f"Run stopped: estimated cost ${current_total:.4f} exceeded the "
                f"${self.budget_usd:.2f} budget after {current_count} LLM calls."
            )

    def reset(self) -> None:
        """Start a fresh budget - call this at the start of each new API request,
        otherwise the budget cap applies across your server's ENTIRE lifetime,
        not per-request, and every request after the first few would start
        dangerously close to (or already past) the limit."""
        with self._lock:
            self.total_cost = 0.0
            self.call_count = 0


cost_tracker = CostTracker(MAX_BUDGET_USD)

# --- Tavily cost tracking (separate from LLM cost above) ---
# Confirmed from docs.tavily.com/documentation/api-credits: basic search = 1 credit/request.
# Free tier: 1,000 credits/month. Pay-as-you-go beyond that: $0.008/credit.
TAVILY_CREDITS_PER_BASIC_SEARCH = 1
TAVILY_PAYG_PRICE_PER_CREDIT = 0.008


class TavilyUsageTracker:
    def __init__(self):
        self.credits_used = 0
        self.call_count = 0
        self._lock = threading.Lock()

    def track(self):
        with self._lock:
            self.credits_used += TAVILY_CREDITS_PER_BASIC_SEARCH
            self.call_count += 1

    def reset(self) -> None:
        with self._lock:
            self.credits_used = 0
            self.call_count = 0

    @property
    def estimated_payg_cost(self) -> float:
        """What this would cost if you were past the free tier and on PAYG pricing."""
        return self.credits_used * TAVILY_PAYG_PRICE_PER_CREDIT


tavily_tracker = TavilyUsageTracker()


@traceable(name="llm_call")
def call_llm(prompt: str):
    """Every LLM call goes through here - this is the one place cost is tracked."""
    response = llm.invoke(prompt)
    cost_tracker.track(response)
    return response


@traceable(name="tavily_search")
def call_search_with_timeout(query: str):
    """
    Runs the Tavily search with a hard timeout. If it hangs or errors,
    returns an empty result instead of letting the whole pipeline freeze -
    the researcher already knows how to handle 'no search results found'.
    """
    future = _executor.submit(search_tool.invoke, {"query": query})
    tavily_tracker.track()  # counts the call attempt, matching how Tavily bills it
    try:
        return future.result(timeout=SEARCH_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        print(f"  [TIMEOUT] Search for '{query}' took too long - skipping, no hallucinated fallback.")
        return {"results": []}
    except Exception as e:
        print(f"  [SEARCH ERROR] '{query}' failed ({e}) - skipping, no hallucinated fallback.")
        return {"results": []}


@traceable(name="resolve_official_domains")
def resolve_official_domains(goal: str) -> List[str]:
    """
    Dynamic replacement for the old hardcoded OFFICIAL_DOMAINS list.
    1. Ask the LLM which companies/products the goal is actually about.
    2. For each one, search '<company> official website' and take the top
       result's real domain (extracted in code, same no-LLM-typed-domains
       principle as research_one_question).
    This means the agent can research ANY companies, not just Notion/Monday.com,
    and still knows which domain to trust as primary for THIS specific run.
    """
    extract_prompt = (
        f"What company or product names are mentioned in this research goal? "
        f"Return ONLY the names, one per line, no extra text.\n\nGoal: {goal}"
    )
    response = call_llm(extract_prompt)
    company_names = [c.strip() for c in response.content.split("\n") if c.strip()]
    print(f"  [DOMAIN RESOLVER] Extracted company names: {company_names}")

    # Discovered via testing: Tavily's own 'company knowledge' pages
    # (tavily.com/knowledge/company/...) outrank the REAL official site for a
    # query like '<company> official website' - plus generic wikis/aggregators
    # also show up high. None of these are ever a legitimate "official site"
    # for someone else's company, so they're excluded before picking a domain.
    NEVER_OFFICIAL = {"tavily.com", "wikipedia.org", "trustpilot.com", "g2.com", "capterra.com"}

    resolved_domains = []
    for company in company_names:
        search_query = f"{company} official website"
        raw_results = call_search_with_timeout(search_query)
        result_list = raw_results.get("results", []) if isinstance(raw_results, dict) else []
        print(f"  [DOMAIN RESOLVER] Query '{search_query}' -> "
              f"{[r.get('url', '') for r in result_list[:3]]}")

        # Filter out known non-official domains first.
        candidates = []
        for r in result_list:
            domain = get_domain(r.get("url", ""))
            base = base_domain(domain)
            if domain != "unknown" and base not in NEVER_OFFICIAL:
                candidates.append(domain)

        if not candidates:
            continue

        # Prefer a candidate whose domain actually contains the company's name
        # (e.g. 'slack' in 'slack.com') over just taking whatever's first.
        company_slug = re.sub(r"[^a-z0-9]", "", company.lower())
        best = next(
            (d for d in candidates if company_slug in d.lower().replace("-", "").replace(".", "")),
            candidates[0],  # fall back to the first non-excluded result
        )
        resolved_domains.append(best)

    print(f"  [DOMAIN RESOLVER] Treating as official for this run: {resolved_domains}")
    return resolved_domains


def get_domain(url: str) -> str:
    """Extract the bare domain from a URL, in code - never typed by an LLM."""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def base_domain(domain: str) -> str:
    """
    Reduce a domain to its registrable root: platform.openai.com -> openai.com,
    docs.anthropic.com -> anthropic.com. Needed because official product/API
    pages often live on a subdomain, not the bare marketing homepage, and a
    literal string prefix match (the old approach) fails on that: 'platform.
    openai.com'.startswith('openai.com') is False even though it's clearly
    the same company. This isn't a perfect public-suffix-list implementation
    (edge cases like co.uk exist), but it's correct for the common case and
    far better than exact-prefix matching.
    """
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def label_domain(domain: str, official_domains: List[str]) -> str:
    is_official = any(base_domain(domain) == base_domain(d) for d in official_domains)
    return "PRIMARY" if is_official else "SECONDARY"


def fix_mislabeled_sources(text: str, official_domains: List[str]) -> str:
    """
    Code-level guardrail: force 'Source: <domain> (PRIMARY)' to (SECONDARY)
    if the domain isn't actually on this run's resolved whitelist. Runs before
    the verifier LLM call so there's no mislabeled tag left for it to trust.
    """
    def correct_label(match):
        domain = match.group(1)
        is_official = any(base_domain(domain) == base_domain(d) for d in official_domains)
        label = "PRIMARY" if is_official else "SECONDARY"
        return f"Source: {domain} ({label})"

    return re.sub(r"Source:\s*([\w\.\-/]+)\s*\((?:PRIMARY|SECONDARY)\)", correct_label, text)


def fix_invalid_verified_tags(text: str, official_domains: List[str]) -> str:
    """
    Code-level guardrail for the OUTPUT side: the input-side fix above only
    stops the LLM from trusting a mislabeled incoming tag - it does nothing
    if the LLM independently invents its own '[VERIFIED: <domain>]' tag for
    a domain that was never on this run's whitelist to begin with. This scans
    the LLM's finished output and force-downgrades any such tag to [LIKELY].
    """
    def correct_tag(match):
        domain = match.group(1)
        if any(base_domain(domain) == base_domain(d) for d in official_domains):
            return match.group(0)  # legitimate, leave untouched
        return f"[LIKELY: {domain}, unconfirmed by official source]"

    return re.sub(r"\[VERIFIED:\s*([\w\.\-/]+)[^\]]*\]", correct_tag, text)


# --- Shared state passed between every node ---
class AgentState(TypedDict):
    goal: str                      # original user goal
    sub_questions: List[str]       # planner's output
    official_domains: List[str]    # resolved per-request whitelist (NOT hardcoded)
    findings: List[str]            # researcher's output (one per sub-question)
    verified: str                  # verifier's output (flags contradictions/confidence)
    report: str                    # writer's final output


# --- Node 1: Planner ---
@traceable(name="planner_node")
def planner_node(state: AgentState) -> AgentState:
    prompt = (
        f"Break this research goal into 2-4 specific, searchable sub-questions.\n"
        f"Goal: {state['goal']}\n"
        f"Return ONLY the sub-questions, one per line, no numbering."
    )
    response = call_llm(prompt)
    sub_questions = [q.strip() for q in response.content.split("\n") if q.strip()]

    if len(sub_questions) > MAX_SUB_QUESTIONS:
        print(f"  [GUARDRAIL] Planner generated {len(sub_questions)} sub-questions, "
              f"capping at {MAX_SUB_QUESTIONS}.")
        sub_questions = sub_questions[:MAX_SUB_QUESTIONS]

    print(f"\n[PLANNER] Sub-questions:\n" + "\n".join(f"  - {q}" for q in sub_questions))

    official_domains = resolve_official_domains(state["goal"])
    return {**state, "sub_questions": sub_questions, "official_domains": official_domains}


# --- Node 2: Researcher (processes each sub-question in parallel) ---
@traceable(name="research_one_question")
def research_one_question(q: str, official_domains: List[str]) -> str:
    """Everything needed to answer ONE sub-question: search + extract + source-tag.
    Pulled out as its own function so sub-questions can run concurrently -
    they don't depend on each other, so there's no reason to wait in line."""
    raw_results = call_search_with_timeout(q)
    # Tavily returns {"results": [{"url": ..., "content": ...}, ...]}
    result_list = raw_results.get("results", []) if isinstance(raw_results, dict) else []

    # Build numbered references with the REAL url extracted in code -
    # the LLM only ever sees an index number, never types a domain itself.
    ref_domains = {}
    ref_text_blocks = []
    for i, r in enumerate(result_list, start=1):
        domain = get_domain(r.get("url", ""))
        ref_domains[i] = domain
        ref_text_blocks.append(f"[REF:{i}] (from {domain}): {r.get('content', '')}")
    references_text = "\n\n".join(ref_text_blocks) if ref_text_blocks else "No search results found."

    summary_prompt = (
        f"Question: {q}\n\nNumbered sources:\n{references_text}\n\n"
        f"Extract the specific answer. For EVERY factual claim (price, rating, feature):\n"
        f"1. Tag it with which numbered source it came from, like '[REF:2]'. "
        f"Do NOT write a domain name yourself - only use [REF:n], the domain will be "
        f"attached automatically afterward.\n"
        f"2. If a claim has no clear source among the numbered results, tag it [REF:NONE].\n"
        f"3. Only call it a CONTRADICTION if sources disagree on the SAME thing (same plan, "
        f"same billing frequency). Monthly price vs annual price for the same plan is NOT "
        f"a contradiction — list both normally.\n"
        f"4. If sources genuinely disagree on the same fact, list ALL values explicitly:\n"
        f"   'CONTRADICTION on [fact]: $X [REF:1], $Y [REF:3]'\n"
        f"Never merge disagreeing sources into a single silent answer."
    )
    answer = call_llm(summary_prompt).content

    # Code-level substitution: [REF:n] -> the REAL domain + label, computed in code.
    # The LLM never typed a domain, so it can never misattribute or invent one.
    def substitute_ref(match):
        ref_id = match.group(1)
        if ref_id == "NONE":
            return "Source: NONE"
        domain = ref_domains.get(int(ref_id), "unknown")
        return f"Source: {domain} ({label_domain(domain, official_domains)})"

    answer = re.sub(r"\[REF:(\d+|NONE)\]", substitute_ref, answer)
    return f"Q: {q}\nA: {answer}"


@traceable(name="researcher_node")
def researcher_node(state: AgentState) -> AgentState:
    sub_questions = state["sub_questions"]
    official_domains = state["official_domains"]
    start = time.time()

    # Run all sub-questions concurrently instead of one at a time in a loop.
    # They're independent, so there's no reason to wait for #1 before starting #2.
    with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
        research_with_domains = partial(research_one_question, official_domains=official_domains)
        # .map preserves order, so findings[i] still matches sub_questions[i]
        findings = list(executor.map(research_with_domains, sub_questions))

    elapsed = time.time() - start
    print(f"\n[RESEARCHER] Collected {len(findings)} findings in {elapsed:.1f}s (parallel)")
    return {**state, "findings": findings}


# --- Node 3: Verifier (rewrites findings with inline confidence tags) ---
@traceable(name="verifier_node")
def verifier_node(state: AgentState) -> AgentState:
    official_domains = state["official_domains"]
    domains_str = ", ".join(official_domains) if official_domains else "(none resolved)"
    prompt = (
        f"You will rewrite each finding below, attaching an inline confidence tag "
        f"to EVERY specific factual claim. The ONLY domains that may ever be tagged VERIFIED "
        f"are: {domains_str} — these were resolved as the official sites for the companies in "
        f"this specific goal. Any other domain (review sites, aggregators, blogs) is SECONDARY "
        f"no matter what label it arrived with. "
        f"Double-check every '(PRIMARY)' label against this rule before trusting it; "
        f"the researcher step sometimes mislabels. "
        f"Apply these MECHANICAL rules in order, never judge by how confident the sentence sounds:\n\n"
        "1. If marked CONTRADICTION with a PRIMARY source among the disagreeing values -> "
        "state the PRIMARY source's value as the fact, tag it "
        "[VERIFIED: <primary domain> - note: <secondary domain> reports a different value].\n"
        "2. If marked CONTRADICTION with NO primary source (only secondary sources disagree) -> "
        "tag it [UNCERTAIN: sources disagree - <domain A> says $X, <domain B> says $Y].\n"
        "3. If a single value has 'Source: <domain> (PRIMARY)' -> tag it [VERIFIED: <domain>].\n"
        "4. If a single value has 'Source: <domain> (SECONDARY)' with no contradiction -> "
        "tag it [LIKELY: <domain>, unconfirmed by official source].\n"
        "5. If 'Source: NONE' -> tag it [UNCERTAIN: no source cited].\n\n"
        "Example:\n"
        "  'The Business Plan costs $18/user/month [VERIFIED: notion.so - note: sync2sheets.com "
        "reports $20].'\n"
        "  'The Pro Plan costs $19/seat/month [LIKELY: softreviewlab.com, unconfirmed by official source].'\n\n"
        "Rewrite the findings below in place with tags attached directly to each claim:\n\n"
        + fix_mislabeled_sources("\n\n".join(state["findings"]), official_domains)
    )
    verification = call_llm(prompt).content
    verification = fix_invalid_verified_tags(verification, official_domains)  # <-- output-side correction
    print(f"\n[VERIFIER] Annotated findings with confidence tags")
    return {**state, "verified": verification}


# --- Node 4: Writer (synthesizes final report) ---
@traceable(name="writer_node")
def writer_node(state: AgentState) -> AgentState:
    prompt = (
        f"Write a clear, structured comparison report answering: {state['goal']}\n\n"
        f"Use ONLY these confidence-tagged findings as your source of truth "
        f"(ignore any raw findings elsewhere):\n{state['verified']}\n\n"
        f"CRITICAL RULES:\n"
        f"- Every [VERIFIED] claim: state it plainly as fact. If it includes a 'note' about a "
        f"secondary source disagreeing, mention that briefly in the same sentence.\n"
        f"- Every [LIKELY] claim: state it but note it comes from a third-party source, not "
        f"confirmed on the company's own site (e.g. 'according to softreviewlab.com').\n"
        f"- Every [UNCERTAIN] claim: you MUST hedge it directly in the sentence "
        f"where you state it (e.g. 'reportedly', 'sources disagree', 'not independently confirmed') "
        f"— the reader must see the doubt at the exact point the claim is made, not in a separate section.\n"
        f"- Do NOT create a separate 'Verification Notes' or 'Confidence Note' section at the end. "
        f"All confidence signaling happens inline, where the claim lives.\n"
    )
    report = call_llm(prompt).content
    report = fix_invalid_verified_tags(report, state["official_domains"])  # <-- same guardrail, applied again here
    print(f"\n[WRITER] Final report generated")
    return {**state, "report": report}


# --- Build the graph ---
graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("verifier", verifier_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "verifier")
graph.add_edge("verifier", "writer")
graph.add_edge("writer", END)

app = graph.compile()


@traceable(name="competitive_intel_pipeline")
def run_pipeline(goal: str):
    return app.invoke({
        "goal": goal, "sub_questions": [], "official_domains": [],
        "findings": [], "verified": "", "report": ""
    })


if __name__ == "__main__":
    goal = "Compare Notion's and Monday.com's pricing and positioning for small teams"

    try:
        result = run_pipeline(goal)
        print("\n--- FINAL REPORT ---\n")
        print(result["report"])
        print(f"\n[COST] LLM: {cost_tracker.call_count} calls, ${cost_tracker.total_cost:.4f} "
              f"(budget: ${MAX_BUDGET_USD:.2f})")
        print(f"[COST] Tavily: {tavily_tracker.call_count} calls, "
              f"{tavily_tracker.credits_used} credits used "
              f"(free tier: 1,000/month — if on PAYG this would cost "
              f"${tavily_tracker.estimated_payg_cost:.4f})")
    except BudgetExceededError as e:
        print(f"\n[STOPPED] {e}")
        print("The run was halted before finishing to protect your API spend. "
              "Raise MAX_BUDGET_USD if this goal genuinely needs more calls.")