"""
Phase 4: Eval harness for the Verifier node.

Instead of eyeballing full report transcripts, we test the verifier in isolation
on small, hand-written findings where WE already know the correct tag.
This turns "does it feel better?" into a measurable pass rate.
"""

import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

OFFICIAL_DOMAINS = ["notion.so", "monday.com"]


def fix_mislabeled_sources(text: str) -> str:
    """
    Code-level guardrail: find every 'Source: <domain> (PRIMARY)' tag and
    FORCE it to (SECONDARY) if the domain isn't actually on the whitelist.
    This runs before the LLM ever sees the text, so there's no label left
    to accidentally trust - the lie is corrected deterministically, not
    probabilistically.
    """
    def correct_label(match):
        domain = match.group(1)
        is_official = any(domain.startswith(d) for d in OFFICIAL_DOMAINS)
        label = "PRIMARY" if is_official else "SECONDARY"
        return f"Source: {domain} ({label})"

    return re.sub(r"Source:\s*([\w\.\-/]+)\s*\((?:PRIMARY|SECONDARY)\)", correct_label, text)


def verify_finding(finding_text: str) -> str:
    finding_text = fix_mislabeled_sources(finding_text)  # <-- correction happens here, in code
    """Same verifier logic as phase2, isolated to a single finding for testing."""
    prompt = (
        "You will rewrite the finding below, attaching an inline confidence tag "
        "to EVERY specific factual claim. The ONLY domains that may ever be tagged VERIFIED "
        "are notion.so and monday.com — if you see any other domain (vendr.com, g2.com, "
        "any aggregator or review site), it is SECONDARY no matter what label it arrived with. "
        "Double-check every '(PRIMARY)' label against this rule before trusting it. "
        "Apply these MECHANICAL rules in order, never judge by how confident the sentence sounds:\n\n"
        "1. If marked CONTRADICTION with a PRIMARY source among the disagreeing values -> "
        "state the PRIMARY source's value as the fact, tag it "
        "[VERIFIED: <primary domain> - note: <secondary domain> reports a different value].\n"
        "2. If marked CONTRADICTION with NO primary source (only secondary sources disagree) -> "
        "tag it [UNCERTAIN: sources disagree - <domain A> says $X, <domain B> says $Y].\n"
        "3. If a single value has 'Source: <domain> (PRIMARY)' -> tag it [VERIFIED: <domain>].\n"
        "4. If a single value has 'Source: <domain> (SECONDARY)' with no contradiction -> "
        "tag it [LIKELY: <domain>, unconfirmed by official source].\n"
        "5. If 'Source: NONE' -> tag it [UNCERTAIN: no source cited].\n\n"
        "Rewrite this finding in place with tags attached directly to each claim:\n\n"
        + finding_text
    )
    return llm.invoke(prompt).content


# --- Ground-truth test cases ---
# Each case has a "check" function: given the verifier's output text, return True/False.
# Simple, literal string checks on purpose — the scoring itself shouldn't be fooled
# the same way the verifier can be.

def verified_with_domain(domain):
    return lambda out: "[VERIFIED" in out and domain in out

def not_verified(out):
    return "[VERIFIED" not in out

TEST_CASES = [
    # --- Original 6 (easy, isolated) ---
    {
        "name": "Official source should be VERIFIED",
        "input": "Notion Free Plan price: $0/member/month. Source: notion.so (PRIMARY).",
        "check": verified_with_domain("notion.so"),
    },
    {
        "name": "Secondary source alone should NOT be VERIFIED",
        "input": "Notion Business Plan price: $18/member/month. Source: vendr.com (SECONDARY).",
        "check": not_verified,
    },
    {
        "name": "Review site rating about a company is NOT that company's own source",
        "input": "Monday.com rating: 4.7/5. Source: g2.com (SECONDARY).",
        "check": not_verified,
    },
    {
        "name": "Contradiction with NO primary source should be UNCERTAIN, not VERIFIED",
        "input": (
            "CONTRADICTION on Business Plan price: $18 per vendr.com (SECONDARY), "
            "$20 per sync2sheets.com (SECONDARY)."
        ),
        "check": not_verified,
    },
    {
        "name": "Contradiction WITH a primary source should prefer the primary value",
        "input": (
            "CONTRADICTION on Plus Plan price: $10 per notion.so (PRIMARY), "
            "$12 per vendr.com (SECONDARY)."
        ),
        "check": verified_with_domain("notion.so"),
    },
    {
        "name": "No source at all should be UNCERTAIN",
        "input": "Notion has a 7-day page history on the Free plan. Source: NONE.",
        "check": not_verified,
    },

    # --- Harder cases: messier, closer to real pipeline output ---
    {
        "name": "HARD: monthly vs annual price is NOT a contradiction",
        "input": (
            "Notion Plus Plan price: $10/member/month (billed annually) or "
            "$12/member/month (billed monthly). Source: notion.so (PRIMARY)."
        ),
        # Should be VERIFIED (real primary source), and must NOT say sources disagree
        "check": lambda out: verified_with_domain("notion.so")(out)
                              and "disagree" not in out.lower()
                              and "CONTRADICTION" not in out,
    },
    {
        "name": "HARD: researcher mislabels a non-whitelisted domain as PRIMARY",
        "input": "Notion Business Plan price: $18/member/month. Source: vendr.com (PRIMARY).",
        # The researcher got it wrong — verifier must catch it anyway, domain isn't notion.so/monday.com
        "check": not_verified,
    },
    {
        "name": "HARD: multiple facts in one finding, mixed trust levels",
        "input": (
            "Notion Free Plan price: $0/member/month. Source: notion.so (PRIMARY). "
            "Notion has a 7-day page history. Source: NONE."
        ),
        # Must correctly split them: one VERIFIED, one UNCERTAIN, not blend into one tag
        "check": lambda out: verified_with_domain("notion.so")(out) and "UNCERTAIN" in out,
    },
    {
        "name": "HARD: subdomain / URL path of an official site should still count as PRIMARY",
        "input": "Monday.com Pro Plan price: $19/seat/month. Source: monday.com/pricing (PRIMARY).",
        "check": verified_with_domain("monday.com"),
    },
]


def run_eval():
    passed = 0
    results = []

    for case in TEST_CASES:
        output = verify_finding(case["input"])
        ok = case["check"](output)
        results.append((case["name"], ok, output.strip()))
        if ok:
            passed += 1

    print(f"\n{'='*60}")
    print(f"VERIFIER EVAL RESULTS: {passed}/{len(TEST_CASES)} passed ({passed/len(TEST_CASES)*100:.0f}%)")
    print(f"{'='*60}\n")

    for name, ok, output in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            print(f"   Output: {output[:200]}")
        print()


if __name__ == "__main__":
    run_eval()