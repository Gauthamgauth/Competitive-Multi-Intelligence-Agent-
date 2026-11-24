"""
FastAPI wrapper around the competitive intelligence pipeline.
Run with: uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Note: phase2_multi_agent.py's compiled graph is also named "app" internally,
# so we only import what we actually need by name - run_pipeline already wraps
# that graph's .invoke() call, so we never need the name "app" from that file.
from phase2_multi_agent import (
    run_pipeline,
    cost_tracker,
    tavily_tracker,
    BudgetExceededError,
    MAX_BUDGET_USD,
)

app = FastAPI(
    title="Autonomous Competitive Intelligence Agent",
    description="Give it a research goal, get back a sourced, confidence-tagged comparison report.",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    goal: str = Field(
        ...,
        min_length=10,
        max_length=300,
        examples=["Compare Notion's and Monday.com's pricing and positioning for small teams"],
    )


class ResearchResponse(BaseModel):
    report: str
    llm_calls: int
    llm_cost_usd: float
    tavily_calls: int
    tavily_credits_used: int
    tavily_estimated_payg_cost_usd: float


@app.get("/health")
def health():
    """Simple liveness check - confirms the API is up without running the full pipeline."""
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    """
    Runs the full planner -> researcher -> verifier -> writer pipeline on the
    given goal and returns the finished report plus real cost numbers for that run.
    """
    # Each request starts with a clean budget - otherwise MAX_BUDGET_USD would
    # apply across the server's entire lifetime instead of per request, and
    # every request after the first few would start dangerously close to the cap.
    cost_tracker.reset()
    tavily_tracker.reset()

    try:
        result = run_pipeline(request.goal)
    except BudgetExceededError as e:
        # 402 Payment Required is the closest standard HTTP status for "this
        # would cost too much to complete" - a real, honest signal to the caller.
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    return ResearchResponse(
        report=result["report"],
        llm_calls=cost_tracker.call_count,
        llm_cost_usd=round(cost_tracker.total_cost, 4),
        tavily_calls=tavily_tracker.call_count,
        tavily_credits_used=tavily_tracker.credits_used,
        tavily_estimated_payg_cost_usd=round(tavily_tracker.estimated_payg_cost, 4),
    )