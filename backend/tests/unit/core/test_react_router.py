from core.graph import react_router, data_review_router, AgentState
from config import settings


def _state(confidence: float, react_iterations: int, stage: str = "collect") -> AgentState:
    return {
        "run_id": "r",
        "session_id": "s",
        "user_query": "q",
        "plans": [],
        "collection_manifest": {},
        "synthesis_chapters": [],
        "confidence": confidence,
        "react_iterations": react_iterations,
        "stage": stage,
        "error": None,
    }


# react_router now sends settled collection to the Gate 2 node (data_review), which
# proceeds to synthesize only after human approval. Silent backtrack retries are bounded
# by collect_max_retries so the human is never re-shown Gate 2 on transient failures.

def test_routes_to_data_review_when_confidence_above_threshold():
    threshold = settings.react.confidence_threshold
    assert react_router(_state(confidence=threshold, react_iterations=0)) == "data_review"


def test_routes_to_data_review_when_confidence_exceeds_threshold():
    assert react_router(_state(confidence=1.0, react_iterations=0)) == "data_review"


def test_routes_to_backtrack_when_confidence_below_threshold_and_retries_remain():
    assert react_router(_state(confidence=0.0, react_iterations=0)) == "backtrack"


def test_routes_to_data_review_when_silent_retries_exhausted():
    cap = settings.react.collect_max_retries
    assert react_router(_state(confidence=0.0, react_iterations=cap)) == "data_review"


def test_routes_to_backtrack_just_below_retry_cap():
    cap = settings.react.collect_max_retries
    assert react_router(_state(confidence=0.0, react_iterations=cap - 1)) == "backtrack"


def test_error_stage_routes_to_synthesize():
    assert react_router(_state(confidence=0.0, react_iterations=0, stage="error")) == "synthesize"


def test_partial_stage_routes_to_partial_brief():
    assert react_router(_state(confidence=0.0, react_iterations=0, stage="partial")) == "partial_brief"


def test_data_review_router_redirect_loops_back_to_collect():
    assert data_review_router(_state(confidence=0.0, react_iterations=0, stage="collect")) == "collect"


def test_data_review_router_approve_proceeds_to_synthesize():
    assert data_review_router(_state(confidence=1.0, react_iterations=0, stage="synthesize")) == "synthesize"
