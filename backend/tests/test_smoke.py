"""Smoke tests — no API calls, no audio hardware required."""
import pytest
from backend.app.graph.state import initial_state
from backend.app.schemas.interview import EvaluationResult


def test_initial_state_shape():
    state = initial_state("Test Candidate", "RAG and Vector Databases")
    assert state["candidate_name"] == "Test Candidate"
    assert state["interview_stage"] == "intro"
    assert state["difficulty"] == "medium"
    assert state["turn_count"] == 0
    assert state["transcript"] == []


def test_evaluation_result_model():
    ev = EvaluationResult(
        score=7,
        depth="intermediate",
        missing_topics=["HNSW"],
        confidence=0.8,
        feedback="Good answer.",
        hallucinations_detected=False,
    )
    assert ev.score == 7
    assert not ev.hallucinations_detected


def test_difficulty_logic_direct():
    """Test difficulty rules without importing OpenAI client."""
    def _difficulty(scores, current):
        if len(scores) < 2:
            return current
        last_two = scores[-2:]
        if all(s >= 8 for s in last_two):
            return "hard"
        if all(s <= 4 for s in last_two):
            return "easy"
        return "medium"

    assert _difficulty([9, 9], "medium") == "hard"
    assert _difficulty([3, 3], "medium") == "easy"
    assert _difficulty([6, 7], "medium") == "medium"
    assert _difficulty([9], "medium") == "medium"  # < 2 scores, no change
