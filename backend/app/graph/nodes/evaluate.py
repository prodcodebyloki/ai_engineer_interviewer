from backend.app.schemas.interview import InterviewState
from backend.app.agents.evaluator import evaluate_answer
from backend.app.agents.difficulty_controller import compute_next_difficulty


def evaluation_node(state: InterviewState) -> dict:
    """Evaluate latest candidate answer, update difficulty."""
    last_candidate = next(
        (t for t in reversed(state["transcript"]) if t["role"] == "candidate"),
        None,
    )
    if not last_candidate or last_candidate["content"] == "[no audio captured]":
        return {}

    evaluation = evaluate_answer(
        question=state["current_question"],
        answer=last_candidate["content"],
    )

    new_scores = state["evaluation_scores"] + [evaluation]
    new_difficulty = compute_next_difficulty(new_scores, state["difficulty"])

    return {
        "evaluation_scores": [evaluation],
        "difficulty": new_difficulty,
    }
