from openai import OpenAI
from langsmith import traceable
from backend.app.prompts.interviewer import DIFFICULTY_PROMPT
from backend.app.schemas.interview import EvaluationResult

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@traceable(name="compute_next_difficulty", run_type="chain")
def compute_next_difficulty(
    scores: list[EvaluationResult],
    current_difficulty: str,
) -> str:
    """Determine next difficulty based on recent scores."""
    if len(scores) < 2:
        return current_difficulty

    scores_summary = "\n".join(
        f"Q{i+1}: score={s.score}, depth={s.depth}" for i, s in enumerate(scores[-3:])
    )

    prompt = DIFFICULTY_PROMPT.format(
        scores=scores_summary,
        current_difficulty=current_difficulty,
    )

    response = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )

    result = response.choices[0].message.content.strip().lower()
    if result in ("easy", "medium", "hard"):
        return result
    return current_difficulty
