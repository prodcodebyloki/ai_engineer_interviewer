import json
from openai import OpenAI
from langsmith import traceable
from backend.app.prompts.interviewer import EVALUATION_PROMPT
from backend.app.schemas.interview import EvaluationResult

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@traceable(name="evaluate_answer", run_type="llm")
def evaluate_answer(question: str, answer: str) -> EvaluationResult:
    """Evaluate candidate answer. Returns structured EvaluationResult."""
    prompt = EVALUATION_PROMPT.format(question=question, answer=answer)

    response = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    data = json.loads(response.choices[0].message.content)
    return EvaluationResult(**data)
