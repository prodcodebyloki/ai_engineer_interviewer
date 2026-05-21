from openai import OpenAI
from langsmith import traceable
from backend.app.prompts.interviewer import REPORT_PROMPT
from backend.app.schemas.interview import InterviewState

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@traceable(name="generate_report", run_type="chain")
def generate_report(state: InterviewState) -> str:
    """Generate final interview report from complete state."""
    transcript_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in state["transcript"]
    )

    evaluations_text = "\n".join(
        f"Q{i+1}: score={e.score}, depth={e.depth}, missing={e.missing_topics}, feedback={e.feedback}"
        for i, e in enumerate(state["evaluation_scores"])
    )

    topics_covered = list({q.split("?")[0][:40] for q in state["questions_asked"]})

    prompt = REPORT_PROMPT.format(
        candidate_name=state["candidate_name"],
        transcript=transcript_text,
        evaluations=evaluations_text,
        topics=", ".join(topics_covered),
    )

    response = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content
