from openai import OpenAI
from langsmith import traceable
from backend.app.schemas.interview import InterviewState
from backend.app.prompts.interviewer import (
    INTERVIEWER_SYSTEM,
    NEXT_QUESTION_PROMPT,
    INTRO_PROMPT,
)

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@traceable(name="ask_question", run_type="chain")
def ask_question_node(state: InterviewState) -> dict:
    """Generate the next interview question."""
    is_intro = state["interview_stage"] == "intro"

    if is_intro:
        prompt = INTRO_PROMPT.format(
            candidate_name=state["candidate_name"],
            topic=state["topic"],
            difficulty=state["difficulty"],
        )
        content = _get_client().chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        ).choices[0].message.content
    else:
        last_entry = next(
            (t for t in reversed(state["transcript"]) if t["role"] == "candidate"),
            None,
        )
        last_answer = last_entry["content"] if last_entry else ""
        last_eval = state["evaluation_scores"][-1] if state["evaluation_scores"] else None
        eval_summary = (
            f"score={last_eval.score}, depth={last_eval.depth}, missing={last_eval.missing_topics}"
            if last_eval else "no evaluation yet"
        )

        system = INTERVIEWER_SYSTEM.format(
            difficulty=state["difficulty"],
            topic=state["topic"],
            stage=state["interview_stage"],
            questions_asked=state["questions_asked"],
        )
        prompt = NEXT_QUESTION_PROMPT.format(
            last_answer=last_answer,
            evaluation=eval_summary,
            difficulty=state["difficulty"],
            topic=state["topic"],
        )
        content = _get_client().chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        ).choices[0].message.content

    new_transcript = {"role": "interviewer", "content": content}
    return {
        "current_question": content,
        "questions_asked": state["questions_asked"] + [content],
        "transcript": [new_transcript],
        "interview_stage": "technical",
        "turn_count": state["turn_count"] + 1,
    }
