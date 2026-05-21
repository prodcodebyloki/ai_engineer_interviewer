from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel


class EvaluationResult(BaseModel):
    score: int  # 1-10
    depth: str  # "beginner" | "intermediate" | "advanced"
    missing_topics: list[str]
    confidence: float
    feedback: str
    hallucinations_detected: bool


class InterviewState(TypedDict):
    candidate_name: str
    transcript: Annotated[list[dict], "append"]
    current_question: str
    questions_asked: list[str]
    evaluation_scores: Annotated[list[EvaluationResult], "append"]
    interview_stage: str  # "intro" | "technical" | "followup" | "wrap"
    difficulty: str  # "easy" | "medium" | "hard"
    topic: str
    turn_count: int
    should_end: bool
    final_report: str
    audio_path: str | None
