from backend.app.schemas.interview import InterviewState


def initial_state(candidate_name: str, topic: str = "RAG and Vector Databases") -> InterviewState:
    return {
        "candidate_name": candidate_name,
        "transcript": [],
        "current_question": "",
        "questions_asked": [],
        "evaluation_scores": [],
        "interview_stage": "intro",
        "difficulty": "medium",
        "topic": topic,
        "turn_count": 0,
        "should_end": False,
        "final_report": "",
        "audio_path": None,
    }
