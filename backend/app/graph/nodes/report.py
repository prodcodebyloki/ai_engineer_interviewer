from backend.app.schemas.interview import InterviewState
from backend.app.agents.report_generator import generate_report


def report_node(state: InterviewState) -> dict:
    """Generate final interview report."""
    report = generate_report(state)
    return {
        "final_report": report,
        "interview_stage": "complete",
    }
