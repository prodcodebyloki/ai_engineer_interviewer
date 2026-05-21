from langgraph.graph import StateGraph, END
from backend.app.schemas.interview import InterviewState
from backend.app.graph.nodes.ask_question import ask_question_node
from backend.app.graph.nodes.transcribe import transcription_node
from backend.app.graph.nodes.evaluate import evaluation_node
from backend.app.graph.nodes.report import report_node

MAX_TURNS = 8


def should_end(state: InterviewState) -> str:
    if state["should_end"] or state["turn_count"] >= MAX_TURNS:
        return "report"
    return "ask_question"


def build_graph() -> StateGraph:
    graph = StateGraph(InterviewState)

    graph.add_node("ask_question", ask_question_node)
    graph.add_node("transcribe", transcription_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("ask_question")
    graph.add_edge("ask_question", "transcribe")
    graph.add_edge("transcribe", "evaluate")
    graph.add_conditional_edges("evaluate", should_end, {
        "ask_question": "ask_question",
        "report": "report",
    })
    graph.add_edge("report", END)

    return graph.compile()


interview_graph = build_graph()
