import os
import io
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

st.set_page_config(
    page_title="AI Engineering Interviewer",
    page_icon="🤖",
    layout="centered",
)

from langsmith import traceable, trace
from backend.app.graph.state import initial_state
from backend.app.graph.nodes.ask_question import ask_question_node
from backend.app.graph.nodes.evaluate import evaluation_node
from backend.app.graph.nodes.report import report_node
from backend.app.audio.tts import speak
from backend.app.audio.stt import transcribe

TOPICS = [
    "RAG and Vector Databases",
    "LLM Fine-tuning",
    "AI Agents and Tool Use",
    "Embeddings and Similarity Search",
    "LLM Evaluation and Observability",
    "Prompt Engineering",
    "Multi-modal AI",
]

MAX_TURNS = 8


def init_session():
    if "state" not in st.session_state:
        st.session_state.state = None
    if "phase" not in st.session_state:
        st.session_state.phase = "setup"


def play_audio(path: str):
    with open(path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)


def render_transcript(transcript: list[dict]):
    for entry in transcript:
        role = entry["role"]
        content = entry["content"]
        if role == "interviewer":
            with st.chat_message("assistant"):
                st.write(content)
        elif role == "candidate":
            with st.chat_message("user"):
                st.write(content)


def apply_updates(state: dict, updates: dict):
    for key, value in updates.items():
        if key in ("transcript", "evaluation_scores") and isinstance(value, list):
            state[key] = state.get(key, []) + value
        else:
            state[key] = value


def render_score_badge(score: int) -> str:
    if score >= 8:
        return f"🟢 {score}/10"
    elif score >= 5:
        return f"🟡 {score}/10"
    else:
        return f"🔴 {score}/10"


def _evaluate_and_advance(state: dict, answer_text: str):
    try:
        with st.spinner("Evaluating your answer..."):
            with trace(
                name="evaluate-turn",
                run_type="chain",
                metadata={
                    "candidate": state["candidate_name"],
                    "turn": state["turn_count"],
                    "difficulty": state["difficulty"],
                    "topic": state["topic"],
                },
            ):
                updates = evaluation_node(state)
                apply_updates(state, updates)
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return

    if state["turn_count"] >= MAX_TURNS:
        state["should_end"] = True
        st.session_state.state = state
        _finish_interview()
        return

    try:
        with st.spinner("Generating next question..."):
            updates = ask_question_node(state)
            apply_updates(state, updates)
            audio_path = speak(state["current_question"])
            state["audio_path"] = audio_path
    except Exception as e:
        st.error(f"Failed to generate next question: {e}")
        return

    st.session_state.state = state
    st.rerun()


def _finish_interview():
    state = st.session_state.state
    try:
        with st.spinner("Generating your interview report..."):
            updates = report_node(state)
            apply_updates(state, updates)
    except Exception as e:
        st.error(f"Report generation failed: {e}")
        return
    st.session_state.state = state
    st.session_state.phase = "report"
    st.rerun()


# ── Setup ──────────────────────────────────────────────────────────────────────
def setup_screen():
    st.title("AI Engineering Interviewer")
    st.markdown("*Adaptive voice interview · GPT-4.1 · LangGraph*")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Your name", placeholder="e.g. Alex Chen")
    with col2:
        topic = st.selectbox("Interview topic", TOPICS)

    st.caption("8 adaptive questions. Difficulty adjusts to your answers.")

    if st.button("Start Interview", type="primary", disabled=not name):
        try:
            with st.spinner("Generating first question..."):
                with trace(
                    name=f"interview-session",
                    run_type="chain",
                    metadata={"candidate": name, "topic": topic},
                ) as session_trace:
                    state = initial_state(candidate_name=name, topic=topic)
                    updates = ask_question_node(state)
                    apply_updates(state, updates)
                    audio_path = speak(state["current_question"])
                    state["audio_path"] = audio_path
                    st.session_state.trace_id = session_trace.id if hasattr(session_trace, "id") else None
        except Exception as e:
            st.error(f"Failed to start interview: {e}")
            return
        st.session_state.state = state
        st.session_state.phase = "interview"
        st.rerun()


# ── Interview ──────────────────────────────────────────────────────────────────
def interview_screen():
    state = st.session_state.state
    turn = state["turn_count"]

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"### {state['candidate_name']}")
    with col2:
        st.metric("Turn", f"{turn}/{MAX_TURNS}")
    with col3:
        icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(state["difficulty"], "")
        st.metric("Difficulty", f"{icon} {state['difficulty'].title()}")

    st.divider()

    if state["transcript"]:
        render_transcript(state["transcript"])

    st.info(f"**Question:** {state['current_question']}")

    if state.get("audio_path") and os.path.exists(state["audio_path"]):
        play_audio(state["audio_path"])

    st.markdown("---")
    st.markdown("#### Your Answer")
    tab_voice, tab_text = st.tabs(["🎤 Voice (browser mic)", "⌨️ Type"])

    with tab_voice:
        st.caption("Click the mic button below to record. Click stop when done, then submit.")
        audio_input = st.audio_input("Record your answer", key=f"audio_q{turn}")
        if audio_input is not None:
            if st.button("Submit Voice Answer", type="primary", key=f"submit_voice_{turn}"):
                with st.spinner("Transcribing..."):
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            f.write(audio_input.getvalue())
                            tmp_path = f.name
                        transcript_text = transcribe(tmp_path)
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        st.stop()
                state["transcript"].append({"role": "candidate", "content": transcript_text})
                st.success(f"Transcribed: *{transcript_text}*")
                _evaluate_and_advance(state, transcript_text)

    with tab_text:
        typed = st.text_area("Type your answer:", height=150, key=f"typed_q{turn}")
        if st.button("Submit Answer", type="primary", key=f"submit_text_{turn}"):
            if not typed.strip():
                st.warning("Answer cannot be empty.")
            else:
                state["transcript"].append({"role": "candidate", "content": typed.strip()})
                _evaluate_and_advance(state, typed.strip())

    if state["evaluation_scores"]:
        with st.expander("📊 Score history"):
            for i, ev in enumerate(state["evaluation_scores"]):
                st.write(f"Q{i+1}: {render_score_badge(ev.score)} | {ev.depth} | confidence {ev.confidence:.0%}")
                if ev.missing_topics:
                    st.caption(f"Missing: {', '.join(ev.missing_topics)}")

    if turn >= 3:
        st.divider()
        if st.button("⏹ End interview early"):
            _finish_interview()


# ── Report ─────────────────────────────────────────────────────────────────────
def report_screen():
    state = st.session_state.state

    st.title("Interview Complete")
    st.markdown(f"**{state['candidate_name']}** · {state['topic']}")
    st.divider()

    if state["evaluation_scores"]:
        avg = sum(e.score for e in state["evaluation_scores"]) / len(state["evaluation_scores"])
        c = st.columns(4)
        c[0].metric("Avg Score", f"{avg:.1f}/10")
        c[1].metric("Questions", len(state["questions_asked"]))
        c[2].metric("Final Difficulty", state["difficulty"].title())
        c[3].metric("Hallucinations", sum(1 for e in state["evaluation_scores"] if e.hallucinations_detected))
        st.divider()

    st.markdown(state["final_report"])
    st.divider()

    with st.expander("📝 Full transcript"):
        render_transcript(state["transcript"])

    with st.expander("📊 Per-question evaluations"):
        for i, ev in enumerate(state["evaluation_scores"]):
            st.markdown(f"**Q{i+1}:** {render_score_badge(ev.score)} | {ev.depth} | confidence {ev.confidence:.0%} | hallucination {'⚠️' if ev.hallucinations_detected else '✅'}")
            if ev.missing_topics:
                st.write(f"Missing: {', '.join(ev.missing_topics)}")
            st.caption(ev.feedback)
            st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New interview"):
            for key in ("state", "phase"):
                del st.session_state[key]
            st.rerun()
    with col2:
        st.download_button(
            "📥 Download report",
            data=state["final_report"],
            file_name=f"report_{state['candidate_name'].replace(' ', '_')}.md",
            mime="text/markdown",
        )


# ── Main ───────────────────────────────────────────────────────────────────────
init_session()

if st.session_state.phase == "setup":
    setup_screen()
elif st.session_state.phase == "interview":
    interview_screen()
elif st.session_state.phase == "report":
    report_screen()
