import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile
import json

from backend.app.graph.state import initial_state
from backend.app.graph.workflow import interview_graph
from backend.app.audio.tts import speak
from backend.app.audio.stt import transcribe
from backend.app.agents.evaluator import evaluate_answer
from backend.app.schemas.interview import InterviewState

app = FastAPI(title="AI Interviewer Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (replace with Redis for production)
sessions: dict[str, InterviewState] = {}


@app.post("/session/start")
async def start_session(candidate_name: str = Form(...), topic: str = Form("RAG and Vector Databases")):
    state = initial_state(candidate_name=candidate_name, topic=topic)
    # Run ask_question node to get first question
    result = interview_graph.invoke(state, {"configurable": {"thread_id": candidate_name}})
    session_id = candidate_name.lower().replace(" ", "_")
    sessions[session_id] = result

    # Generate TTS for first question
    audio_path = speak(result["current_question"])
    return {
        "session_id": session_id,
        "question": result["current_question"],
        "turn": result["turn_count"],
        "difficulty": result["difficulty"],
        "audio_url": f"/audio/{session_id}",
        "audio_path": audio_path,
    }


@app.post("/session/{session_id}/answer")
async def submit_answer(session_id: str, audio: UploadFile = File(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id]

    # Save uploaded audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(await audio.read())
        audio_path = f.name

    # Transcribe
    transcript_text = transcribe(audio_path)

    # Update state with audio path and transcription
    state["audio_path"] = audio_path
    state["transcript"].append({"role": "candidate", "content": transcript_text})

    # Evaluate the answer
    evaluation = evaluate_answer(
        question=state["current_question"],
        answer=transcript_text,
    )
    state["evaluation_scores"].append(evaluation)

    # Check if we should end
    if state["turn_count"] >= 8:
        state["should_end"] = True

    sessions[session_id] = state

    return {
        "transcript": transcript_text,
        "evaluation": evaluation.model_dump(),
        "turn": state["turn_count"],
        "should_end": state["should_end"],
    }


@app.post("/session/{session_id}/next")
async def next_question(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id]

    if state["should_end"] or state["turn_count"] >= 8:
        return {"done": True, "message": "Interview complete. Call /session/{id}/report"}

    from backend.app.graph.nodes.ask_question import ask_question_node
    updates = ask_question_node(state)
    state.update(updates)
    sessions[session_id] = state

    audio_path = speak(state["current_question"])
    return {
        "question": state["current_question"],
        "turn": state["turn_count"],
        "difficulty": state["difficulty"],
        "audio_path": audio_path,
        "done": False,
    }


@app.get("/session/{session_id}/report")
async def get_report(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id]
    from backend.app.graph.nodes.report import report_node
    updates = report_node(state)
    state.update(updates)
    sessions[session_id] = state

    return {"report": state["final_report"]}


@app.get("/session/{session_id}/transcript")
async def get_transcript(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"transcript": sessions[session_id]["transcript"]}
