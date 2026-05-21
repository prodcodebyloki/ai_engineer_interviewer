from backend.app.schemas.interview import InterviewState
from backend.app.audio.stt import transcribe


def transcription_node(state: InterviewState) -> dict:
    """Transcribe recorded audio to text and add to transcript."""
    audio_path = state.get("audio_path")
    if not audio_path:
        return {"transcript": [{"role": "candidate", "content": "[no audio captured]"}]}

    text = transcribe(audio_path)
    return {
        "transcript": [{"role": "candidate", "content": text}],
    }
