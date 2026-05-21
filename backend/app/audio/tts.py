import tempfile
from openai import OpenAI

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def speak(text: str, voice: str = "alloy") -> str:
    """Convert text to speech. Returns path to audio file."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name

    response = _get_client().audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    )
    response.stream_to_file(output_path)
    return output_path
