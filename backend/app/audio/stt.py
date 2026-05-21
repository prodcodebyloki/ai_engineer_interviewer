import tempfile
import wave
from openai import OpenAI

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def transcribe(audio_path: str) -> str:
    """Transcribe audio file to text using Whisper."""
    with open(audio_path, "rb") as f:
        result = _get_client().audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            response_format="text",
        )
    return result.strip()


def record_audio(duration_seconds: int = 60, sample_rate: int = 16000) -> str:
    """Record audio from mic. Returns path to WAV file. Stops at silence or max duration."""
    import sounddevice as sd
    import numpy as np

    frames = []
    silence_threshold = 0.01
    silence_frames = 0
    max_silence_frames = sample_rate * 3  # 3 seconds silence = done

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())
        rms = np.sqrt(np.mean(indata ** 2))
        nonlocal silence_frames
        if rms < silence_threshold:
            silence_frames += frame_count
        else:
            silence_frames = 0

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        import time
        start = time.time()
        while time.time() - start < duration_seconds:
            time.sleep(0.1)
            if silence_frames >= max_silence_frames and len(frames) > sample_rate * 2:
                break

    audio_data = np.concatenate(frames, axis=0)
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name

    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return output_path
