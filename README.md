<img src="header.png" width="100%"/>

# AI Engineering Interviewer Agent

> 📺 **Tutorial on YouTube:** [ProdCode by Loki](https://www.youtube.com/@ProdCodebyLoki)

A real-time adaptive AI interviewer built with GPT-4.1, LangGraph, and Streamlit. It speaks questions aloud via OpenAI TTS, transcribes your answers through the browser mic using Whisper, evaluates technical depth with a structured scoring agent, and adjusts difficulty turn by turn — finishing with a full hiring report and roadmap.

> Built with OpenAI GPT-4.1 · LangGraph · LangSmith · Streamlit · Python 3.12

---

## How It Works

You open the app, enter your name, pick a topic (RAG, Agents, Fine-tuning, etc.) and hit Start. The AI greets you and asks the first question aloud. You answer via the browser mic or type — the answer is transcribed, scored, and the next question is generated at the right difficulty level. After 8 turns the system produces a full interview report with a hiring signal and study roadmap.

```
Setup → Ask Question (TTS) → Record Answer → Transcribe → Evaluate
   └──────────────────── loop (up to 8 turns) ────────────────────┘
                                    ↓
                            Generate Report
```

Every node is a step in a LangGraph state machine. Every LLM call is a named span in LangSmith.

---

## Agents

**Interviewer** (`graph/nodes/ask_question.py`) generates one question per turn using GPT-4.1, reading the last answer, evaluation summary, and current difficulty level. It never repeats a question and adapts its phrasing based on how the candidate is performing.

**Transcription** (`audio/stt.py`) receives raw audio bytes from the browser mic widget, saves them as a temp WAV, and sends them to `gpt-4o-mini-transcribe`. The returned text feeds directly into the evaluator.

**Evaluator** (`agents/evaluator.py`) scores each answer with structured JSON output — score 1–10, depth label, missing topics, confidence, feedback, and a hallucination flag. Uses `response_format: json_object` to guarantee parseable output and `@traceable` so every call appears as a span in LangSmith.

**Difficulty Controller** (`agents/difficulty_controller.py`) reads the last 2–3 scores and returns one word — `easy`, `medium`, or `hard`. If the last two scores are both ≥ 8 it increases difficulty; both ≤ 4 it drops it; otherwise it holds.

**Report Generator** (`agents/report_generator.py`) reads the full transcript and all evaluation scores to write a structured markdown report: overall score, technical depth, communication, strengths, gaps, hiring signal (Strong Yes / Yes / Maybe / No), and a personalised study roadmap.

---

## Orchestration

LangGraph compiles a `StateGraph` that carries `InterviewState` across all nodes. Fields like `transcript` and `evaluation_scores` accumulate each turn rather than being overwritten.

```
START → ask_question → transcribe → evaluate
                                       ├── [turn < 8] → ask_question  (loop)
                                       └── [turn ≥ 8] → report → END
```

The Streamlit frontend calls each node individually rather than using a blocking `.invoke()`, which lets the UI update after every step. The state is a plain `TypedDict` held in `st.session_state` between turns.

---

## Folder Structure

```
├── backend/app/
│   ├── agents/          difficulty_controller · evaluator · report_generator
│   ├── audio/           stt (Whisper) · tts (OpenAI TTS)
│   ├── graph/
│   │   ├── nodes/       ask_question · evaluate · report · transcribe
│   │   ├── state.py     initial state factory
│   │   └── workflow.py  compiled LangGraph
│   ├── prompts/         all LLM prompts in one file
│   └── schemas/         InterviewState · EvaluationResult
├── frontend/app.py      Streamlit UI
└── .env.example
```

---

## Setup

You need Python 3.12+, the `uv` package manager, an OpenAI API key, and a LangSmith API key (free tier works).

```bash
git clone https://github.com/prodcodebyloki/ai_engineer_interviewer.git
cd ai_engineer_interviewer
uv sync
cp .env.example .env   # then fill in your keys
```

Your `.env` should look like:

```env
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=ai-interviewer-agent
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Run with:

```bash
PYTHONPATH=. uv run streamlit run frontend/app.py
```

Opens at `http://localhost:8501`. Enter your name, pick a topic, and click **Start Interview**. Answer via the browser mic or the Type tab. After 8 questions (or an early exit) you get the full report and can download it as Markdown.

---

## LangSmith Tracing

Every session is traced end-to-end. Go to [smith.langchain.com](https://smith.langchain.com) → project `ai-interviewer-agent`. Each interview creates an `interview-session` chain span, with child spans for every `evaluate-turn`, `ask_question`, `evaluate_answer`, `compute_next_difficulty`, and `generate_report` call. You can filter by candidate, turn number, difficulty level, or hallucination flag, and compare prompt versions across sessions.

---

## Screenshots

<table>
<tr>
<td width="33%"><img src="Screenshot/image.png" width="100%"/><br/><sub>Setup — empty</sub></td>
<td width="33%"><img src="Screenshot/image2.png" width="100%"/><br/><sub>Setup — ready to start</sub></td>
<td width="33%"><img src="Screenshot/image4.png" width="100%"/><br/><sub>Turn 1 — question spoken aloud</sub></td>
</tr>
<tr>
<td width="33%"><img src="Screenshot/image3.png" width="100%"/><br/><sub>Voice recording in progress</sub></td>
<td width="33%"><img src="Screenshot/image5.png" width="100%"/><br/><sub>Turn 2 — score history visible</sub></td>
<td width="33%"><img src="Screenshot/image6.png" width="100%"/><br/><sub>Report — full transcript</sub></td>
</tr>
<tr>
<td width="33%"><img src="Screenshot/image7.png" width="100%"/><br/><sub>Per-question evaluations</sub></td>
<td width="33%"><img src="Screenshot/image8.png" width="100%"/><br/><sub>Report — summary metrics</sub></td>
<td width="33%"><img src="Screenshot/image9.png" width="100%"/><br/><sub>Strengths and gaps</sub></td>
</tr>
<tr>
<td width="33%"><img src="Screenshot/image10.png" width="100%"/><br/><sub>Hiring signal and study areas</sub></td>
<td></td>
<td></td>
</tr>
</table>

---

## Tech Stack

| | |
|---|---|
| LLM | OpenAI GPT-4.1 |
| TTS / STT | `gpt-4o-mini-tts` / `gpt-4o-mini-transcribe` |
| Orchestration | LangGraph `StateGraph` |
| Observability | LangSmith `@traceable` |
| Frontend | Streamlit |
| Validation | Pydantic v2 |
| Package manager | uv |
