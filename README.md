# Podfast

Podfast downloads a podcast episode, transcribes it, summarizes it with an AI agent, and converts the summary back to audio — giving you a concise, listenable version of any podcast.

## How it works

1. **Download** — fetches the podcast audio from a URL
2. **Transcribe** — sends audio to OpenAI Whisper (handles files >25MB by chunking on MP3 frame boundaries)
3. **Summarize** — a CrewAI agent (GPT-4o) produces a 250–400 word conversational summary optimized for speech
4. **Text-to-speech** — OpenAI TTS converts the summary to `podcast_summary.mp3`

## Requirements

- Python 3.10–3.13
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

## Setup

```bash
pip install uv
cd podfast
uv sync
```

Copy `.env.example` to `.env` and add your key:

```
OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Set the podcast URL and run
PODCAST_URL=https://example.com/episode.mp3 crewai run
```

Output is saved to `podcast_summary.mp3` in the project root.

### Visualize the flow

```bash
crewai plot
```

## Project structure

```
src/podfast/
├── main.py                          # Flow orchestration (4 steps)
└── crews/summarization_crew/
    ├── summarization_crew.py        # CrewAI crew & agent definition
    └── config/
        ├── agents.yaml              # Agent role, goal, backstory
        └── tasks.yaml               # Task prompt and output spec
```

## Configuration

- **Agent** — edit `src/podfast/crews/summarization_crew/config/agents.yaml`
- **Task prompt** — edit `src/podfast/crews/summarization_crew/config/tasks.yaml`
- **Flow logic** — edit `src/podfast/main.py`
