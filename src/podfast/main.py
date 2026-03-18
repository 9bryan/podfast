#!/usr/bin/env python
import json
import math
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from pydantic import BaseModel

from crewai.flow import Flow, listen, start
from openai import OpenAI

from podfast.crews.summarization_crew.summarization_crew import SummarizationCrew


WHISPER_MAX_BYTES = 24 * 1024 * 1024  # 24 MB (1 MB under the 25 MB API limit)


def _nearest_mp3_sync(data: bytes, target: int, window: int = 4096) -> int:
    """Return the offset of the nearest MP3 frame sync word (0xFF 0xEx) to target.

    Searches forward first, then backward within window bytes. Falls back to
    the exact target offset if no sync word is found — Whisper tolerates a few
    corrupt frames at the start of a chunk gracefully.
    """
    end = min(len(data) - 1, target + window)
    for i in range(target, end):
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            return i
    start = max(0, target - window)
    for i in range(target - 1, start, -1):
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            return i
    return target


def _chunk_audio(file_path: str) -> list[str]:
    """Split an audio file into chunks under WHISPER_MAX_BYTES using pure Python.

    Reads raw bytes and snaps each split point to the nearest MP3 frame sync
    word so chunks begin on clean frame boundaries. No external dependencies.
    Returns a list of temp file paths (or [file_path] if no split is needed).
    """
    file_size = os.path.getsize(file_path)
    if file_size <= WHISPER_MAX_BYTES:
        return [file_path]

    num_chunks = math.ceil(file_size / WHISPER_MAX_BYTES)
    print(f"Audio is {file_size / 1024 / 1024:.1f} MB — splitting into {num_chunks} chunks")

    with open(file_path, "rb") as f:
        data = f.read()

    suffix = Path(file_path).suffix or ".mp3"
    target_chunk_size = len(data) // num_chunks
    chunk_paths = []
    offset = 0

    for i in range(num_chunks):
        if i == num_chunks - 1:
            end = len(data)
        else:
            end = _nearest_mp3_sync(data, offset + target_chunk_size)

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(data[offset:end])
        tmp.close()
        chunk_paths.append(tmp.name)
        offset = end

    return chunk_paths


def _split_text(text: str, max_chars: int = 4096) -> list[str]:
    """Split text into chunks no larger than max_chars, breaking on sentence boundaries."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        # A single sentence longer than max_chars must be hard-split
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
            continue
        if len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        chunks.append(current.strip())
    return chunks


class PodcastState(BaseModel):
    podcast_url: str = ""
    audio_file_path: str = ""
    transcription: str = ""
    summary: str = ""
    output_audio_path: str = ""


class PodcastFlow(Flow[PodcastState]):

    @start()
    def download_audio(self, crewai_trigger_payload: dict = None):
        """Download the podcast audio file from the provided URL."""
        if crewai_trigger_payload:
            self.state.podcast_url = crewai_trigger_payload.get(
                "podcast_url", self.state.podcast_url
            )

        if not self.state.podcast_url:
            raise ValueError("podcast_url must be provided in inputs or trigger payload")

        print(f"Downloading audio from: {self.state.podcast_url}")

        # Preserve the original file extension so Whisper can detect the format
        url_path = Path(self.state.podcast_url.split("?")[0])
        suffix = url_path.suffix if url_path.suffix else ".mp3"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()

        urllib.request.urlretrieve(self.state.podcast_url, tmp.name)
        self.state.audio_file_path = tmp.name
        print(f"Audio saved to: {self.state.audio_file_path}")

    @listen(download_audio)
    def transcribe_audio(self):
        """Transcribe the downloaded audio using OpenAI Whisper.

        Splits files larger than 25 MB into chunks and concatenates the results.
        """
        print("Transcribing audio with OpenAI Whisper (whisper-1)...")
        client = OpenAI()

        chunks = _chunk_audio(self.state.audio_file_path)
        parts = []
        for i, chunk_path in enumerate(chunks, 1):
            print(f"  transcribing chunk {i}/{len(chunks)}...")
            with open(chunk_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                )
            parts.append(result)
            if chunk_path != self.state.audio_file_path:
                os.unlink(chunk_path)

        self.state.transcription = " ".join(parts)
        print(f"Transcription complete — {len(self.state.transcription)} characters")

    @listen(transcribe_audio)
    def summarize_transcript(self):
        """Summarize the transcription using the SummarizationCrew."""
        print("Summarizing transcript with SummarizationCrew...")

        result = SummarizationCrew().crew().kickoff(
            inputs={"transcription": self.state.transcription}
        )
        self.state.summary = result.raw
        print(f"Summary complete — {len(self.state.summary)} characters")

    @listen(summarize_transcript)
    def generate_summary_audio(self):
        """Convert the summary to audio using OpenAI TTS.

        Splits text on sentence boundaries into chunks of at most 4096 characters
        and concatenates the raw MP3 bytes so the output is a single file.
        """
        print("Generating summary audio with OpenAI TTS (tts-1)...")
        client = OpenAI()

        chunks = _split_text(self.state.summary, max_chars=4096)
        print(f"TTS: {len(chunks)} chunk(s) to synthesize")

        output_path = os.path.join(os.getcwd(), "podcast_summary.mp3")
        with open(output_path, "wb") as out:
            for i, chunk in enumerate(chunks, 1):
                print(f"  synthesizing chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=chunk,
                )
                out.write(response.content)

        self.state.output_audio_path = output_path
        print(f"Summary audio saved to: {self.state.output_audio_path}")

        # Clean up the downloaded source audio
        try:
            os.unlink(self.state.audio_file_path)
        except OSError:
            pass


def kickoff():
    """Run the podcast flow.

    URL resolution order:
      1. PODCAST_URL environment variable  (works with `crewai run`)
      2. First positional CLI argument      (works with `uv run kickoff <url>`)
    """
    url = os.environ.get("PODCAST_URL") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not url:
        print("Error: podcast URL required.")
        print("  Set PODCAST_URL env var:        PODCAST_URL=https://... crewai run")
        print("  Or pass as CLI argument:         uv run kickoff https://...")
        sys.exit(1)

    flow = PodcastFlow()
    flow.kickoff(inputs={"podcast_url": url})


def plot():
    flow = PodcastFlow()
    flow.plot()


def run_with_trigger():
    """Run the flow with a JSON trigger payload containing 'podcast_url'."""
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Pass JSON as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    flow = PodcastFlow()
    result = flow.kickoff({"crewai_trigger_payload": trigger_payload})
    return result


if __name__ == "__main__":
    kickoff()
