---
slug: video-search
title: "Video Search & Summarization"
time: "30 min"
color: orange
desc: "Transcribe and analyze video with Whisper + VLMs"
tags: [video, whisper]
spark: "Video Search & Summarization"
category: applications
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Video search and summarization pipelines work in two stages: first, transcribe audio to text; then, use an LLM to analyze the text. Whisper is OpenAI's speech recognition model trained on 680,000 hours of multilingual audio. mlx-whisper is a port of Whisper to Apple's MLX framework, which runs natively on the Metal GPU in Apple Silicon Macs — achieving real-time or faster transcription where the same model on a laptop CPU would take 5-10x longer.

The resulting transcript is a searchable, queryable document. A 60-minute lecture becomes a text file you can summarize in seconds, search for any topic, or ask questions about — all with a local LLM, nothing sent to the cloud.

## What you'll accomplish

A pipeline that takes any local video file, transcribes it with word-level timestamps using mlx-whisper (large-v3-turbo model), saves a structured transcript, and uses Ollama to summarize key points, answer questions about the video content, and generate a chapter table of contents with timestamps.

## What to know before starting

- **Whisper**: OpenAI's multilingual speech recognition model. `whisper-large-v3-turbo` is a distilled version trained to be 4x faster than the full large model with minimal quality loss — it's the right choice for most transcription tasks.
- **Word timestamps**: Whisper can output a start and end time for every word in the transcript. This enables jumping to specific moments: "the speaker mentioned neural networks at 4:32".
- **Faster than realtime**: A 60-minute video transcribed in under 60 minutes = realtime. mlx-whisper achieves ~10x realtime on M2+ (a 60-minute video transcribed in ~6 minutes). On CPU, the same model takes ~60+ minutes.
- **VAD (Voice Activity Detection)**: Skips silence in the audio, speeding up processing and preventing Whisper from hallucinating text over silent sections.
- **Context window limits**: Long transcripts may exceed the LLM's context window. A 2-hour lecture transcript (~15,000 words) exceeds the context of smaller Ollama models — you'll need to chunk or truncate.
- **ffmpeg**: mlx-whisper delegates audio decoding to ffmpeg. It handles video-to-audio extraction automatically, so you can pass .mp4, .mov, .mkv, etc. directly.

## Prerequisites

- macOS 14.0+, Apple Silicon (M1 or later)
- Python 3.10+
- Ollama running with `qwen2.5:7b` pulled
- ffmpeg: `brew install ffmpeg`

## Time & risk

- **Duration:** 30 minutes
- **Risk level:** Low — read-only operations on your video file, no system changes

<!-- tab: Setup -->
## Step 1: Install mlx-whisper

mlx-whisper is a thin wrapper around Apple's MLX framework. The actual Whisper model weights (~1.6GB for large-v3-turbo) are downloaded from HuggingFace on first use, so the initial transcription run will take a minute to download.

```bash
pip install mlx-whisper

# Verify installation
python -c "import mlx_whisper; print('mlx-whisper ready')"
```

The package requires an Apple Silicon Mac. On Intel Macs, use `openai-whisper` instead (`pip install openai-whisper`), which runs on CPU.

## Step 2: Install and verify ffmpeg

mlx-whisper uses ffmpeg internally to decode video formats into raw audio. Without ffmpeg, it can only process .wav files. Most video files (.mp4, .mov, .mkv, .webm) require ffmpeg.

```bash
brew install ffmpeg

# Verify ffmpeg is available on PATH
ffmpeg -version | head -1
# Expected: ffmpeg version 7.x.x ...

# Test: extract audio from a video file
ffmpeg -i your_video.mp4 -vn -acodec copy test_audio.aac
```

If `brew install ffmpeg` fails, check that Homebrew is installed: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

## Step 3: Install analysis tools

LangChain-Ollama provides a clean interface to send transcript chunks to your local Ollama models. The `ollama` package is the lower-level Python client — LangChain is a higher-level wrapper that simplifies prompt management.

```bash
pip install langchain-ollama

# Pull the analysis model
ollama pull qwen2.5:7b
```

## Step 4: Test with a short audio clip

Before processing a full video, verify the pipeline works end-to-end on a short test. This catches ffmpeg path issues and model download problems before you commit to a 30-minute transcription job.

```python
import mlx_whisper

# Test with any short video or audio file (even 30 seconds)
result = mlx_whisper.transcribe(
    "test_clip.mp4",  # Replace with any file you have
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
)
print(result["text"])
print(f"Detected language: {result.get('language', 'unknown')}")
```

Expected: printed transcript text and language code (e.g., `en`). If you see the model downloading, wait for it to finish — the 1.6GB download is a one-time step.

<!-- tab: Transcribe -->
## Step 1: Basic transcription

The `path_or_hf_repo` parameter accepts either a local directory path or a HuggingFace model ID (prefixed with `mlx-community/`). The result dict contains `text` (full transcript), `segments` (time-chunked segments), and `language` (detected language code).

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "lecture.mp4",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
)

print(f"Language detected: {result['language']}")
print(f"Transcript length: {len(result['text'])} characters")
print("\nFirst 500 characters:")
print(result["text"][:500])
```

For audio-only files, pass the audio file directly. mlx-whisper handles .mp3, .wav, .m4a, .flac through ffmpeg.

## Step 2: Transcription with word-level timestamps

Setting `word_timestamps=True` adds a `words` list to each segment, with `start`, `end`, and `word` for every spoken word. This enables precise navigation: you can find where any topic was mentioned down to the second.

```python
result = mlx_whisper.transcribe(
    "lecture.mp4",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    word_timestamps=True,
)

# Format into a readable transcript with timestamps
def format_transcript(result: dict) -> str:
    """Format segments into a readable timestamped transcript."""
    lines = []
    for seg in result["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        # Format as [MM:SS - MM:SS] Text
        start_fmt = f"{int(start//60):02d}:{int(start%60):02d}"
        end_fmt = f"{int(end//60):02d}:{int(end%60):02d}"
        lines.append(f"[{start_fmt} - {end_fmt}] {text}")
    return "\n".join(lines)

formatted = format_transcript(result)
print(formatted[:1000])
```

## Step 3: Save the transcript in multiple formats

Saving in multiple formats serves different downstream uses: JSON preserves all timestamps for programmatic processing, plain text is easiest for LLM input, and SRT format works with video players and subtitle editors.

```python
import json

def save_transcript(result: dict, base_path: str = "transcript"):
    """Save transcript as JSON (with timestamps), plain text, and SRT."""

    # 1. JSON with full segment data (for programmatic use)
    with open(f"{base_path}.json", "w") as f:
        json.dump({
            "language": result["language"],
            "text": result["text"],
            "segments": result["segments"],
        }, f, indent=2)

    # 2. Plain text (for LLM input)
    with open(f"{base_path}.txt", "w") as f:
        f.write(result["text"])

    # 3. SRT subtitle format
    with open(f"{base_path}.srt", "w") as f:
        for i, seg in enumerate(result["segments"], 1):
            start = seg["start"]
            end = seg["end"]
            # SRT timestamps: HH:MM:SS,mmm
            def fmt_srt(t):
                h, m = int(t // 3600), int((t % 3600) // 60)
                s, ms = int(t % 60), int((t % 1) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            f.write(f"{i}\n{fmt_srt(start)} --> {fmt_srt(end)}\n{seg['text'].strip()}\n\n")

    print(f"Saved: {base_path}.json, {base_path}.txt, {base_path}.srt")

save_transcript(result)
```

## Step 4: Multilingual transcription and translation

Whisper auto-detects the spoken language. For non-English content, you can either transcribe in the original language or set `task="translate"` to produce an English transcript. Specifying the language explicitly (when you know it) improves accuracy and speeds up the first few seconds of transcription.

```python
# Explicit language for a Spanish video (improves accuracy)
result_es = mlx_whisper.transcribe(
    "video_spanish.mp4",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    language="es",       # ISO 639-1 language code
)

# Translate non-English audio directly to English transcript
result_translated = mlx_whisper.transcribe(
    "video_spanish.mp4",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    task="translate",    # Produce English output regardless of input language
)

print(f"Original language: {result_es['language']}")
print(f"Translated transcript: {result_translated['text'][:300]}")
```

Language codes: `en` (English), `es` (Spanish), `fr` (French), `zh` (Chinese), `ja` (Japanese), `de` (German). Whisper supports 99 languages.

<!-- tab: Analyze -->
## Step 1: Summarize with Ollama

Long transcripts may exceed the LLM's context window. `qwen2.5:7b` handles ~8,000 tokens comfortably (~6,000 words). For transcripts longer than this, take the first N characters for a summary of the opening, or chunk and summarize sections separately.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0.1)

with open("transcript.txt") as f:
    transcript = f.read()

# Truncate to ~6000 words (safe for 8k context model)
MAX_WORDS = 6000
words = transcript.split()
if len(words) > MAX_WORDS:
    transcript_trimmed = " ".join(words[:MAX_WORDS])
    print(f"Note: transcript truncated from {len(words)} to {MAX_WORDS} words")
else:
    transcript_trimmed = transcript

summary_prompt = f"""You are summarizing a video transcript. Be factual — only include information present in the transcript. Do not add outside knowledge.

Transcript:
{transcript_trimmed}

Provide:
1. A 2-3 sentence overview of the video
2. 5 key points as bullet points
3. Any specific data, numbers, or claims mentioned"""

response = llm.invoke(summary_prompt)
print(response.content)
```

## Step 2: Semantic search with timestamps

The LLM can find where specific topics were discussed and return the approximate timestamps from the formatted transcript. This is more useful than keyword search because it handles synonyms and paraphrases.

```python
def search_video(formatted_transcript: str, query: str) -> str:
    """Find where a topic is discussed, with timestamps."""
    prompt = f"""In this video transcript (formatted as [MM:SS - MM:SS] text),
find all segments discussing: "{query}"

For each relevant segment, quote the timestamp and a brief summary of what was said.
If the topic is not discussed, say "Not mentioned in this video."

Transcript:
{formatted_transcript[:5000]}"""

    response = llm.invoke(prompt)
    return response.content

# Load the formatted transcript
with open("transcript.json") as f:
    data = json.load(f)
    result_data = {"segments": data["segments"], "language": data["language"]}

formatted = format_transcript(result_data)

results = search_video(formatted, "machine learning inference speed")
print(results)
# Example output: "At [04:32 - 05:10], the speaker discussed..."
```

## Step 3: Q&A over the video content

The context-injection pattern passes the full transcript as context so the LLM answers only from what was said in the video. Explicitly instructing it to say "not mentioned" prevents confabulation.

```python
def ask_video(transcript: str, question: str) -> str:
    """Answer a question based only on the video transcript."""
    prompt = f"""Answer the following question based ONLY on the provided video transcript.
If the answer is not in the transcript, say "This was not covered in the video."

Question: {question}

Transcript:
{transcript[:5000]}

Answer:"""

    response = llm.invoke(prompt)
    return response.content

# Examples
questions = [
    "What hardware was used for the benchmarks?",
    "What are the main conclusions?",
    "Did the speaker mention any limitations?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask_video(transcript_trimmed, q)}\n")
```

## Step 4: Auto-generate chapter markers

Ask the LLM to identify natural topic transitions in the transcript and produce a table of contents. This works best on structured content (lectures, tutorials, interviews) with clear topic shifts.

```python
def generate_chapters(formatted_transcript: str) -> str:
    """Generate a chapter table of contents from the transcript."""
    prompt = f"""Analyze this video transcript and identify 5-8 natural topic transitions.
For each chapter, provide:
- The timestamp where the chapter starts (use the nearest segment timestamp)
- A short chapter title (3-6 words)

Format as a markdown table:
| Timestamp | Chapter Title |
|-----------|--------------|
| 00:00 | Introduction |
...

Transcript:
{formatted_transcript[:6000]}"""

    response = llm.invoke(prompt)
    return response.content

chapters = generate_chapters(formatted)
print(chapters)
```

## Step 5: Build a reusable CLI script

Combine all steps into a single command-line script that processes any video file with a single command.

```python
#!/usr/bin/env python3
"""video_analyze.py — transcribe and analyze any video file."""
import sys
import json
import mlx_whisper
from langchain_ollama import ChatOllama

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_analyze.py <video_file> [--summarize] [--chapters]")
        sys.exit(1)

    video_path = sys.argv[1]
    do_summarize = "--summarize" in sys.argv
    do_chapters = "--chapters" in sys.argv

    print(f"Transcribing {video_path}...")
    result = mlx_whisper.transcribe(
        video_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        word_timestamps=True,
    )

    # Save plain text
    with open("transcript.txt", "w") as f:
        f.write(result["text"])
    print(f"Saved transcript.txt ({len(result['text'])} chars, language={result['language']})")

    llm = ChatOllama(model="qwen2.5:7b", temperature=0.1)
    transcript = result["text"][:5000]  # Truncate for context window

    if do_summarize:
        resp = llm.invoke(f"Summarize in 5 bullet points:\n\n{transcript}")
        print("\n=== SUMMARY ===\n" + resp.content)

    if do_chapters:
        formatted = format_transcript(result)
        resp = llm.invoke(f"Generate a chapter table of contents with timestamps:\n\n{formatted[:5000]}")
        print("\n=== CHAPTERS ===\n" + resp.content)

if __name__ == "__main__":
    main()
```

Run it: `python video_analyze.py lecture.mp4 --summarize --chapters`

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: ffmpeg not found` | ffmpeg not installed or not on PATH | `brew install ffmpeg`; restart Terminal |
| Model download fails or hangs | HuggingFace download interrupted or disk full | Check `df -h` for disk space; retry — downloads resume |
| Poor transcription quality (many errors) | Wrong model size or noisy audio | Try `whisper-large-v3` (full model, slower but more accurate); specify `language=` explicitly |
| Very slow transcription (not 10x realtime) | MLX not using Metal GPU | Update mlx-whisper: `pip install -U mlx-whisper`; verify with `python -c "import mlx; print(mlx.default_device())"` |
| Transcript has repeated phrases or gibberish | Whisper hallucination on silence | Enable VAD: add `vad=True` parameter |
| Wrong language detected | Auto-detection fails on short clips | Specify `language="en"` (or your language code) explicitly |
| Video format not supported | Unusual codec or container | Pre-convert: `ffmpeg -i input.mkv -c:v copy -c:a aac output.mp4` |

### Enabling VAD to prevent hallucination on silence

Whisper sometimes generates text over silent sections (music, background noise). Voice Activity Detection skips these sections:

```python
result = mlx_whisper.transcribe(
    "video.mp4",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    vad=True,           # Skip silence segments
    vad_parameters={"min_silence_duration_ms": 500},  # Tune silence threshold
)
```

### Handling very long videos

For videos over 2 hours, the transcript may exceed the LLM context window. Process in chunks:

```python
def summarize_long_transcript(transcript: str, chunk_words: int = 5000) -> str:
    """Summarize a long transcript by summarizing chunks then combining."""
    llm = ChatOllama(model="qwen2.5:7b")
    words = transcript.split()
    chunks = [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        resp = llm.invoke(f"Summarize this section in 3 bullet points:\n\n{chunk}")
        chunk_summaries.append(f"Section {i+1}:\n{resp.content}")

    # Combine chunk summaries into final summary
    combined = "\n\n".join(chunk_summaries)
    final = llm.invoke(f"Combine these section summaries into a coherent overall summary:\n\n{combined}")
    return final.content
```

### Transcription is accurate but timestamps are off

Word-level timestamps can drift on very long recordings. For precise navigation in long videos, use segment-level timestamps (which are more reliable) rather than word-level:

```python
# Use segment timestamps instead of word timestamps for long videos
for seg in result["segments"]:
    print(f"[{seg['start']:.1f}s] {seg['text'].strip()}")
```
