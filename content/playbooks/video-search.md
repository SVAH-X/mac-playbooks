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

Transcribe video/audio using Whisper running locally via MLX, then use an LLM or VLM to search, summarize, or answer questions about the content.

## Prerequisites

- macOS 14.0+
- Apple Silicon Mac
- Python 3.10+
- A video or audio file

## Time & risk

- **Duration:** 30 minutes setup
- **Risk level:** Low

<!-- tab: Setup -->
## Install mlx-whisper

```bash
pip install mlx-whisper
```

## Install analysis tools

```bash
pip install ollama langchain-ollama
ollama pull qwen2.5:7b
```

<!-- tab: Transcribe -->
## Transcribe with mlx-whisper

```python
import mlx_whisper

# Transcribe audio/video
result = mlx_whisper.transcribe(
    "video.mp4",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
)
print(result["text"])

# Save transcript
with open("transcript.txt", "w") as f:
    f.write(result["text"])
```

## With timestamps (for search)

```python
result = mlx_whisper.transcribe(
    "video.mp4",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    word_timestamps=True,
)

for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
```

<!-- tab: Analyze -->
## Summarize the transcript

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b")

with open("transcript.txt") as f:
    transcript = f.read()

# Truncate if needed
transcript = transcript[:4000]

response = llm.invoke(f"Summarize this video transcript in 3-5 bullet points:\n\n{transcript}")
print(response.content)
```

## Search for topics

```python
def search_transcript(transcript: str, query: str) -> str:
    response = llm.invoke(
        f"In this transcript, find all mentions of '{query}' and give timestamps if available:\n\n{transcript}"
    )
    return response.content

result = search_transcript(transcript, "Apple Silicon performance")
print(result)
```

## Q&A over video content

```python
def answer_question(transcript: str, question: str) -> str:
    response = llm.invoke(
        f"Based on this transcript, answer: {question}\n\nTranscript:\n{transcript}"
    )
    return response.content

answer = answer_question(transcript, "What were the main topics discussed?")
print(answer)
```
