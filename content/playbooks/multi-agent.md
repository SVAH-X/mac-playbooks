---
slug: multi-agent
title: "Multi-Agent Chatbot"
time: "30 min"
color: orange
desc: "Build a multi-agent system with LangGraph + local Ollama"
tags: [agents, langchain]
spark: "Multi-Agent Chatbot"
category: applications
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Build a multi-agent system using LangGraph where specialized agents collaborate on complex tasks. All models run locally via Ollama — no cloud API needed.

## Prerequisites

- Ollama installed and running
- A capable model pulled (qwen2.5:32b recommended for quality)
- Python 3.10+

## Time & risk

- **Duration:** 30 minutes
- **Risk level:** Low

<!-- tab: Setup -->
## Install dependencies

```bash
pip install langgraph langchain-ollama langchain-core
```

## Ensure Ollama is running with a model

```bash
ollama serve &
ollama pull qwen2.5:7b
```

<!-- tab: Build -->
## Basic researcher + writer pipeline

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

def researcher(state: MessagesState):
    response = llm.invoke(state["messages"] + [
        {"role": "system", "content": "You are a research analyst. Find key facts and data points."}
    ])
    return {"messages": [response]}

def writer(state: MessagesState):
    response = llm.invoke(state["messages"] + [
        {"role": "system", "content": "You are a writer. Synthesize the research into clear, engaging prose."}
    ])
    return {"messages": [response]}

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge("researcher", "writer")
graph.set_entry_point("researcher")
graph.set_finish_point("writer")

app = graph.compile()
result = app.invoke({
    "messages": [{"role": "user", "content": "Analyze the advantages of Apple Silicon for ML workloads."}]
})
print(result["messages"][-1].content)
```

<!-- tab: Examples -->
## Code reviewer agent

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b")

def coder(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"] + [
        {"role": "system", "content": "You write clean, efficient Python code."}
    ])]}

def reviewer(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"] + [
        {"role": "system", "content": "You review code for bugs, security issues, and improvements."}
    ])]}

graph = StateGraph(MessagesState)
graph.add_node("coder", coder)
graph.add_node("reviewer", reviewer)
graph.add_edge("coder", "reviewer")
graph.set_entry_point("coder")
graph.set_finish_point("reviewer")

app = graph.compile()
result = app.invoke({"messages": [
    {"role": "user", "content": "Write a function to parse JSONL files with error handling."}
]})
```
