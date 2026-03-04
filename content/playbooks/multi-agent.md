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

Multi-agent systems divide complex tasks among specialized AI agents that communicate through a shared message state. Instead of asking one LLM to do everything (research AND write AND critique), you route work to agents with focused system prompts. LangGraph models this as a directed graph where nodes are Python functions that call an LLM, and edges define the control flow between them. The graph's state — a list of messages — is passed through each node and updated. On Mac, all agents share the same local Ollama model via LangChain's Ollama integration, so no cloud API or API key is required.

## What you'll accomplish

A working multi-agent pipeline where a researcher agent gathers information and a writer agent synthesizes it into structured output, orchestrated by LangGraph's state machine. You will also build a conditional routing example that branches based on agent output, demonstrating how to create more sophisticated agentic workflows.

## What to know before starting

- **State machine** — A system that transitions between named states based on defined conditions. LangGraph manages a message list as state; each agent node appends to it, and edges determine which node runs next.
- **MessagesState** — LangGraph's built-in state type. It holds a list of messages (HumanMessage, AIMessage, SystemMessage) that grows as agents respond. Each node receives the full history and returns a message to append.
- **Nodes vs edges** — Nodes are Python functions that take state and return a state update. Edges are directed connections between nodes. Conditional edges inspect the state and choose which node to route to next — this is where branching logic lives.
- **StateGraph lifecycle** — You define nodes, add edges, set an entry point, set a finish point, then call `compile()`. The compiled graph is an invokable object. `invoke()` runs synchronously; `stream()` yields step-by-step output for debugging.
- **Why temperature=0** — LLM outputs for routing decisions (e.g., should we search or write?) must be deterministic. Setting `temperature=0` makes the model pick the highest-probability token at each step, reducing randomness in control-flow decisions.

## Prerequisites

- Ollama installed and running (`ollama serve` in a background terminal)
- `qwen2.5:7b` or larger pulled (`ollama pull qwen2.5:7b`)
- Python 3.10+

## Time & risk

- **Duration:** 30 minutes
- **Risk level:** Low — pure Python, all models run locally
- **Rollback:** `pip uninstall langgraph langchain-ollama langchain-core`

<!-- tab: Setup -->
## Step 1: Install LangGraph and LangChain Ollama

```bash
pip install langgraph langchain-ollama langchain-core
# langgraph       — the state machine orchestration framework
# langchain-ollama — Ollama integration (ChatOllama class)
# langchain-core  — base abstractions: BaseMessage, HumanMessage, AIMessage

# Verify versions
python -c "import langgraph; print(langgraph.__version__)"
# Expect 0.2.x or newer
python -c "from langchain_ollama import ChatOllama; print('OK')"
```

## Step 2: Verify Ollama connection

Before building the graph, confirm that LangChain can reach your local Ollama instance. Ollama must be running (`ollama serve`) and the model must be pulled.

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Test the connection with a simple invocation
llm = ChatOllama(model="qwen2.5:7b", temperature=0)
response = llm.invoke([HumanMessage(content="Say 'connection successful' and nothing else.")])
print(response.content)
# Expected: "connection successful"
# If you get a connection error, run: ollama serve  (in another terminal)
# If you get a model error, run: ollama pull qwen2.5:7b
```

## Step 3: Understand the graph model

Before writing code, understand what you are building:

```
Entry point
    │
    ▼
[researcher node] ── appends AIMessage with research findings to state
    │
    ▼
[writer node] ────── reads all messages in state, writes synthesis
    │
    ▼
Finish point

State at each step:
  After user input:    [HumanMessage("Analyze Apple Silicon for ML")]
  After researcher:    [HumanMessage(...), AIMessage("Key facts: ...")]
  After writer:        [HumanMessage(...), AIMessage("Key facts: ..."), AIMessage("Report: ...")]
```

The final `state["messages"][-1].content` is the writer's synthesized output.

<!-- tab: Build -->
## Step 1: Define the state

`MessagesState` is LangGraph's built-in state that holds a list of messages. For more complex pipelines, define a custom TypedDict state with additional fields.

```python
from langgraph.graph import StateGraph, MessagesState
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

# Option A: Use built-in MessagesState (simplest)
# MessagesState = TypedDict with {"messages": List[BaseMessage]}

# Option B: Custom state with extra fields (for complex pipelines)
class ResearchState(TypedDict):
    messages: List[BaseMessage]   # required for LangGraph message passing
    task: str                     # the original research topic
    research_complete: bool       # flag for conditional routing

# We'll use MessagesState for the basic example (cleaner) and
# ResearchState for the conditional routing example
```

## Step 2: Write the agent nodes

Each agent node is a Python function. It receives the current state, calls the LLM with a specialized system prompt, and returns a dict with the new message to append.

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
# temperature=0: makes routing decisions deterministic; use 0.7+ for creative writing nodes

def researcher(state: MessagesState) -> dict:
    # The system prompt defines this agent's role and output format
    system = SystemMessage(content=(
        "You are a research analyst. Given a topic, identify 5 key facts, "
        "current data points, and relevant context. Be specific and factual. "
        "Format: numbered list of facts."
    ))
    # Prepend system message to the conversation history
    response = llm.invoke([system] + state["messages"])
    # Return dict to append this AIMessage to the state
    return {"messages": [response]}

def writer(state: MessagesState) -> dict:
    system = SystemMessage(content=(
        "You are a technical writer. The conversation history contains research findings. "
        "Write a clear, well-structured 3-paragraph summary for a technical audience. "
        "Include specific data and end with actionable conclusions."
    ))
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}
```

## Step 3: Wire the graph

Define the graph structure by adding nodes and edges, then compile it into an invokable object.

```python
from langgraph.graph import StateGraph, MessagesState, END

# Create a new graph that uses MessagesState as its state type
graph = StateGraph(MessagesState)

# Add nodes — each node name maps to a function
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

# Add edges — define the control flow
graph.add_edge("researcher", "writer")  # researcher always goes to writer

# Define entry and exit
graph.set_entry_point("researcher")     # first node to run when invoked
graph.set_finish_point("writer")        # last node; after this, execution stops

# Compile into a runnable object
app = graph.compile()
print("Graph compiled successfully")
```

## Step 4: Invoke and trace execution

`invoke()` runs the full graph synchronously and returns the final state. `stream()` yields each step's output, which is useful for debugging.

```python
from langchain_core.messages import HumanMessage

# Invoke synchronously — returns the final state
result = app.invoke({
    "messages": [
        HumanMessage(content="Analyze the advantages of Apple Silicon for ML workloads.")
    ]
})

# The last message is the writer's output
print("=== FINAL REPORT ===")
print(result["messages"][-1].content)

# Inspect intermediate steps with stream()
print("\n=== STEP-BY-STEP TRACE ===")
for step in app.stream({
    "messages": [HumanMessage(content="Compare RISC-V and ARM architectures.")]
}):
    node_name = list(step.keys())[0]
    last_msg = step[node_name]["messages"][-1].content[:100]
    print(f"[{node_name}]: {last_msg}...")
```

## Step 5: Add conditional routing

Conditional edges route to different nodes based on the current state. This enables branching workflows — for example, sending to a search node only when the researcher decides it is needed.

```python
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def researcher_with_signal(state: MessagesState) -> dict:
    """Researcher signals whether it needs web search by outputting a keyword."""
    system = SystemMessage(content=(
        "You are a researcher. If you need current web data to answer properly, "
        "start your response with 'SEARCH_NEEDED:'. Otherwise just respond with facts."
    ))
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def web_search_stub(state: MessagesState) -> dict:
    """Placeholder: in production, call a real search API here."""
    mock_results = AIMessage(content="[Search results: Latest benchmark data from 2024...]")
    return {"messages": [mock_results]}

def route_after_research(state: MessagesState) -> str:
    """Return the name of the next node based on the researcher's last message."""
    last_message = state["messages"][-1].content
    if last_message.startswith("SEARCH_NEEDED:"):
        return "web_search"  # branch to search node
    return "writer"          # skip search, go straight to writer

# Build the conditional graph
cond_graph = StateGraph(MessagesState)
cond_graph.add_node("researcher", researcher_with_signal)
cond_graph.add_node("web_search", web_search_stub)
cond_graph.add_node("writer", writer)

cond_graph.set_entry_point("researcher")
# Conditional edge: after researcher, call route_after_research() to decide next node
cond_graph.add_conditional_edges("researcher", route_after_research)
cond_graph.add_edge("web_search", "writer")  # after search, always go to writer
cond_graph.set_finish_point("writer")

cond_app = cond_graph.compile()
```

<!-- tab: Examples -->
## Code review pipeline

A three-node pipeline where a coder writes code, a reviewer critiques it, and a fixer applies improvements.

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

def coder(state: MessagesState) -> dict:
    system = SystemMessage(content="You write clean, efficient Python code with docstrings and type hints.")
    return {"messages": [llm.invoke([system] + state["messages"])]}

def reviewer(state: MessagesState) -> dict:
    system = SystemMessage(content=(
        "You review Python code for: bugs, security issues, edge cases, and style. "
        "List specific improvements as numbered items."
    ))
    return {"messages": [llm.invoke([system] + state["messages"])]}

def fixer(state: MessagesState) -> dict:
    system = SystemMessage(content="Apply the reviewer's suggestions and output the improved, complete code.")
    return {"messages": [llm.invoke([system] + state["messages"])]}

g = StateGraph(MessagesState)
g.add_node("coder", coder)
g.add_node("reviewer", reviewer)
g.add_node("fixer", fixer)
g.add_edge("coder", "reviewer")
g.add_edge("reviewer", "fixer")
g.set_entry_point("coder")
g.set_finish_point("fixer")

code_app = g.compile()
result = code_app.invoke({"messages": [
    {"role": "user", "content": "Write a function to parse JSONL files with error handling."}
]})
print(result["messages"][-1].content)  # the fixed, improved code
```

## Document Q&A with citation

A two-node pipeline that extracts relevant passages, then answers with citations.

```python
def extractor(state: MessagesState) -> dict:
    """First agent: find relevant quotes from the document."""
    system = SystemMessage(content=(
        "You extract the most relevant 2-3 verbatim quotes from context that answer the user's question. "
        "Format each quote as: [QUOTE N]: 'exact text...'"
    ))
    return {"messages": [llm.invoke([system] + state["messages"])]}

def answerer(state: MessagesState) -> dict:
    """Second agent: synthesize the extracted quotes into an answer."""
    system = SystemMessage(content=(
        "Using the extracted quotes in the conversation, provide a direct answer to the original question. "
        "Reference each quote as (Quote 1), (Quote 2) etc."
    ))
    return {"messages": [llm.invoke([system] + state["messages"])]}
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` on invoke | Ollama not running | Run `ollama serve` in a separate terminal; verify with `curl localhost:11434` |
| Graph runs forever (infinite loop) | Missing `set_finish_point` | Add `graph.set_finish_point("last_node_name")` before `compile()` |
| Agent ignores its system prompt | System prompt appended after messages | Prepend system message: `[system] + state["messages"]`, not the reverse |
| `GraphCompileError: node not found` | Node name typo in `add_edge` | Verify node names in `add_node` and `add_edge` match exactly (case-sensitive) |
| Inconsistent routing decisions | `temperature` too high for routing | Set `temperature=0` on the LLM used for routing/classification decisions |
| State grows unboundedly in long sessions | Every message is appended forever | Trim old messages: `state["messages"] = state["messages"][-10:]` to keep last 10 |
| `langgraph` import error | Package not installed | Run `pip install langgraph>=0.2.0` — older versions have different APIs |

### Preventing infinite loops

A common mistake is building a graph with no finish point, causing it to loop back to the entry node forever. Always set a finish point:

```python
# Wrong: circular graph with no exit
graph.add_edge("writer", "researcher")  # loops back — will run until timeout

# Correct: add a finish point
graph.set_finish_point("writer")  # execution stops after writer completes

# For conditional loops (e.g., retry on bad output), use END explicitly:
from langgraph.graph import END
graph.add_conditional_edges("critic", lambda s: "retry" if "poor quality" in s["messages"][-1].content else END)
```

### Message trimming for long sessions

The messages list grows with every agent invocation. For a researcher → writer pipeline run 100 times in a loop, the state becomes enormous and LLM context windows overflow. Trim early messages:

```python
from langchain_core.messages import trim_messages

def researcher(state: MessagesState) -> dict:
    # Keep only the last 10 messages to avoid context window overflow
    trimmed = trim_messages(state["messages"], max_tokens=4000, strategy="last")
    response = llm.invoke([system] + trimmed)
    return {"messages": [response]}
```
