---
slug: txt2kg
title: "Text to Knowledge Graph"
time: "30 min"
color: orange
desc: "Extract knowledge triples with local LLMs and graph DBs"
tags: [knowledge graph, nlp]
spark: "Text to Knowledge Graph"
category: applications
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Extract structured knowledge triples (subject → relationship → object) from unstructured text using local LLMs via Ollama, then store them in a graph database or visualize them.

## Prerequisites

- Ollama installed and running with llama3.1:8b or similar
- Docker (for Neo4j option)
- Python 3.10+

## Time & risk

- **Duration:** 30 minutes
- **Risk level:** Low

<!-- tab: Setup -->
## Pull a good reasoning model

```bash
ollama pull llama3.1:8b
```

## Option A: Use the DGX Spark txt2kg project (macOS compatible)

```bash
git clone https://github.com/leokwsw/dgx-spark-txt2kg.git
cd dgx-spark-txt2kg

# Edit docker-compose.yml: remove GPU reservation lines
# Then start:
docker compose up -d
```

Access at http://localhost:3001.

## Option B: Python + NetworkX (no Docker)

```bash
pip install langchain langchain-ollama networkx matplotlib pyvis
```

<!-- tab: Extract Triples -->
## Extract triples with Ollama

```python
from langchain_ollama import ChatOllama
import json

llm = ChatOllama(model="llama3.1:8b", temperature=0)

def extract_triples(text: str) -> list[dict]:
    prompt = f"""Extract knowledge graph triples from this text.
Return ONLY a JSON array of objects with keys: subject, predicate, object.
Text: {text}

JSON:"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return []

text = """
Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
Apple developed the iPhone in 2007. Tim Cook became CEO in 2011.
"""

triples = extract_triples(text)
for t in triples:
    print(f"{t['subject']} --[{t['predicate']}]--> {t['object']}")
```

<!-- tab: Visualize -->
## Build and visualize with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
for t in triples:
    G.add_edge(t["subject"], t["object"], label=t["predicate"])

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=2000, font_size=10, arrows=True)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.savefig("knowledge_graph.png", dpi=150, bbox_inches="tight")
```

## Interactive HTML visualization

```python
from pyvis.network import Network

net = Network(height="600px", width="100%", directed=True)
for node in G.nodes():
    net.add_node(node, label=node)
for u, v, data in G.edges(data=True):
    net.add_edge(u, v, label=data.get("label", ""))

net.save_graph("knowledge_graph.html")
```
