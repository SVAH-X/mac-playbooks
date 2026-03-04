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

A knowledge graph represents information as (subject, predicate, object) triples — for example, (Apple, founded_by, Steve Jobs) or (iPhone, released_in, 2007). Extracting these triples from unstructured text with a local LLM converts free-form documents into a structured, queryable graph where you can ask "what are all the things connected to Apple, two hops away?"

On your Mac, Ollama provides the LLM for extraction. NetworkX stores the graph in memory for immediate visualization, and Neo4j (optional, via Docker) provides persistent storage with a query language built for graph traversal.

## What you'll accomplish

A Python pipeline that takes any text document, extracts knowledge triples using a local LLM, and stores them as a directed graph. You'll produce an interactive HTML visualization (drag nodes, zoom, inspect edges) and optionally export the graph to Neo4j for persistent, Cypher-queryable storage.

## What to know before starting

- **Knowledge graphs**: Nodes are entities (Apple, Steve Jobs, 1976). Edges are typed, directional relationships (founded_by, acquired, competes_with). Direction matters: Apple --[founded_by]--> Steve Jobs is not the same as Steve Jobs --[founded_by]--> Apple.
- **Triple extraction limitations**: LLMs hallucinate relationships. Always spot-check extracted triples against the source text before treating them as facts. Critical applications need human review.
- **Graph databases vs relational**: Relational databases optimize for row-based queries. Graph databases optimize for traversal — "find everything connected to node X within N hops" is a single query in Cypher but requires complex JOINs in SQL.
- **NetworkX**: Python's in-memory graph library. Fast for analysis and visualization, but data is lost when the process ends. Use Neo4j if you need persistence.
- **Cypher**: Neo4j's query language. `MATCH (a)-[:founded_by]->(b) RETURN a, b` is the graph equivalent of a SQL SELECT with JOIN.
- **Few-shot prompting**: Including 2-3 examples in the prompt dramatically improves JSON formatting consistency from the LLM.

## Prerequisites

- Ollama running with `llama3.1:8b` pulled (or `qwen2.5:7b` as fallback)
- Python 3.10+
- Docker Desktop (optional, for Neo4j persistent storage)

## Time & risk

- **Duration:** 30 minutes
- **Risk level:** Low — pure Python, no system changes, no data leaves your machine

<!-- tab: Setup -->
## Step 1: Install Python dependencies

LangChain orchestrates LLM calls so you can swap models without rewriting code. NetworkX handles the in-memory graph structure. PyVis wraps NetworkX graphs into interactive HTML using a physics simulation engine. The Neo4j driver is optional — only install it if you plan to use persistent storage.

```bash
pip install langchain langchain-ollama networkx matplotlib pyvis

# Optional: Neo4j persistent storage
pip install neo4j
```

Verify the install worked:

```bash
python -c "import networkx, pyvis; print('Ready')"
```

## Step 2: Pull a reasoning-capable model

`llama3.1:8b` is recommended for triple extraction because it follows JSON schema instructions more reliably than smaller models. Its instruction-tuned variant understands the "output only JSON" constraint better than base models.

```bash
# Recommended: better JSON schema adherence
ollama pull llama3.1:8b

# Fallback if disk space is limited (~4.7GB vs ~8.5GB)
ollama pull qwen2.5:7b
```

Verify the model is loaded and responding:

```bash
ollama run llama3.1:8b "Reply with just the word: ready"
```

Expected output: `ready` (or similar single-word confirmation)

## Step 3: Start Neo4j with Docker (optional)

Neo4j stores your graph persistently so you can query it across sessions with Cypher. The configuration below disables authentication (fine for local development) and caps memory usage to 1GB so it doesn't compete with your LLM.

```bash
docker run -d \
  --name neo4j-kg \
  -p 7474:7474 \   # Browser UI
  -p 7687:7687 \   # Bolt protocol (Python driver)
  -e NEO4J_AUTH=none \
  -e NEO4J_dbms_memory_heap_max__size=1G \
  neo4j:latest
```

Access the browser UI at http://localhost:7474 — no login needed. You should see the Neo4j dashboard. Leave this step for the end if you're not sure you need it.

## Step 4: Test LLM extraction on a simple sentence

Before processing full documents, verify that your model outputs valid JSON for a single sentence. This catches model or format issues early.

```python
from langchain_ollama import ChatOllama
import json

llm = ChatOllama(model="llama3.1:8b", temperature=0)

test_prompt = """Extract knowledge triples from this sentence.
Return ONLY a JSON array. Each element must have keys: subject, predicate, object.
No explanation. No markdown. Just the JSON array.

Sentence: Apple was founded by Steve Jobs in 1976.

JSON:"""

response = llm.invoke(test_prompt)
print(repr(response.content))  # Show raw output including whitespace/newlines

triples = json.loads(response.content)
print(triples)
# Expected: [{"subject": "Apple", "predicate": "founded_by", "object": "Steve Jobs"}, ...]
```

If you get a `json.JSONDecodeError`, the model added extra text around the JSON. Add "Do not include any text before or after the JSON array" to the prompt.

<!-- tab: Extract Triples -->
## Step 1: Define the extraction prompt with few-shot examples

Few-shot prompting (showing the model 2-3 input/output examples in the prompt itself) dramatically improves JSON formatting consistency. The model learns the exact output format from the examples rather than trying to infer it from the instruction alone.

```python
EXTRACTION_PROMPT = """Extract knowledge graph triples from text.
Return ONLY a JSON array. Each element has keys: subject, predicate, object.
Use snake_case for predicates. Keep entities concise (no articles).

Examples:
Input: "Marie Curie won the Nobel Prize in 1903."
Output: [{"subject": "Marie Curie", "predicate": "won", "object": "Nobel Prize"}]

Input: "Google acquired DeepMind in 2014 for $500 million."
Output: [
  {{"subject": "Google", "predicate": "acquired", "object": "DeepMind"}},
  {{"subject": "Google", "predicate": "acquired_in_year", "object": "2014"}}
]

Now extract from this text:
Input: "{text}"
Output:"""
```

## Step 2: Write the extraction function with error handling

LLMs sometimes output malformed JSON — extra trailing commas, unescaped quotes, or text before the array. The retry logic below re-prompts with a stricter instruction when parsing fails, rather than silently dropping the result.

```python
from langchain_ollama import ChatOllama
import json
import re

llm = ChatOllama(model="llama3.1:8b", temperature=0)

def extract_triples(text: str, retries: int = 2) -> list[dict]:
    """Extract knowledge triples from text. Retries on JSON parse failure."""
    prompt = EXTRACTION_PROMPT.format(text=text.strip())

    for attempt in range(retries + 1):
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Remove markdown code fences if model added them
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        try:
            triples = json.loads(raw)
            # Normalize: strip whitespace from all values
            return [
                {k: v.strip() for k, v in t.items()}
                for t in triples
                if all(k in t for k in ("subject", "predicate", "object"))
            ]
        except json.JSONDecodeError:
            if attempt < retries:
                # Retry with stricter instruction
                prompt = prompt + "\nIMPORTANT: Output ONLY the JSON array, nothing else."
            else:
                print(f"Warning: Could not parse JSON after {retries} retries. Raw: {raw[:200]}")
                return []
```

## Step 3: Process a full document paragraph by paragraph

Processing the entire document at once causes hallucinations — the model loses track of which facts came from which sentence. Chunking by paragraph keeps context tight. Deduplication removes repeated triples that appear when the same fact is mentioned multiple times.

```python
def process_document(text: str) -> list[dict]:
    """Extract triples from all paragraphs, deduplicate results."""
    # Split into paragraphs, skip empty ones and headings
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]

    all_triples = []
    seen = set()

    for i, para in enumerate(paragraphs):
        print(f"Processing paragraph {i+1}/{len(paragraphs)}...")
        triples = extract_triples(para)

        for t in triples:
            # Create a canonical key for deduplication
            key = (t["subject"].lower(), t["predicate"].lower(), t["object"].lower())
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

    print(f"Extracted {len(all_triples)} unique triples from {len(paragraphs)} paragraphs")
    return all_triples

# Example usage
sample_text = """
Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
Apple developed the Macintosh computer in 1984. Steve Jobs was CEO of Apple.

Apple acquired NeXT in 1997, which brought Steve Jobs back to the company.
The iPhone was released in 2007 and changed the smartphone industry.
Tim Cook became CEO of Apple in 2011 after Steve Jobs resigned.
"""

triples = process_document(sample_text)
for t in triples:
    print(f"  {t['subject']} --[{t['predicate']}]--> {t['object']}")
```

## Step 4: Build the NetworkX directed graph

Knowledge graphs must be directed — Apple --[founded_by]--> Steve Jobs conveys that Apple was founded by Jobs, not the reverse. Adding edge attributes (the predicate as the "label" attribute) lets us display relationship types in visualizations.

```python
import networkx as nx

def build_graph(triples: list[dict]) -> nx.DiGraph:
    """Build a directed graph from extracted triples."""
    G = nx.DiGraph()

    for t in triples:
        G.add_edge(
            t["subject"],
            t["object"],
            label=t["predicate"],  # Relationship type stored as edge attribute
        )

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # Inspect most-connected nodes
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Most connected: {top_nodes}")
    return G

G = build_graph(triples)
```

Expected output: `Graph: 12 nodes, 10 edges` followed by the most-connected entity (likely "Apple" or "Steve Jobs").

<!-- tab: Visualize -->
## Step 1: Static visualization with NetworkX and matplotlib

The spring_layout algorithm positions nodes using a force-directed simulation — entities with more connections end up near the center, isolated entities at the periphery. Node size scaled by degree makes highly-connected entities visually prominent.

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_static(G: nx.DiGraph, output_path: str = "knowledge_graph.png"):
    """Save a static PNG visualization of the knowledge graph."""
    plt.figure(figsize=(14, 10))

    # Force-directed layout: connected nodes attract, all nodes repel
    pos = nx.spring_layout(G, k=2, seed=42)  # k controls spacing

    # Scale node size by number of connections (degree)
    degrees = dict(G.degree())
    node_sizes = [300 + degrees[n] * 200 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color="lightblue",
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20,
                           edge_color="gray", alpha=0.7)

    # Annotate edges with relationship labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, font_color="darkred")

    plt.title("Knowledge Graph", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

visualize_static(G)
```

Open `knowledge_graph.png` to see the result. Node sizes reflect how many relationships each entity participates in.

## Step 2: Interactive HTML with PyVis

PyVis wraps the vis.js physics engine to produce an HTML file where you can drag nodes, zoom, and hover over edges to see relationship labels. This is more useful than the static PNG for exploring dense graphs.

```python
from pyvis.network import Network

def visualize_interactive(G: nx.DiGraph, output_path: str = "knowledge_graph.html"):
    """Create an interactive HTML visualization."""
    net = Network(
        height="700px", width="100%",
        directed=True,
        bgcolor="#1a1a2e",   # Dark background
        font_color="white",
    )

    # Enable physics controls in the UI
    net.show_buttons(filter_=["physics"])

    # Add nodes — size by degree, color by connectivity
    for node in G.nodes():
        degree = G.degree(node)
        net.add_node(
            node,
            label=node,
            size=15 + degree * 5,           # Larger = more connections
            color="#4ecdc4" if degree > 2 else "#95e1d3",
            title=f"{node}\nConnections: {degree}",  # Hover tooltip
        )

    # Add edges with relationship labels
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data.get("label", ""), arrows="to")

    net.save_graph(output_path)
    print(f"Interactive graph saved to {output_path}")
    print("Open in your browser: open knowledge_graph.html")

visualize_interactive(G)
```

Run `open knowledge_graph.html` in Terminal to open in your default browser. Drag nodes to rearrange, scroll to zoom.

## Step 3: Export to Neo4j (optional)

Cypher `MERGE` is used instead of `CREATE` to avoid duplicate nodes when re-importing. Batch imports are much faster than one-at-a-time — the `unwind` pattern sends all triples in a single transaction.

```python
from neo4j import GraphDatabase

def export_to_neo4j(triples: list[dict], uri: str = "bolt://localhost:7687"):
    """Export triples to Neo4j using batch Cypher import."""
    driver = GraphDatabase.driver(uri, auth=None)  # auth=None because NEO4J_AUTH=none

    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")

        # Batch import all triples in one transaction
        session.run("""
            UNWIND $triples AS triple
            MERGE (s:Entity {name: triple.subject})
            MERGE (o:Entity {name: triple.object})
            MERGE (s)-[:RELATION {type: triple.predicate}]->(o)
        """, triples=triples)

    print(f"Exported {len(triples)} triples to Neo4j")
    print("Browse at http://localhost:7474 — run: MATCH (n) RETURN n")
    driver.close()

export_to_neo4j(triples)
```

## Step 4: Query the graph

NetworkX traversal queries let you explore the graph in Python. Neo4j Cypher queries are more expressive for complex patterns.

```python
# NetworkX: find all nodes reachable from "Apple" within 2 hops
apple_neighborhood = nx.ego_graph(G, "Apple", radius=2)
print(f"Entities within 2 hops of Apple: {list(apple_neighborhood.nodes())}")

# NetworkX: shortest path between two entities
path = nx.shortest_path(G, "Steve Jobs", "Tim Cook")
print(f"Connection path: {' -> '.join(path)}")
```

For Neo4j, open http://localhost:7474 and run Cypher:

```cypher
-- Find everything Apple is directly connected to
MATCH (apple:Entity {name: "Apple"})-[r]->(other)
RETURN apple.name, r.type, other.name

-- Two-hop discovery: what is Apple connected to indirectly?
MATCH (apple:Entity {name: "Apple"})-[*1..2]->(other)
RETURN DISTINCT other.name
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `json.JSONDecodeError` on LLM output | Model added explanation text around JSON | Add "Output ONLY the JSON array, no explanation" to prompt; increase retries |
| Hallucinated relationships in output | LLM invented facts not in source text | Always spot-check triples against source; reduce chunk size for better focus |
| Same entity appears as "Apple" and "Apple Inc." | Entity disambiguation: same entity, different strings | Add a post-processing step that maps aliases to canonical names |
| Very slow processing on large documents | Sequential LLM calls are slow | Parallelize with `concurrent.futures.ThreadPoolExecutor` |
| `ConnectionRefusedError` to Neo4j | Docker container not running | Run `docker ps` to check; `docker start neo4j-kg` to restart |
| `knowledge_graph.html` won't open | Browser security blocking local file | Try `python -m http.server` in the directory and open via localhost |

### Handling hallucinated relationships

LLMs are more likely to hallucinate when the input text is ambiguous, long, or contains proper nouns they weren't trained on. Mitigations:

```python
# After extraction, filter triples where subject or object don't appear in source
def validate_triples(triples: list[dict], source_text: str) -> list[dict]:
    source_lower = source_text.lower()
    valid = []
    for t in triples:
        # Both subject and object should appear verbatim in source
        if (t["subject"].lower() in source_lower and
                t["object"].lower() in source_lower):
            valid.append(t)
        else:
            print(f"Filtered (not in source): {t}")
    return valid
```

### Entity canonicalization

When the same entity appears as "Steve Jobs", "Jobs", and "Steve", the graph will have three separate nodes. A simple approach uses exact-match aliases:

```python
ALIASES = {
    "jobs": "Steve Jobs",
    "steve": "Steve Jobs",
    "apple inc": "Apple",
    "apple inc.": "Apple",
}

def canonicalize(entity: str) -> str:
    return ALIASES.get(entity.lower(), entity)
```

### Neo4j won't start

Check that Docker has enough memory allocated (Neo4j needs at least 1GB):

```bash
docker stats neo4j-kg          # Check memory usage
docker logs neo4j-kg           # Check startup errors
# If OOM: increase Docker memory limit in Docker Desktop → Settings → Resources
```

<!-- tab: Troubleshooting -->
## Quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `json.JSONDecodeError` | Model wrapped JSON in markdown or added text | Strip code fences with regex; add "ONLY JSON array" to prompt |
| Triples reference non-existent entities | LLM hallucination | Validate: both subject and object must appear in source text |
| Entity "Apple" and "Apple Inc." are separate nodes | No canonicalization | Build an alias dict and normalize before adding to graph |
| Processing 50-page document takes 30+ min | Sequential LLM calls | Parallelize paragraph processing with ThreadPoolExecutor |
| `ConnectionRefusedError` on bolt://localhost:7687 | Neo4j Docker not running | `docker start neo4j-kg` or re-run the docker run command |
| PyVis HTML shows blank page | File opened directly; browser blocks local JS | Serve with `python -m http.server 8080` and open http://localhost:8080 |

### Improving JSON output reliability

If the model consistently produces malformed JSON, switch to structured output mode. Some Ollama models support a `format` parameter:

```python
# Force JSON output mode (works with llama3.1 and qwen2.5)
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    format="json",   # Constrains output to valid JSON
)
```

Note: `format="json"` ensures valid JSON syntax but not the specific schema you want — you still need the prompt to specify the keys.

### Debugging missing relationships

If obvious relationships are missing from the output, the chunk may be too large or the model is prioritizing certain entity types. Test with a focused single-sentence input:

```python
# Test extraction quality on a known sentence
test = "OpenAI was founded in 2015 by Sam Altman, Elon Musk, and Greg Brockman."
triples = extract_triples(test)
# Should produce at least: OpenAI --[founded_by]--> Sam Altman (and others)
print(triples)
```

### Performance for large document collections

For processing hundreds of documents, the bottleneck is LLM inference speed. Strategies:

- Use `qwen2.5:7b` instead of `llama3.1:8b` (2x faster, slightly lower quality)
- Parallelize: run 3-4 extraction calls concurrently (Ollama handles concurrent requests)
- Cache results: save extracted triples to JSON after each document so you don't re-process on failure
