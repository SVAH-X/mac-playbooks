---
slug: rag-langchain
title: "RAG with LangChain + Ollama"
time: "30 min"
color: orange
desc: "Fully local retrieval-augmented generation pipeline"
tags: [rag, langchain]
spark: "RAG in AI Workbench"
category: applications
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Retrieval-Augmented Generation (RAG) solves the problem of LLMs not knowing your private data. Instead of fine-tuning — which is expensive, slow, and bakes knowledge into weights that go stale — RAG retrieves relevant document passages at query time and injects them into the LLM's context window. The pipeline: embed your documents as dense vectors → store them in a vector database → at query time, embed the query → find the k nearest document vectors → send those chunks as context to the LLM → get a grounded answer with citations. Updating your knowledge base is as simple as re-embedding new documents; no retraining needed. The entire pipeline runs locally on your Mac using Ollama for both embeddings and the LLM.

## What you'll accomplish

A fully local RAG pipeline: documents loaded from text or PDF files, split into overlapping chunks, embedded with `nomic-embed-text` via Ollama, stored in ChromaDB on disk, and queried with semantic search. Answers come from `qwen2.5:7b` with source document citations. The vector store persists between sessions — you embed once and query repeatedly.

## What to know before starting

- **Text embeddings** — Dense floating-point vectors (768 dimensions for nomic-embed-text) where semantically similar texts are geometrically close. "The cat sat on the mat" and "A feline rested on a rug" produce vectors with high cosine similarity (~0.9), even though they share no words. This is the magic that makes semantic search work.
- **Vector database** — ChromaDB stores your document vectors and supports approximate nearest-neighbor (ANN) search. Given a query vector, it finds the k document vectors with highest cosine similarity in milliseconds, even across millions of documents.
- **Chunking and why it matters** — LLMs have context windows (e.g., 8k tokens for qwen2.5:7b). You can't fit a 100-page PDF. Chunks of 1000 tokens balance two competing concerns: large chunks give the LLM more context per retrieval, but small chunks give more precise retrieval (less noise). Overlapping chunks (200-token overlap) prevent losing context at chunk boundaries.
- **k-nearest-neighbor retrieval** — At query time, the top-k most similar chunks (typically k=3 to 5) are retrieved and concatenated into the LLM prompt. If k is too small, you miss relevant context; if too large, you dilute the context window with noise.
- **Why a dedicated embedding model** — `nomic-embed-text` is a 137M-parameter model optimized specifically for embedding quality. A general-purpose 7B model can also embed, but nomic-embed-text is 50× smaller (faster) and produces embeddings that rank comparably on retrieval benchmarks.

## Prerequisites

- Ollama installed and running (`ollama serve`)
- `nomic-embed-text` and `qwen2.5:7b` pulled
- Python 3.10+

## Time & risk

- **Duration:** 30 minutes setup; embedding time depends on document volume (~1 min per 100 pages)
- **Risk level:** Low — reads local files, writes to a local `chroma_db/` directory
- **Rollback:** Delete `chroma_db/` directory and `pip uninstall chromadb langchain`

<!-- tab: Setup -->
## Step 1: Install the RAG stack

Each package serves a distinct role. Installing them together resolves version compatibility.

```bash
pip install langchain langchain-ollama langchain-community chromadb pypdf
# langchain            — orchestration: chains, prompts, document loaders
# langchain-ollama     — OllamaEmbeddings and ChatOllama classes
# langchain-community  — document loaders (TextLoader, PyPDFLoader, DirectoryLoader)
#                        and Chroma vector store integration
# chromadb             — local vector database with persistent on-disk storage
# pypdf                — PDF parser (required for PyPDFLoader)

# Verify the key packages
python -c "import chromadb; print(f'ChromaDB {chromadb.__version__}')"
python -c "from langchain_ollama import OllamaEmbeddings; print('Ollama integration OK')"
```

## Step 2: Pull the embedding model

`nomic-embed-text` is a dedicated embedding model — much faster than using a full 7B model for embeddings, with comparable retrieval quality.

```bash
# Pull the embedding model (~137 MB)
ollama pull nomic-embed-text
# Also ensure your LLM is available
ollama pull qwen2.5:7b

# Verify both are available
ollama list
# Expected output includes:
# nomic-embed-text  latest  ...
# qwen2.5:7b        latest  ...
```

## Step 3: Test embeddings

Verify that Ollama is serving embeddings correctly before building the full pipeline.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Embed a test string and verify dimensions
test_vector = embeddings.embed_query("Hello, world!")
print(f"Embedding dimensions: {len(test_vector)}")
# Expected: 768 — nomic-embed-text produces 768-dimensional vectors

# Verify semantic similarity works
vec_a = embeddings.embed_query("The cat sat on the mat")
vec_b = embeddings.embed_query("A feline rested on a rug")
vec_c = embeddings.embed_query("The stock market closed up today")

import numpy as np
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Similar sentences: {cosine_sim(vec_a, vec_b):.3f}")  # expect ~0.85-0.95
print(f"Different topics:  {cosine_sim(vec_a, vec_c):.3f}")  # expect ~0.40-0.60
```

<!-- tab: Build RAG -->
## Step 1: Load documents

Choose the loader based on your file type. Each loader attaches source metadata that will appear in citations.

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

# --- Option A: Single text file ---
loader = TextLoader("my_document.txt", encoding="utf-8")
# encoding="utf-8" prevents errors on files with special characters
documents = loader.load()
# Result: list of Document objects, each with .page_content and .metadata["source"]

# --- Option B: Single PDF ---
loader = PyPDFLoader("my_document.pdf")
documents = loader.load()
# PyPDFLoader splits by page — each Document is one page
# metadata includes: {"source": "my_document.pdf", "page": 0}

# --- Option C: Entire directory ---
loader = DirectoryLoader(
    "./docs/",             # directory path
    glob="**/*.txt",       # file pattern (also: "**/*.pdf")
    loader_cls=TextLoader  # which loader to use per file
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")
print(f"First document snippet: {documents[0].page_content[:200]}")
print(f"Metadata: {documents[0].metadata}")
```

## Step 2: Split into chunks

The RecursiveCharacterTextSplitter tries to split on natural boundaries (paragraphs → sentences → words) to keep semantically coherent chunks.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # max characters per chunk (~250 tokens at 4 chars/token)
    chunk_overlap=200,    # overlap between adjacent chunks (prevents losing context at boundaries)
    length_function=len,  # measure chunk size in characters
    separators=["\n\n", "\n", ". ", " ", ""],
    # tries separators in order: paragraph breaks → line breaks → sentence ends → words → chars
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
print(f"First chunk ({len(chunks[0].page_content)} chars):")
print(chunks[0].page_content)
print(f"Metadata: {chunks[0].metadata}")
# Chunk metadata inherits from parent document: source, page number, etc.

# Verify overlap is working — chunks[0] and chunks[1] should share ~200 chars
overlap_check = chunks[0].page_content[-100:]
print(f"\nEnd of chunk 0: ...{overlap_check}")
print(f"Start of chunk 1: {chunks[1].page_content[:100]}...")
```

## Step 3: Create embeddings and vector store

`Chroma.from_documents()` embeds all chunks and stores them on disk. This is the expensive step — allow 1 minute per 100 pages of PDF.

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Embed all chunks and persist to disk
# This step: (1) sends each chunk to Ollama for embedding, (2) stores vectors + text in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # saves to disk — survives Python restarts
    collection_name="my_documents",   # name for this set of documents
)
print(f"Vector store created with {vectorstore._collection.count()} vectors")
# Each chunk becomes one vector — this number should equal len(chunks)
```

## Step 4: Build the retrieval chain

Assemble the retriever and LLM into a chain. The retriever does the vector search; the LLM generates the answer using retrieved context.

```python
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
# temperature=0: factual Q&A should be deterministic

# Configure the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",          # Maximum Marginal Relevance: avoids returning duplicate chunks
    search_kwargs={
        "k": 4,                 # retrieve 4 chunks per query
        "fetch_k": 20,          # MMR candidate pool size — fetch 20, re-rank to 4 diverse results
        "lambda_mult": 0.7,     # diversity weight: 0=max diversity, 1=max similarity
    }
)

# Assemble the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",         # "stuff" = concatenate all retrieved chunks into one prompt
    retriever=retriever,
    return_source_documents=True,  # include retrieved chunks in the response
    verbose=True,               # prints the prompt sent to the LLM (useful for debugging)
)
print("RAG chain ready")
```

## Step 5: Query and inspect results

Evaluate retrieval quality by examining source documents, not just the final answer.

```python
# Run a query
result = qa_chain.invoke({"query": "What are the main topics covered in this document?"})

print("=== ANSWER ===")
print(result["result"])

print("\n=== SOURCE DOCUMENTS (retrieved chunks) ===")
for i, doc in enumerate(result["source_documents"]):
    print(f"\nChunk {i+1} (from {doc.metadata.get('source', 'unknown')}):")
    print(doc.page_content[:200] + "...")

# To evaluate retrieval quality:
# 1. Check if the source chunks actually contain the answer
# 2. If the answer is wrong but chunks are correct → LLM generation issue (try rephrasing)
# 3. If the chunks are irrelevant → retrieval issue (adjust chunk_size, k, or search_type)
```

<!-- tab: Usage -->
## Loading an existing vector store

Once you've embedded your documents, load the persisted store directly without re-embedding.

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load existing store from disk — no embedding happens here
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_documents",
)
print(f"Loaded {vectorstore._collection.count()} existing vectors")
# Proceed to build the retriever and qa_chain as in Step 4 of Build RAG
```

## Adding new documents to an existing store

Add documents incrementally — existing vectors are preserved, new ones are appended.

```python
new_loader = TextLoader("new_document.txt")
new_docs = new_loader.load()
new_chunks = text_splitter.split_documents(new_docs)

# add_documents embeds and appends without rebuilding the full store
vectorstore.add_documents(new_chunks)
print(f"Added {len(new_chunks)} new chunks — total: {vectorstore._collection.count()}")
```

## Conversational RAG with memory

Extend the basic pipeline with conversation history so follow-up questions are understood in context.

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Memory keeps the last 5 exchanges (10 messages)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)

# First question
r1 = conv_chain.invoke({"question": "What is the main argument of this document?"})
print(r1["answer"])

# Follow-up uses conversation history — "it" refers to the previous context
r2 = conv_chain.invoke({"question": "Can you give me a specific example of it?"})
print(r2["answer"])
```

## Interactive Q&A loop

```python
print("RAG Q&A — type 'quit' to exit, 'sources' to see last retrieved chunks\n")
last_result = None

while True:
    query = input("Question: ").strip()
    if query.lower() == "quit":
        break
    if query.lower() == "sources" and last_result:
        for doc in last_result["source_documents"]:
            print(f"  [{doc.metadata.get('source')}]: {doc.page_content[:100]}...")
        continue
    last_result = qa_chain.invoke({"query": query})
    print(f"\nAnswer: {last_result['result']}\n")
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Embedding mismatch` | Changed embedding model after creating the store | Delete `chroma_db/` directory and re-embed with the new model |
| Embedding takes 30+ min | Large document collection, single-threaded | Batch documents in groups of 50 and embed each batch separately |
| "No documents returned" for a query | Chunk size too large vs query specificity | Reduce `chunk_size` to 500 and rebuild the store; specificity improves with smaller chunks |
| Answer is wrong despite correct chunks | LLM hallucinating instead of citing context | Add `"Answer only from the provided context. If unsure, say so."` to the LLM system prompt |
| `chromadb` version conflict | Incompatible langchain-community + chromadb versions | Pin: `pip install chromadb==0.4.24 langchain-community==0.2.x` |
| `UnicodeDecodeError` loading PDF | PDF has non-UTF-8 encoding | Try `PyMuPDFLoader` instead: `pip install pymupdf`, `from langchain_community.document_loaders import PyMuPDFLoader` |
| Ollama disconnects mid-embedding | Timeout on large batches | Add retry logic (see below) or reduce batch size |

### Diagnosing poor retrieval quality

If answers are wrong, the problem is almost always retrieval, not the LLM. Test retrieval directly:

```python
# Test retrieval quality without the LLM
docs = vectorstore.similarity_search_with_score(
    "your test query here", k=4
)
for doc, score in docs:
    print(f"Score: {score:.3f}")
    print(f"Content: {doc.page_content[:150]}\n")
# Lower score = more similar (cosine distance, not similarity)
# If scores are all > 0.5, your chunks are not matching — adjust chunk_size or rephrase query
```

### Retry logic for large embedding jobs

For collections with hundreds of documents, Ollama can time out. Wrap embedding in a retry loop:

```python
import time

def embed_with_retry(chunks, embeddings, persist_dir, max_retries=3):
    for attempt in range(max_retries):
        try:
            return Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # wait 5 seconds before retry
    raise RuntimeError("All embedding attempts failed")

vectorstore = embed_with_retry(chunks, embeddings, "./chroma_db")
```

### ChromaDB version pinning

LangChain's Chroma integration has breaking changes across ChromaDB versions. If you see `AttributeError` or import errors:

```bash
pip install "chromadb>=0.4.0,<0.5.0" "langchain-community>=0.2.0"
# Check what langchain-community expects:
pip show langchain-community | grep Requires
```
