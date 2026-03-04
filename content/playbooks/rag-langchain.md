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

Build a retrieval-augmented generation pipeline that runs entirely locally. Your documents are embedded with a local model, stored in a local vector database, and retrieved to answer questions using a local LLM.

## Prerequisites

- Ollama installed and running
- Python 3.10+
- Documents to index (PDF, text, etc.)

## Time & risk

- **Duration:** 30 minutes
- **Risk level:** Low

<!-- tab: Setup -->
## Install dependencies

```bash
pip install langchain langchain-ollama langchain-community chromadb sentence-transformers pypdf
```

## Pull embedding and chat models

```bash
ollama pull nomic-embed-text  # fast local embeddings
ollama pull qwen2.5:7b         # for answering
```

<!-- tab: Build RAG -->
## Create the RAG pipeline

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load documents
loader = TextLoader("my_document.txt")  # or PyPDFLoader("doc.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# Create RAG chain
llm = ChatOllama(model="qwen2.5:7b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
```

## Query your documents

```python
result = qa_chain.invoke({"query": "What are the main topics in this document?"})
print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:100]}...")
```

<!-- tab: Usage -->
## Load existing vector store

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

## Interactive Q&A loop

```python
while True:
    query = input("Question (or 'quit'): ")
    if query.lower() == "quit":
        break
    result = qa_chain.invoke({"query": query})
    print(f"\nAnswer: {result['result']}\n")
```
