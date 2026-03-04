# MacBook AI/ML Playbooks

> **The macOS counterpart to [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)** — a collection of step-by-step playbooks for setting up AI/ML workloads on Apple Silicon MacBooks.

<p align="center"><em>Optimized for Apple Silicon (M1/M2/M3/M4/M5) · Unified Memory Architecture · Metal GPU Acceleration</em></p>

---

## About

These playbooks provide detailed instructions for:

- Installing and configuring popular AI frameworks **natively on macOS**
- Running inference with models optimized for Apple Silicon
- Setting up development environments leveraging Metal and MLX
- Fine-tuning and training models locally
- Robotics simulation without NVIDIA GPUs

Each playbook includes prerequisites, step-by-step instructions, troubleshooting guidance, and example code. Where a DGX Spark playbook relies on NVIDIA-only technology, we provide the closest macOS-native or cross-platform alternative.

---

## Mapping: DGX Spark → MacBook

| DGX Spark Playbook | MacBook Equivalent | Notes |
|---|---|---|
| Ollama | **[Ollama](#ollama)** | Native macOS support, identical workflow |
| Open WebUI with Ollama | **[Open WebUI with Ollama](#open-webui-with-ollama)** | Identical on macOS |
| vLLM for Inference | **[MLX LM for Inference](#mlx-lm-for-inference)** | MLX is the Apple-native high-perf engine |
| TRT-LLM for Inference | **[llama.cpp with Metal](#llamacpp-with-metal)** | llama.cpp with Metal backend replaces TensorRT |
| SGLang Inference Server | **[MLX Serving with LM Studio](#mlx-serving-with-lm-studio)** | LM Studio uses MLX backend natively |
| NIM on Spark | **[Ollama + Open WebUI API](#ollama)** | Local model serving via Ollama API |
| NVFP4 Quantization | **[MLX Quantization](#mlx-quantization)** | MLX supports 2/4/8-bit quantization natively |
| Speculative Decoding | **[Speculative Decoding with llama.cpp](#speculative-decoding-with-llamacpp)** | llama.cpp supports spec decoding on Metal |
| CUDA-X Data Science | **[Apple Accelerate + MLX Data Science](#accelerate-mlx-data-science)** | Accelerate framework + MLX for numerical compute |
| Optimized JAX | **[JAX on Apple Silicon](#jax-on-apple-silicon)** | JAX supports Metal via jax-metal plugin |
| PyTorch Fine-tune | **[PyTorch MPS Fine-tuning](#pytorch-mps-fine-tuning)** | PyTorch MPS backend for Apple GPU |
| NeMo Fine-tune | **[MLX Fine-tuning (LoRA/QLoRA)](#mlx-fine-tuning)** | MLX LM supports LoRA fine-tuning natively |
| LLaMA Factory | **[LLaMA Factory on macOS](#llama-factory-on-macos)** | Works on macOS with MPS backend |
| Unsloth | **[MLX LoRA Fine-tuning](#mlx-fine-tuning)** | Unsloth is CUDA-only; MLX LoRA is the replacement |
| FLUX.1 Dreambooth LoRA | **[MLX Stable Diffusion / FLUX](#mlx-image-generation)** | MLX has native FLUX and SD support |
| Comfy UI | **[ComfyUI on macOS](#comfyui-on-macos)** | ComfyUI works on macOS with MPS |
| Multi-modal Inference | **[MLX VLM Inference](#mlx-vlm-inference)** | MLX supports vision-language models |
| Multi-Agent Chatbot | **[Multi-Agent Chatbot (LangGraph)](#multi-agent-chatbot)** | Framework-level, works cross-platform |
| RAG in AI Workbench | **[RAG with LangChain + Ollama](#rag-with-langchain-ollama)** | Fully local RAG pipeline on macOS |
| Text to Knowledge Graph | **[Text to Knowledge Graph](#text-to-knowledge-graph)** | Ollama + Neo4j/ArangoDB, works on macOS |
| Isaac Sim / Isaac Lab | **[MuJoCo + MuJoCo Playground](#mujoco-robotics-simulation)** | Cross-platform robotics sim, runs on Apple Silicon |
| NCCL for Two Sparks | **[Distributed Training (Thunderbolt)](#distributed-training)** | Limited; use cloud offload or single-node |
| Connect Two Sparks | *(No direct equivalent)* | Single-machine focus |
| DGX Dashboard | **[macOS Activity Monitor + asitop](#system-monitoring)** | asitop shows GPU/ANE/CPU usage on Apple Silicon |
| VS Code | **[VS Code on macOS](#vscode-setup)** | Native Apple Silicon build |
| Vibe Coding in VS Code | **[Vibe Coding with Continue.dev](#vibe-coding)** | Continue.dev + local Ollama on macOS |
| Tailscale | **[Tailscale on macOS](#tailscale-on-macos)** | Native macOS app |
| Set Up Local Network Access | **[macOS Network Sharing](#network-sharing)** | Built-in macOS features |
| Nemotron-3-Nano with llama.cpp | **[llama.cpp with Metal](#llamacpp-with-metal)** | Same workflow, Metal instead of CUDA |
| LM Studio on DGX Spark | **[LM Studio on macOS](#mlx-serving-with-lm-studio)** | LM Studio is Mac-native (MLX backend) |
| Video Search & Summarization | **[Video Search & Summarization](#video-search-summarization)** | Whisper.cpp + VLM pipeline |
| Portfolio Optimization | **[Portfolio Optimization](#portfolio-optimization)** | Python scientific stack, fully cross-platform |
| Single-cell RNA Sequencing | **[scRNA-seq Analysis](#scrna-seq-analysis)** | scanpy/AnnData, cross-platform |

---

## Available Playbooks

### Inference

#### Ollama

> **DGX Spark equivalent: Ollama**

Run large language models locally with a single command.

**Prerequisites**
- macOS 13.0+ (Ventura or later)
- Apple Silicon Mac (M1 or later)
- 8 GB+ unified memory (16 GB+ recommended for larger models)

**Install**
```bash
# Install via Homebrew (recommended)
brew install ollama

# Or download from https://ollama.com/download/mac
```

**Run**
```bash
# Start the Ollama server (runs as a background service)
ollama serve &

# Pull and run a model
ollama pull qwen2.5:32b
ollama run qwen2.5:32b

# For machines with 8GB RAM, use smaller models
ollama pull qwen2.5:7b
ollama run qwen2.5:7b
```

**API Usage**
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:32b",
  "messages": [{ "role": "user", "content": "Write me a haiku about Apple Silicon." }],
  "stream": false
}'
```

**Memory Guidelines**
| Unified Memory | Recommended Max Model Size |
|---|---|
| 8 GB | 7B Q4 |
| 16 GB | 14B Q4 or 7B FP16 |
| 32 GB | 32B Q4 or 14B FP16 |
| 64 GB | 70B Q4 or 32B FP16 |
| 96–192 GB | 70B FP16, 120B Q4+ |

**Troubleshooting**
- If Ollama is slow, ensure no other large processes are consuming memory. Check with `Activity Monitor > Memory`.
- Use `OLLAMA_MAX_LOADED_MODELS=1` to avoid loading multiple models simultaneously on constrained memory.

---

#### MLX LM for Inference

> **DGX Spark equivalent: vLLM for Inference**

MLX LM is Apple's high-performance LLM inference engine, purpose-built for Apple Silicon's unified memory architecture. It achieves ~230 tok/s on M2 Ultra — the fastest local inference runtime on Mac.

**Prerequisites**
- macOS 14.0+ (Sonoma or later recommended)
- Apple Silicon Mac
- Python 3.10+

**Install**
```bash
pip install mlx-lm
```

**Run Inference (CLI)**
```bash
# Run a 4-bit quantized model directly from Hugging Face
mlx_lm.generate \
  --model mlx-community/Qwen2.5-32B-Instruct-4bit \
  --prompt "Explain the unified memory architecture of Apple Silicon." \
  --max-tokens 512

# Start an OpenAI-compatible API server
mlx_lm.server --model mlx-community/Qwen2.5-32B-Instruct-4bit --port 8080
```

**Run Inference (Python)**
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-32B-Instruct-4bit")
response = generate(
    model, tokenizer,
    prompt="What is Metal GPU acceleration?",
    max_tokens=256,
    verbose=True  # prints tokens/sec
)
print(response)
```

**Why MLX over vLLM on Mac?**
- MLX is built from scratch for Apple Silicon's unified memory — zero-copy between CPU and GPU
- vLLM requires CUDA; it does not run on macOS
- MLX achieves higher throughput than llama.cpp, Ollama, and PyTorch MPS on Apple hardware
- Supports quantized models (2-bit to 8-bit) natively with minimal accuracy loss

**Troubleshooting**
- For M5 with Neural Accelerators: ensure macOS 26.2+ and latest MLX (`pip install -U mlx mlx-lm`)
- Out of memory: use a more aggressively quantized model (e.g., 3-bit or 2-bit)

---

#### llama.cpp with Metal

> **DGX Spark equivalent: TRT-LLM for Inference / Nemotron-3-Nano with llama.cpp**

llama.cpp treats Apple Silicon as a first-class citizen with optimized Metal kernels, ARM NEON, and Accelerate framework integration.

**Install**
```bash
# Option 1: Homebrew
brew install llama.cpp

# Option 2: Build from source (latest optimizations)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

**Run**
```bash
# Download a GGUF model (e.g., from Hugging Face)
# Then run with full GPU offload
./build/bin/llama-cli \
  -m ~/models/qwen2.5-32b-instruct-q4_k_m.gguf \
  -ngl 99 \
  -c 8192 \
  -p "Explain how Metal acceleration works for LLM inference."

# Start an OpenAI-compatible API server
./build/bin/llama-server \
  -m ~/models/qwen2.5-32b-instruct-q4_k_m.gguf \
  -ngl 99 \
  -c 8192 \
  --port 8080
```

**Key flags**
- `-ngl 99` — offload all layers to Metal GPU (critical for performance)
- `-c 8192` — context window size
- `-t $(sysctl -n hw.perflevel0.logicalcpu)` — use only performance cores

**Troubleshooting**
- Ensure Metal is being used: look for `ggml_metal_init` in output
- If build fails, install Xcode command line tools: `xcode-select --install`

---

#### MLX Serving with LM Studio

> **DGX Spark equivalent: SGLang Inference Server / LM Studio on DGX Spark**

LM Studio provides a polished GUI for local LLM inference on Mac, using MLX as its backend for maximum Apple Silicon performance.

**Install**
1. Download from [lmstudio.ai](https://lmstudio.ai) (native Apple Silicon build)
2. Launch and search for a model (e.g., `Qwen 2.5 32B Instruct`)
3. Download the MLX variant for best performance

**Start Local Server**
1. Go to the **Server** tab in LM Studio
2. Load your model
3. Click **Start Server** — exposes an OpenAI-compatible API at `http://localhost:1234/v1`

```bash
# Test the server
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-32b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

#### Speculative Decoding with llama.cpp

> **DGX Spark equivalent: Speculative Decoding**

Use a small draft model to speed up generation from a large target model.

```bash
./build/bin/llama-speculative \
  -m ~/models/qwen2.5-32b-instruct-q4_k_m.gguf \
  -md ~/models/qwen2.5-1.5b-instruct-q8_0.gguf \
  -ngl 99 -ngld 99 \
  -c 4096 \
  --draft-max 8 \
  -p "Write a detailed analysis of transformer architectures."
```

**How it works**: The 1.5B draft model generates candidate tokens at high speed; the 32B target model verifies them in a single forward pass. Accepted tokens are essentially "free" — typical acceptance rates of 60–80% yield 1.5–2.5x speedup.

---

#### MLX VLM Inference

> **DGX Spark equivalent: Multi-modal Inference / Live VLM WebUI**

Run vision-language models locally with MLX.

```bash
pip install mlx-vlm
```

```python
from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

output = generate(
    model, processor,
    "Describe what you see in this image.",
    image="path/to/image.jpg",
    max_tokens=256
)
print(output)
```

---

### Quantization

#### MLX Quantization

> **DGX Spark equivalent: NVFP4 Quantization**

Quantize any Hugging Face model for efficient Apple Silicon inference.

```bash
# Quantize a model to 4-bit
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-32B-Instruct \
  --mlx-path ./qwen2.5-32b-4bit \
  --quantize \
  --q-bits 4

# Quantize to 8-bit for higher quality
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-32B-Instruct \
  --mlx-path ./qwen2.5-32b-8bit \
  --quantize \
  --q-bits 8

# Quantize to 2-bit for maximum compression
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-32B-Instruct \
  --mlx-path ./qwen2.5-32b-2bit \
  --quantize \
  --q-bits 2
```

**Quantization Guidelines**
| Bits | Quality | Memory Savings | Best For |
|---|---|---|---|
| 8-bit | Near-lossless | ~50% | Quality-critical tasks |
| 4-bit | Good | ~75% | General use (recommended default) |
| 3-bit | Acceptable | ~81% | Memory-constrained machines |
| 2-bit | Noticeable degradation | ~87% | Fitting large models in limited RAM |

Many pre-quantized models are available in the [`mlx-community`](https://huggingface.co/mlx-community) organization on Hugging Face.

---

### Fine-tuning

#### MLX Fine-tuning

> **DGX Spark equivalent: NeMo Fine-tune / Unsloth / PyTorch Fine-tune**

Fine-tune language models with LoRA/QLoRA directly on your Mac using MLX.

```bash
pip install mlx-lm
```

**Prepare Data** (JSONL format)
```json
{"text": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>"}
{"text": "<|im_start|>user\nCapital of France?<|im_end|>\n<|im_start|>assistant\nParis<|im_end|>"}
```

**Fine-tune with LoRA**
```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data ./training_data \
  --train \
  --iters 1000 \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --lora-layers 16 \
  --adapter-path ./adapters
```

**Merge and Export**
```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path ./adapters \
  --save-path ./fused-model
```

**Run Fused Model**
```bash
mlx_lm.generate --model ./fused-model --prompt "Test prompt"
```

---

#### PyTorch MPS Fine-tuning

> **DGX Spark equivalent: PyTorch Fine-tune**

Use PyTorch's MPS (Metal Performance Shaders) backend for GPU-accelerated training on Mac.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Verify MPS is available
print(f"MPS available: {torch.backends.mps.is_available()}")

device = torch.device("mps")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16
).to(device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # keep small for memory
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=False,  # MPS doesn't support fp16 training; use float32
    use_mps_device=True,
    save_strategy="epoch",
)
```

**Note**: PyTorch MPS is functional but significantly slower than MLX for most Apple Silicon workloads. Prefer MLX fine-tuning unless you need PyTorch-specific features (custom training loops, specific optimizers, etc.).

---

#### LLaMA Factory on macOS

> **DGX Spark equivalent: LLaMA Factory**

LLaMA Factory works on macOS with the MPS backend.

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# Launch the web UI
llamafactory-cli webui
```

Set the device to `mps` in the configuration. Note that not all features are available on MPS — some operations may fall back to CPU.

---

### Image Generation

#### MLX Image Generation

> **DGX Spark equivalent: FLUX.1 Dreambooth LoRA Fine-tuning**

Generate images with Stable Diffusion and FLUX models using MLX.

```bash
pip install mlx-image
# Or use the dedicated SD package
pip install mflux
```

**Generate with FLUX (via mflux)**
```bash
pip install mflux

# Generate a 1024x1024 image
mflux-generate \
  --model dev \
  --prompt "A serene Japanese garden with cherry blossoms, digital art" \
  --steps 20 \
  --seed 42 \
  --width 1024 --height 1024
```

**Generate with Stable Diffusion (via MLX)**
```python
from mlx_image import StableDiffusionPipeline

pipe = StableDiffusionPipeline("mlx-community/stable-diffusion-xl-base-1.0-4bit")
image = pipe.generate(
    "A futuristic cityscape at sunset, photorealistic",
    num_steps=30,
    seed=42
)
image.save("output.png")
```

---

#### ComfyUI on macOS

> **DGX Spark equivalent: Comfy UI**

ComfyUI works on macOS using the PyTorch MPS backend.

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Run with MPS acceleration
python main.py --force-fp16
```

Open `http://127.0.0.1:8188` in your browser. Download model checkpoints into `models/checkpoints/`.

**Note**: Some custom nodes may not be compatible with MPS. Check node compatibility before installing.

---

### Applications

#### Open WebUI with Ollama

> **DGX Spark equivalent: Open WebUI with Ollama**

Deploy a full ChatGPT-like interface locally.

```bash
# Ensure Ollama is running
ollama serve &

# Option 1: Docker (recommended)
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# Option 2: pip install
pip install open-webui
open-webui serve --port 3000
```

Open `http://localhost:3000`. Select a model from the dropdown and start chatting.

---

#### Multi-Agent Chatbot

> **DGX Spark equivalent: Build and Deploy a Multi-Agent Chatbot**

Build a multi-agent system using LangGraph with local Ollama models.

```bash
pip install langgraph langchain-ollama
```

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:32b", temperature=0)

# Define agent nodes
def researcher(state: MessagesState):
    response = llm.invoke(state["messages"] + [
        {"role": "system", "content": "You are a research analyst. Find key facts."}
    ])
    return {"messages": [response]}

def writer(state: MessagesState):
    response = llm.invoke(state["messages"] + [
        {"role": "system", "content": "You are a writer. Synthesize research into prose."}
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
result = app.invoke({"messages": [{"role": "user", "content": "Analyze trends in AI hardware."}]})
```

---

#### RAG with LangChain + Ollama

> **DGX Spark equivalent: RAG Application in AI Workbench**

Build a fully local Retrieval-Augmented Generation pipeline.

```bash
pip install langchain langchain-ollama langchain-community chromadb sentence-transformers
```

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Embed documents locally
embeddings = OllamaEmbeddings(model="nomic-embed-text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Load and split your documents
docs = text_splitter.create_documents(["Your document text here..."])

# Create vector store
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Create RAG chain
llm = ChatOllama(model="qwen2.5:32b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What are the key points in my documents?"})
print(result["result"])
```

---

#### Text to Knowledge Graph

> **DGX Spark equivalent: Text to Knowledge Graph**

Extract knowledge triples from text using local LLMs.

```bash
# Start Ollama
ollama serve &
ollama pull llama3.1:8b

# Option: Use the DGX Spark txt2kg project (works on macOS with Ollama)
git clone https://github.com/leokwsw/dgx-spark-txt2kg.git
cd dgx-spark-txt2kg

# Modify docker-compose.yml to remove --gpus all flag
# Replace: deploy.resources.reservations.devices (remove GPU reservation)
# Then:
docker compose up -d
```

Access the web UI at `http://localhost:3001`.

---

#### Video Search & Summarization

> **DGX Spark equivalent: Build a Video Search and Summarization Agent**

Transcribe and analyze video content locally.

```bash
pip install mlx-whisper
```

```python
import mlx_whisper

# Transcribe audio/video
result = mlx_whisper.transcribe(
    "video.mp4",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
)
print(result["text"])

# Then use Ollama/MLX LM for summarization
import subprocess, json

summary_prompt = f"Summarize this transcript:\n\n{result['text'][:4000]}"
resp = subprocess.run(
    ["ollama", "run", "qwen2.5:32b", summary_prompt],
    capture_output=True, text=True
)
print(resp.stdout)
```

---

### Data Science

#### Accelerate + MLX Data Science

> **DGX Spark equivalent: CUDA-X Data Science**

Apple's Accelerate framework and MLX provide GPU-accelerated numerical computing on macOS.

```bash
pip install mlx numpy pandas scipy scikit-learn matplotlib jupyter
```

```python
import mlx.core as mx
import numpy as np

# MLX arrays live in unified memory — CPU and GPU access the same data
a = mx.random.normal((10000, 10000))
b = mx.random.normal((10000, 10000))

# Matrix multiply runs on Metal GPU automatically
c = a @ b
mx.eval(c)  # force evaluation (MLX is lazy)

# Seamless interop with NumPy
np_array = np.array(c)
```

**For RAPIDS-like workflows**: Use pandas, cuML alternatives:
- `scikit-learn` for ML (uses Accelerate for BLAS operations)
- `polars` for high-performance DataFrames
- MLX for custom GPU-accelerated numerical compute

---

#### JAX on Apple Silicon

> **DGX Spark equivalent: Optimized JAX**

JAX runs on Apple Silicon via the Metal backend.

```bash
pip install jax jaxlib
# For Metal GPU acceleration (experimental):
pip install jax-metal
```

```python
import jax
import jax.numpy as jnp

print(jax.devices())  # Should show Metal device

# GPU-accelerated computation
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (5000, 5000))
result = jnp.dot(x, x.T)
print(f"Result shape: {result.shape}")
```

**Note**: jax-metal is experimental and may not support all JAX operations. For MuJoCo MJX workloads, MJX runs well on Apple Silicon via JAX (650K steps/sec on M3 Max reported).

---

#### Portfolio Optimization

> **DGX Spark equivalent: Portfolio Optimization**

```bash
pip install numpy scipy pandas yfinance cvxpy matplotlib
```

```python
import numpy as np
import cvxpy as cp
import yfinance as yf

tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
data = yf.download(tickers, start="2023-01-01", end="2025-01-01")["Close"]
returns = data.pct_change().dropna()

mu = returns.mean().values * 252
sigma = returns.cov().values * 252

w = cp.Variable(len(tickers))
ret = mu @ w
risk = cp.quad_form(w, sigma)

prob = cp.Problem(
    cp.Maximize(ret - 0.5 * risk),
    [cp.sum(w) == 1, w >= 0]
)
prob.solve()
print(dict(zip(tickers, np.round(w.value, 4))))
```

---

#### scRNA-seq Analysis

> **DGX Spark equivalent: Single-cell RNA Sequencing**

```bash
pip install scanpy anndata leidenalg
```

```python
import scanpy as sc

adata = sc.read_h5ad("pbmc3k.h5ad")  # or sc.datasets.pbmc3k()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color="leiden", save="clusters.png")
```

---

### Robotics Simulation

#### MuJoCo Robotics Simulation

> **DGX Spark equivalent: Isaac Sim / Isaac Lab**

MuJoCo is the cross-platform alternative to Isaac Sim. It runs natively on macOS, supports Apple Silicon, and is the physics engine behind Google DeepMind's robotics research. MuJoCo Playground (RSS 2025 Outstanding Demo Paper) enables training locomotion and manipulation policies with zero-shot sim-to-real transfer.

**Why MuJoCo over Isaac Sim on Mac?**
- Isaac Sim requires NVIDIA GPUs (PhysX, RTX rendering) — it cannot run on macOS
- MuJoCo runs natively on macOS with excellent Apple Silicon performance (650K steps/sec on M3 Max)
- MuJoCo MJX (JAX backend) enables GPU-accelerated parallel simulation on Apple Silicon
- MuJoCo Playground provides ready-made environments for quadrupeds, humanoids, dexterous hands, and robotic arms

**Install**
```bash
pip install mujoco
# For GPU-accelerated parallel simulation:
pip install mujoco-mjx jax jax-metal
# For the Playground (RL training environments):
pip install playground
```

**Run the Interactive Viewer**
```bash
# Launch the built-in viewer with a humanoid model
python -m mujoco.viewer
```

**MuJoCo MJX on Apple Silicon**
```python
import mujoco
from mujoco import mjx
import jax

# Load model
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# Put model on device (Metal GPU)
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

# Batch simulate (massively parallel on GPU)
batch_size = 1024
batched_data = jax.vmap(lambda _: mjx_data)(jax.numpy.arange(batch_size))

@jax.jit
def step(data):
    return mjx.step(mjx_model, data)

# Run simulation steps
for _ in range(1000):
    batched_data = jax.vmap(step)(batched_data)
```

**MuJoCo Playground — Train a Robot**
```bash
pip install playground

# Train a quadruped to walk
python -m playground.train \
  --env_name "Go2Locomotion" \
  --num_envs 4096 \
  --num_timesteps 50_000_000
```

**Comparison: Isaac Lab vs MuJoCo**
| Feature | Isaac Lab (NVIDIA) | MuJoCo + Playground (macOS) |
|---|---|---|
| Physics engine | PhysX (GPU) | MuJoCo / MJX (CPU + GPU) |
| Rendering | RTX photorealistic | OpenGL / basic |
| Platform | Linux + NVIDIA GPU only | macOS, Linux, Windows |
| Parallel envs | 4,096+ on GPU | 1,024+ via MJX on Apple Silicon |
| RL integration | Built-in | Via Playground / Brax |
| Sim-to-real | Excellent | Excellent (proven zero-shot transfer) |
| Photorealistic data | Yes | No (use Blender for synthetic data) |

For photorealistic synthetic data generation (which Isaac Sim excels at), combine MuJoCo with Blender for rendering.

---

### Development Environment

#### VS Code Setup

> **DGX Spark equivalent: VS Code**

```bash
# Download Apple Silicon native build from https://code.visualstudio.com
# Or via Homebrew:
brew install --cask visual-studio-code
```

**Recommended Extensions for ML on Mac**:
- Python (ms-python)
- Jupyter (ms-toolsai)
- Continue.dev (AI coding assistant with local models)
- GitHub Copilot (cloud-based alternative)

---

#### Vibe Coding

> **DGX Spark equivalent: Vibe Coding in VS Code**

Use your Mac as a local AI coding assistant with Continue.dev + Ollama.

1. Install [Continue.dev](https://continue.dev) extension in VS Code
2. Ensure Ollama is running with a code-capable model:
   ```bash
   ollama pull qwen2.5-coder:32b
   ```
3. Configure Continue (`.continue/config.yaml`):
   ```yaml
   models:
     - model: qwen2.5-coder:32b
       title: Qwen Coder 32B
       provider: ollama
       apiBase: http://localhost:11434
   tabAutocompleteModel:
     model: qwen2.5-coder:7b
     title: Qwen Coder 7B
     provider: ollama
   ```

---

### System & Networking

#### System Monitoring

> **DGX Spark equivalent: DGX Dashboard**

Monitor Apple Silicon GPU, CPU, ANE, and memory usage.

```bash
# Install asitop — a performance monitoring tool for Apple Silicon
pip install asitop

# Run (requires sudo for IOKit access)
sudo asitop
```

`asitop` displays real-time:
- CPU usage per cluster (efficiency + performance cores)
- GPU usage and frequency
- ANE (Apple Neural Engine) usage
- Memory bandwidth and pressure
- Power consumption

Alternative: `powermetrics` (built into macOS)
```bash
sudo powermetrics --samplers gpu_power,cpu_power -i 1000
```

---

#### Tailscale on macOS

> **DGX Spark equivalent: Set up Tailscale on Your Spark**

```bash
# Install via Homebrew
brew install tailscale

# Or download the Mac app from https://tailscale.com/download/mac

# Authenticate
tailscale up

# Check status
tailscale status
```

---

#### Network Sharing

> **DGX Spark equivalent: Set Up Local Network Access / Connect to Your Spark**

macOS has built-in remote access:

1. **SSH**: System Settings → General → Sharing → Remote Login (enable)
2. **Screen Sharing**: System Settings → General → Sharing → Screen Sharing (enable)
3. Find your IP: `ipconfig getifaddr en0`

```bash
# From another machine:
ssh username@your-mac-ip

# Or use Bonjour:
ssh username@your-mac.local
```

---

#### Distributed Training

> **DGX Spark equivalent: NCCL for Two Sparks / Connect Two Sparks**

Multi-Mac distributed training is limited compared to NCCL over InfiniBand. Options:

1. **PyTorch Distributed (Gloo backend)** — works over Ethernet/Thunderbolt between Macs, but significantly slower than NCCL
2. **Cloud offload** — train on cloud GPUs, develop/iterate locally
3. **Single-machine optimization** — Apple Silicon's large unified memory (up to 192GB on Mac Studio) often eliminates the need for multi-node for research-scale work

For most users, a single Mac with sufficient unified memory is the practical path.

---

## Hardware Recommendations

| Workload | Minimum | Recommended |
|---|---|---|
| Chat with 7B models | M1, 8GB | M2+, 16GB |
| Run 32B–70B models | M2 Pro, 32GB | M2 Max/Ultra, 64GB+ |
| Fine-tune 7B models | M2 Pro, 32GB | M3 Max, 64GB |
| Fine-tune 32B+ models | M2 Ultra, 96GB | M2/M4 Ultra, 192GB |
| Image generation (FLUX) | M2, 16GB | M3+, 32GB |
| Robotics simulation | M1, 16GB | M3 Pro+, 36GB |
| Full-stack ML research | M2 Max, 64GB | M4 Ultra, 192GB |

---

## Quick Start: Your First 10 Minutes

```bash
# 1. Install Ollama
brew install ollama && ollama serve &

# 2. Pull a model
ollama pull qwen2.5:7b

# 3. Chat
ollama run qwen2.5:7b

# 4. Install MLX for high-performance inference
pip install mlx-lm
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "Hello!"

# 5. Install monitoring
pip install asitop && sudo asitop
```

---

## Resources

- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **MLX Examples**: https://github.com/ml-explore/mlx-examples
- **MLX Community Models**: https://huggingface.co/mlx-community
- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **MuJoCo**: https://mujoco.org
- **MuJoCo Playground**: https://playground.mujoco.org
- **Ollama**: https://ollama.com
- **LM Studio**: https://lmstudio.ai
- **Apple ML Research**: https://machinelearning.apple.com
- **WWDC25 MLX Sessions**: https://developer.apple.com/videos/play/wwdc2025/315/

---

## License

This project is provided under the [Apache-2.0 License](LICENSE).

The original [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) are licensed under Apache-2.0 by NVIDIA Corporation.
