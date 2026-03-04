# 🍎 mac-playbooks

> **The macOS counterpart to [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)** — step-by-step playbooks for setting up AI/ML workloads on Apple Silicon MacBooks.

[![GitHub Pages](https://img.shields.io/badge/Live%20Site-svah--x.github.io%2Fmac--playbooks-0A84FF?style=flat-square&logo=github)](https://svah-x.github.io/mac-playbooks/)
[![Playbooks](https://img.shields.io/badge/Playbooks-29-30D158?style=flat-square)](https://svah-x.github.io/mac-playbooks/)
[![License](https://img.shields.io/badge/License-Apache%202.0-white?style=flat-square)](LICENSE)

<p align="center">
  <em>Optimized for Apple Silicon (M1/M2/M3/M4/M5) · Unified Memory Architecture · Metal GPU Acceleration</em>
</p>

---

## About

These playbooks provide detailed, step-by-step instructions for running AI/ML workloads on Apple Silicon MacBooks — the macOS equivalent of each DGX Spark playbook.

Each playbook includes:
- **Prerequisites** and hardware requirements
- **Step-by-step installation** and configuration
- **Usage examples** with real code
- **Troubleshooting** for common issues

Where a DGX Spark playbook relies on NVIDIA-only technology (CUDA, TensorRT, PhysX), we provide the closest macOS-native or cross-platform alternative.

---

## 🌐 Live Portal

Browse all playbooks at **[svah-x.github.io/mac-playbooks](https://svah-x.github.io/mac-playbooks/)**

---

## ⚡ Quick Start

```bash
# 1. Install Ollama (5 min)
brew install ollama && ollama serve &

# 2. Pull a model and chat
ollama pull qwen2.5:7b
ollama run qwen2.5:7b

# 3. Install MLX for maximum performance
pip install mlx-lm
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Hello, Apple Silicon!"

# 4. Monitor GPU/CPU/ANE usage
pip install asitop && sudo asitop
```

---

## 📚 Playbooks

### 🚀 Onboarding

| Playbook | Time | Description |
|---|---|---|
| [Ollama](content/playbooks/ollama.md) | 5 min | Install and run LLMs locally with a single command |
| [Open WebUI with Ollama](content/playbooks/open-webui.md) | 15 min | Deploy a full ChatGPT-like interface locally |

### ⚡ Inference

| Playbook | Time | Description |
|---|---|---|
| [MLX LM for Inference](content/playbooks/mlx-lm.md) | 10 min | Apple's high-performance native LLM engine — fastest on Mac |
| [llama.cpp with Metal](content/playbooks/llama-cpp.md) | 15 min | Run GGUF models with first-class Apple Silicon Metal acceleration |
| [LM Studio on macOS](content/playbooks/lm-studio.md) | 5 min | Polished GUI for local LLM inference using MLX backend |
| [Speculative Decoding](content/playbooks/speculative-decoding.md) | 15 min | Use draft models to accelerate generation 1.5–2.5× |
| [MLX VLM Inference](content/playbooks/mlx-vlm.md) | 15 min | Run vision-language models locally with MLX |
| [MLX Quantization](content/playbooks/mlx-quantization.md) | 10 min | Quantize any HF model to 2/4/8-bit for Apple Silicon |
| [Ollama API Serving](content/playbooks/nim-equivalent.md) | 5 min | Serve models via OpenAI-compatible API with Ollama |

### 🧪 Fine-tuning

| Playbook | Time | Description |
|---|---|---|
| [MLX LoRA Fine-tuning](content/playbooks/mlx-lora.md) | 30 min | Fine-tune LLMs with LoRA/QLoRA natively on Apple Silicon |
| [PyTorch MPS Fine-tuning](content/playbooks/pytorch-mps.md) | 1 hr | GPU-accelerated training using PyTorch Metal backend |
| [LLaMA Factory on macOS](content/playbooks/llama-factory.md) | 1 hr | Install and fine-tune models with LLaMA Factory on MPS |
| [FLUX LoRA Fine-tuning](content/playbooks/mlx-flux-finetune.md) | 1 hr | Fine-tune FLUX.1 image models with LoRA on Mac |

### 📊 Data Science

| Playbook | Time | Description |
|---|---|---|
| [MLX + Accelerate Data Science](content/playbooks/mlx-data-science.md) | 20 min | GPU-accelerated numerical computing on Apple Silicon |
| [JAX on Apple Silicon](content/playbooks/jax-apple.md) | 15 min | Run JAX with Metal GPU backend for scientific computing |
| [Portfolio Optimization](content/playbooks/portfolio-optimization.md) | 20 min | Convex optimization for portfolio allocation with cvxpy |
| [Single-cell RNA Sequencing](content/playbooks/scrna-seq.md) | 15 min | End-to-end scRNA-seq workflow with scanpy |

### 🎨 Image Generation

| Playbook | Time | Description |
|---|---|---|
| [FLUX / Stable Diffusion with MLX](content/playbooks/mlx-flux.md) | 15 min | Generate images with FLUX and SD models natively on Mac |
| [ComfyUI on macOS](content/playbooks/comfyui.md) | 45 min | Node-based image generation workflow with MPS backend |

### 🤖 Applications

| Playbook | Time | Description |
|---|---|---|
| [Multi-Agent Chatbot](content/playbooks/multi-agent.md) | 30 min | Build a multi-agent system with LangGraph + local Ollama |
| [RAG with LangChain + Ollama](content/playbooks/rag-langchain.md) | 30 min | Fully local retrieval-augmented generation pipeline |
| [Text to Knowledge Graph](content/playbooks/txt2kg.md) | 30 min | Extract knowledge triples with local LLMs and graph DBs |
| [Video Search & Summarization](content/playbooks/video-search.md) | 30 min | Transcribe and analyze video with Whisper + VLMs |

### 🦾 Robotics

| Playbook | Time | Description |
|---|---|---|
| [MuJoCo + Playground](content/playbooks/mujoco.md) | 30 min | Cross-platform robotics simulation — replaces Isaac Sim/Lab |

### 🔧 Tools

| Playbook | Time | Description |
|---|---|---|
| [VS Code](content/playbooks/vscode.md) | 5 min | Install and configure VS Code for ML development on Mac |
| [Vibe Coding with Continue.dev](content/playbooks/vibe-coding.md) | 15 min | Local AI coding assistant with Ollama + VS Code |
| [System Monitoring (asitop)](content/playbooks/monitoring.md) | 5 min | Monitor GPU, CPU, ANE, and memory on Apple Silicon |
| [Tailscale on macOS](content/playbooks/tailscale.md) | 5 min | Secure mesh networking for remote access |
| [macOS Network Sharing](content/playbooks/network-sharing.md) | 5 min | SSH, screen sharing, and remote access setup |

---

## 🔄 DGX Spark → Mac Mapping

| DGX Spark Playbook | Mac Equivalent | Notes |
|---|---|---|
| Ollama | [Ollama](content/playbooks/ollama.md) | Identical workflow on macOS |
| Open WebUI with Ollama | [Open WebUI](content/playbooks/open-webui.md) | Identical on macOS |
| vLLM for Inference | [MLX LM](content/playbooks/mlx-lm.md) | MLX is the Apple-native high-perf engine |
| TRT-LLM for Inference | [llama.cpp with Metal](content/playbooks/llama-cpp.md) | Metal replaces TensorRT |
| SGLang Inference Server | [LM Studio](content/playbooks/lm-studio.md) | Uses MLX backend natively |
| NIM on Spark | [Ollama API Serving](content/playbooks/nim-equivalent.md) | OpenAI-compatible local API |
| NVFP4 Quantization | [MLX Quantization](content/playbooks/mlx-quantization.md) | 2/4/8-bit quantization natively |
| Speculative Decoding | [Speculative Decoding](content/playbooks/speculative-decoding.md) | llama.cpp Metal backend |
| Multi-modal Inference | [MLX VLM Inference](content/playbooks/mlx-vlm.md) | Vision-language models via MLX |
| NeMo / Unsloth Fine-tune | [MLX LoRA Fine-tuning](content/playbooks/mlx-lora.md) | Unsloth is CUDA-only; MLX LoRA replaces it |
| PyTorch Fine-tune | [PyTorch MPS Fine-tuning](content/playbooks/pytorch-mps.md) | MPS backend for Apple GPU |
| LLaMA Factory | [LLaMA Factory on macOS](content/playbooks/llama-factory.md) | Works on macOS with MPS |
| FLUX.1 Dreambooth LoRA | [FLUX / SD with MLX](content/playbooks/mlx-flux.md) | Native FLUX and SD via mflux |
| FLUX.1 LoRA Fine-tuning | [FLUX LoRA Fine-tuning](content/playbooks/mlx-flux-finetune.md) | LoRA training via mflux |
| Comfy UI | [ComfyUI on macOS](content/playbooks/comfyui.md) | MPS backend |
| CUDA-X Data Science | [MLX + Accelerate](content/playbooks/mlx-data-science.md) | Accelerate + MLX for numerical compute |
| Optimized JAX | [JAX on Apple Silicon](content/playbooks/jax-apple.md) | JAX Metal backend |
| Multi-Agent Chatbot | [Multi-Agent Chatbot](content/playbooks/multi-agent.md) | LangGraph + Ollama |
| RAG in AI Workbench | [RAG with LangChain](content/playbooks/rag-langchain.md) | Fully local RAG pipeline |
| Text to Knowledge Graph | [Text to Knowledge Graph](content/playbooks/txt2kg.md) | Ollama + NetworkX |
| Video Search & Summarization | [Video Search](content/playbooks/video-search.md) | mlx-whisper + Ollama |
| Isaac Sim / Isaac Lab | [MuJoCo + Playground](content/playbooks/mujoco.md) | Cross-platform robotics sim |
| DGX Dashboard | [System Monitoring (asitop)](content/playbooks/monitoring.md) | GPU/ANE/CPU monitoring |
| VS Code | [VS Code](content/playbooks/vscode.md) | Native Apple Silicon build |
| Vibe Coding in VS Code | [Vibe Coding](content/playbooks/vibe-coding.md) | Continue.dev + local Ollama |
| Tailscale | [Tailscale on macOS](content/playbooks/tailscale.md) | Native macOS app |
| Set Up Local Network Access | [macOS Network Sharing](content/playbooks/network-sharing.md) | Built-in macOS features |
| Portfolio Optimization | [Portfolio Optimization](content/playbooks/portfolio-optimization.md) | Cross-platform Python stack |
| Single-cell RNA Sequencing | [scRNA-seq Analysis](content/playbooks/scrna-seq.md) | scanpy / AnnData |

---

## 💻 Hardware Recommendations

| Workload | Minimum | Recommended |
|---|---|---|
| Chat with 7B models | M1, 8 GB | M2+, 16 GB |
| Run 32B–70B models | M2 Pro, 32 GB | M2 Max/Ultra, 64 GB+ |
| Fine-tune 7B models | M2 Pro, 32 GB | M3 Max, 64 GB |
| Fine-tune 32B+ models | M2 Ultra, 96 GB | M4 Ultra, 192 GB |
| Image generation (FLUX) | M2, 16 GB | M3+, 32 GB |
| Robotics simulation | M1, 16 GB | M3 Pro+, 36 GB |
| Full-stack ML research | M2 Max, 64 GB | M4 Ultra, 192 GB |

---

## 🛠 Site Development

The portal is a Next.js static site deployed to GitHub Pages.

```bash
npm install
npm run dev       # → http://localhost:3000
npm run build     # → static export in out/
```

Playbook content lives in `content/playbooks/*.md` with YAML frontmatter and `<!-- tab: Label -->` tab markers.

---

## 📄 License

[Apache-2.0](LICENSE)

The original [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) are licensed under Apache-2.0 by NVIDIA Corporation.
