import { useState, useEffect, useRef } from "react";

// ─── Data ────────────────────────────────────────────────────────────────────

const categories = [
  {
    id: "onboarding",
    label: "onboarding",
    playbooks: [
      { slug: "ollama", title: "Ollama", time: "5 min", color: "green", desc: "Install and run LLMs locally with a single command", tags: ["inference"], spark: "Ollama" },
      { slug: "open-webui", title: "Open WebUI with Ollama", time: "15 min", color: "green", desc: "Deploy a full ChatGPT-like interface locally", tags: ["inference", "ui"], spark: "Open WebUI with Ollama" },
    ],
  },
  {
    id: "inference",
    label: "inference",
    playbooks: [
      { slug: "mlx-lm", title: "MLX LM for Inference", time: "10 min", color: "green", desc: "Apple's high-performance native LLM engine — fastest on Mac", tags: ["mlx", "inference"], spark: "vLLM for Inference" },
      { slug: "llama-cpp", title: "llama.cpp with Metal", time: "15 min", color: "green", desc: "Run GGUF models with first-class Apple Silicon Metal acceleration", tags: ["inference", "metal"], spark: "TRT-LLM / Nemotron with llama.cpp" },
      { slug: "lm-studio", title: "LM Studio on macOS", time: "5 min", color: "green", desc: "Polished GUI for local LLM inference using MLX backend", tags: ["inference", "ui"], spark: "LM Studio / SGLang" },
      { slug: "speculative-decoding", title: "Speculative Decoding", time: "15 min", color: "green", desc: "Use draft models to accelerate generation 1.5–2.5×", tags: ["inference", "optimization"], spark: "Speculative Decoding" },
      { slug: "mlx-vlm", title: "MLX VLM Inference", time: "15 min", color: "green", desc: "Run vision-language models locally with MLX", tags: ["mlx", "multimodal"], spark: "Multi-modal Inference" },
      { slug: "mlx-quantization", title: "MLX Quantization", time: "10 min", color: "green", desc: "Quantize any HF model to 2/4/8-bit for Apple Silicon", tags: ["mlx", "quantization"], spark: "NVFP4 Quantization" },
      { slug: "nim-equivalent", title: "Ollama API Serving", time: "5 min", color: "green", desc: "Serve models via OpenAI-compatible API with Ollama", tags: ["inference", "api"], spark: "NIM on Spark" },
    ],
  },
  {
    id: "fine-tuning",
    label: "fine tuning",
    playbooks: [
      { slug: "mlx-lora", title: "MLX LoRA Fine-tuning", time: "30 min", color: "orange", desc: "Fine-tune LLMs with LoRA/QLoRA natively on Apple Silicon", tags: ["mlx", "fine-tuning"], spark: "NeMo / Unsloth Fine-tune" },
      { slug: "pytorch-mps", title: "PyTorch MPS Fine-tuning", time: "1 hr", color: "red", desc: "GPU-accelerated training using PyTorch Metal backend", tags: ["pytorch", "fine-tuning"], spark: "Fine-tune with PyTorch" },
      { slug: "llama-factory", title: "LLaMA Factory on macOS", time: "1 hr", color: "red", desc: "Install and fine-tune models with LLaMA Factory on MPS", tags: ["fine-tuning", "ui"], spark: "LLaMA Factory" },
      { slug: "mlx-flux-finetune", title: "FLUX LoRA Fine-tuning", time: "1 hr", color: "red", desc: "Fine-tune FLUX.1 image models with LoRA on Mac", tags: ["image generation", "fine-tuning"], spark: "FLUX.1 Dreambooth LoRA" },
    ],
  },
  {
    id: "data-science",
    label: "data science",
    playbooks: [
      { slug: "mlx-data-science", title: "MLX + Accelerate Data Science", time: "20 min", color: "green", desc: "GPU-accelerated numerical computing on Apple Silicon", tags: ["data science", "mlx"], spark: "CUDA-X Data Science" },
      { slug: "jax-apple", title: "JAX on Apple Silicon", time: "15 min", color: "green", desc: "Run JAX with Metal GPU backend for scientific computing", tags: ["jax", "data science"], spark: "Optimized JAX" },
      { slug: "portfolio-optimization", title: "Portfolio Optimization", time: "20 min", color: "green", desc: "Convex optimization for portfolio allocation with cvxpy", tags: ["data science", "finance"], spark: "Portfolio Optimization" },
      { slug: "scrna-seq", title: "Single-cell RNA Sequencing", time: "15 min", color: "green", desc: "End-to-end scRNA-seq workflow with scanpy", tags: ["data science", "bioinformatics"], spark: "Single-cell RNA Sequencing" },
    ],
  },
  {
    id: "image-gen",
    label: "image generation",
    playbooks: [
      { slug: "mlx-flux", title: "FLUX / Stable Diffusion with MLX", time: "15 min", color: "green", desc: "Generate images with FLUX and SD models natively on Mac", tags: ["image generation", "mlx"], spark: "FLUX.1 Dreambooth LoRA" },
      { slug: "comfyui", title: "ComfyUI on macOS", time: "45 min", color: "orange", desc: "Node-based image generation workflow with MPS backend", tags: ["image generation", "ui"], spark: "Comfy UI" },
    ],
  },
  {
    id: "applications",
    label: "applications",
    playbooks: [
      { slug: "multi-agent", title: "Multi-Agent Chatbot", time: "30 min", color: "orange", desc: "Build a multi-agent system with LangGraph + local Ollama", tags: ["agents", "langchain"], spark: "Multi-Agent Chatbot" },
      { slug: "rag-langchain", title: "RAG with LangChain + Ollama", time: "30 min", color: "orange", desc: "Fully local retrieval-augmented generation pipeline", tags: ["rag", "langchain"], spark: "RAG in AI Workbench" },
      { slug: "txt2kg", title: "Text to Knowledge Graph", time: "30 min", color: "orange", desc: "Extract knowledge triples with local LLMs and graph DBs", tags: ["knowledge graph", "nlp"], spark: "Text to Knowledge Graph" },
      { slug: "video-search", title: "Video Search & Summarization", time: "30 min", color: "orange", desc: "Transcribe and analyze video with Whisper + VLMs", tags: ["video", "whisper"], spark: "Video Search & Summarization" },
    ],
  },
  {
    id: "robotics",
    label: "robotics",
    playbooks: [
      { slug: "mujoco", title: "MuJoCo + Playground", time: "30 min", color: "orange", desc: "Cross-platform robotics simulation — replaces Isaac Sim/Lab", tags: ["robotics", "simulation"], spark: "Isaac Sim / Isaac Lab" },
    ],
  },
  {
    id: "tools",
    label: "tools",
    playbooks: [
      { slug: "vscode", title: "VS Code", time: "5 min", color: "green", desc: "Install and configure VS Code for ML development on Mac", tags: ["tools"], spark: "VS Code" },
      { slug: "vibe-coding", title: "Vibe Coding with Continue.dev", time: "15 min", color: "green", desc: "Local AI coding assistant with Ollama + VS Code", tags: ["tools", "coding"], spark: "Vibe Coding in VS Code" },
      { slug: "monitoring", title: "System Monitoring (asitop)", time: "5 min", color: "green", desc: "Monitor GPU, CPU, ANE, and memory on Apple Silicon", tags: ["tools", "monitoring"], spark: "DGX Dashboard" },
      { slug: "tailscale", title: "Tailscale on macOS", time: "5 min", color: "green", desc: "Secure mesh networking for remote access", tags: ["tools", "networking"], spark: "Tailscale" },
      { slug: "network-sharing", title: "macOS Network Sharing", time: "5 min", color: "green", desc: "SSH, screen sharing, and remote access setup", tags: ["tools", "networking"], spark: "Set Up Local Network Access" },
    ],
  },
];

const allPlaybooks = categories.flatMap((c) => c.playbooks.map((p) => ({ ...p, category: c.label })));
const featured = ["ollama", "open-webui", "mlx-lm", "comfyui"];
const whatsNew = ["lm-studio", "mujoco"];

// ─── Playbook detail content (abbreviated for key playbooks) ─────────────────

const playbookDetails = {
  ollama: {
    overview: `Run large language models locally with a single command. Ollama wraps llama.cpp with a clean CLI and API, managing model downloads, quantization, and serving automatically.\n\nThis is the fastest way to get started with local AI on your Mac.`,
    tabs: [
      {
        id: "overview", label: "Overview",
        content: `## Basic idea\n\nOllama is the simplest path to running LLMs on your Mac. One install, one command, and you're chatting with a model.\n\n## What you'll accomplish\n\nYou will have Ollama running on your Mac with models accessible via CLI and API at localhost:11434.\n\n## Prerequisites\n\n- macOS 13.0+ (Ventura or later)\n- Apple Silicon Mac (M1 or later)\n- 8 GB+ unified memory (16 GB+ recommended)\n\n## Time & risk\n\n- **Duration:** 5 minutes\n- **Risk level:** None — standard app install\n- **Rollback:** brew uninstall ollama`,
      },
      {
        id: "install", label: "Install",
        content: `## Step 1: Install Ollama\n\n\`\`\`bash\nbrew install ollama\n\`\`\`\n\nOr download from https://ollama.com/download/mac\n\n## Step 2: Start the server\n\n\`\`\`bash\nollama serve\n\`\`\`\n\n## Step 3: Pull and run a model\n\n\`\`\`bash\nollama pull qwen2.5:7b\nollama run qwen2.5:7b\n\`\`\`\n\nFor machines with 32GB+:\n\n\`\`\`bash\nollama pull qwen2.5:32b\nollama run qwen2.5:32b\n\`\`\``,
      },
      {
        id: "api", label: "API Usage",
        content: `## Test the API\n\n\`\`\`bash\ncurl http://localhost:11434/api/chat -d '{\n  "model": "qwen2.5:7b",\n  "messages": [{\n    "role": "user",\n    "content": "Write me a haiku about Apple Silicon."\n  }],\n  "stream": false\n}'\n\`\`\`\n\n## Memory Guidelines\n\n| Unified Memory | Max Model Size |\n|---|---|\n| 8 GB | 7B Q4 |\n| 16 GB | 14B Q4 |\n| 32 GB | 32B Q4 |\n| 64 GB | 70B Q4 |\n| 96–192 GB | 120B+ Q4 |`,
      },
      {
        id: "troubleshooting", label: "Troubleshooting",
        content: `## Ollama is slow\n\nEnsure no other large processes are consuming memory. Check Activity Monitor → Memory.\n\n## Out of memory\n\nUse \`OLLAMA_MAX_LOADED_MODELS=1\` to avoid loading multiple models.\n\nUse a smaller quantized model.\n\n## Server won't start\n\nCheck if another instance is already running:\n\`\`\`bash\npkill ollama\nollama serve\n\`\`\``,
      },
    ],
  },
};

// ─── Minimal Markdown Renderer ───────────────────────────────────────────────

function MiniMarkdown({ text }) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let inCode = false;
  let codeLines = [];
  let inTable = false;
  let tableRows = [];

  const flushTable = () => {
    if (tableRows.length > 0) {
      elements.push(
        <div key={`t-${elements.length}`} style={{ overflowX: "auto", margin: "16px 0" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr>
                {tableRows[0].map((cell, i) => (
                  <th key={i} style={{ padding: "8px 12px", borderBottom: "2px solid rgba(255,255,255,0.15)", textAlign: "left", color: "rgba(255,255,255,0.7)", fontWeight: 600 }}>{cell}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.slice(2).map((row, ri) => (
                <tr key={ri}>
                  {row.map((cell, ci) => (
                    <td key={ci} style={{ padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.85)" }}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableRows = [];
      inTable = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("```")) {
      if (inCode) {
        elements.push(
          <pre key={`c-${elements.length}`} style={{ background: "rgba(0,0,0,0.4)", borderRadius: 10, padding: "16px 20px", margin: "12px 0", overflowX: "auto", fontSize: 13, lineHeight: 1.6, border: "1px solid rgba(255,255,255,0.08)" }}>
            <code style={{ color: "#e0e0e0", fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace" }}>{codeLines.join("\n")}</code>
          </pre>
        );
        codeLines = [];
        inCode = false;
      } else {
        flushTable();
        inCode = true;
      }
      continue;
    }
    if (inCode) { codeLines.push(line); continue; }
    if (line.startsWith("|")) {
      inTable = true;
      tableRows.push(line.split("|").filter(Boolean).map((c) => c.trim()));
      continue;
    }
    if (inTable) flushTable();
    if (line.startsWith("## ")) {
      elements.push(<h2 key={`h-${elements.length}`} style={{ fontSize: 20, fontWeight: 700, color: "#fff", marginTop: 28, marginBottom: 12, letterSpacing: "-0.02em" }}>{line.slice(3)}</h2>);
    } else if (line.startsWith("- **")) {
      const m = line.match(/- \*\*(.+?)\*\*:?\s*(.*)/);
      if (m) elements.push(<div key={`li-${elements.length}`} style={{ padding: "4px 0 4px 16px", color: "rgba(255,255,255,0.85)", fontSize: 15, lineHeight: 1.6 }}><span style={{ color: "#fff", fontWeight: 600 }}>{m[1]}</span>{m[2] ? ": " + m[2] : ""}</div>);
    } else if (line.startsWith("- ")) {
      elements.push(<div key={`li-${elements.length}`} style={{ padding: "3px 0 3px 16px", color: "rgba(255,255,255,0.8)", fontSize: 15, lineHeight: 1.6 }}>• {line.slice(2)}</div>);
    } else if (line.trim() === "") {
      elements.push(<div key={`sp-${elements.length}`} style={{ height: 8 }} />);
    } else {
      const processed = line.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-size:13px;font-family:SF Mono,monospace">$1</code>');
      elements.push(<p key={`p-${elements.length}`} style={{ color: "rgba(255,255,255,0.8)", fontSize: 15, lineHeight: 1.7, margin: "6px 0" }} dangerouslySetInnerHTML={{ __html: processed }} />);
    }
  }
  flushTable();
  return <>{elements}</>;
}

// ─── Components ──────────────────────────────────────────────────────────────

const colorMap = { green: "#34C759", orange: "#FF9F0A", red: "#FF453A" };

function PlaybookCard({ playbook, onClick }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={() => onClick(playbook)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? "rgba(255,255,255,0.07)" : "rgba(255,255,255,0.04)",
        border: "1px solid",
        borderColor: hovered ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.08)",
        borderRadius: 16,
        padding: "24px 24px 20px",
        cursor: "pointer",
        transition: "all 0.25s cubic-bezier(0.4,0,0.2,1)",
        transform: hovered ? "translateY(-2px)" : "none",
        display: "flex",
        flexDirection: "column",
        gap: 10,
        minHeight: 160,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 36, height: 36, borderRadius: 10, background: "linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03))", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, flexShrink: 0 }}>
            🍎
          </div>
          <span style={{ color: "rgba(255,255,255,0.45)", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Mac Playbook</span>
        </div>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.7)" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: colorMap[playbook.color] || colorMap.green }} />
          {playbook.time}
        </span>
      </div>
      <div style={{ fontSize: 17, fontWeight: 700, color: "#fff", letterSpacing: "-0.02em", lineHeight: 1.3 }}>{playbook.title}</div>
      <div style={{ fontSize: 14, color: "rgba(255,255,255,0.55)", lineHeight: 1.5, flex: 1 }}>{playbook.desc}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4 }}>
        {playbook.tags.slice(0, 3).map((t) => (
          <span key={t} style={{ fontSize: 11, padding: "3px 10px", borderRadius: 20, background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.5)", border: "1px solid rgba(255,255,255,0.06)" }}>{t}</span>
        ))}
      </div>
    </div>
  );
}

function QuickstartCard({ playbook, onClick }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={() => onClick(playbook)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex", alignItems: "center", gap: 16, padding: "16px 20px",
        background: hovered ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 14, cursor: "pointer",
        transition: "all 0.2s ease",
      }}
    >
      <div style={{ width: 42, height: 42, borderRadius: 12, background: "linear-gradient(135deg, #1a1a2e, #16213e)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20, flexShrink: 0 }}>⚡</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 15, fontWeight: 700, color: "#fff" }}>{playbook.title}</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,0.6)" }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: colorMap[playbook.color] }} />
            {playbook.time}
          </span>
        </div>
        <div style={{ fontSize: 13, color: "rgba(255,255,255,0.45)", marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{playbook.desc}</div>
      </div>
      <div style={{ color: "rgba(255,255,255,0.25)", fontSize: 18 }}>→</div>
    </div>
  );
}

// ─── Playbook Detail Page ────────────────────────────────────────────────────

function PlaybookPage({ playbook, onBack }) {
  const detail = playbookDetails[playbook.slug];
  const [activeTab, setActiveTab] = useState(0);
  const tabs = detail?.tabs || [
    { id: "overview", label: "Overview", content: `## Basic idea\n\n${playbook.desc}\n\n## Replaces on DGX Spark\n\n${playbook.spark}\n\n## Prerequisites\n\n- macOS 14.0+ (Sonoma or later)\n- Apple Silicon Mac (M1 or later)\n- Python 3.10+\n\n## Time & risk\n\n- **Duration:** ${playbook.time}\n- **Risk level:** Low` },
    { id: "instructions", label: "Instructions", content: `## Getting Started\n\nDetailed step-by-step instructions for **${playbook.title}** will be generated when this playbook is expanded.\n\nUse Claude Code with the mac-playbooks README.md to generate the full content for this playbook.\n\n\`\`\`bash\n# Placeholder — expand with Claude Code\necho "Setting up ${playbook.title}..."\n\`\`\`` },
    { id: "troubleshooting", label: "Troubleshooting", content: `## Common Issues\n\nTroubleshooting content for **${playbook.title}** will be populated when this playbook is fully implemented.` },
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0f" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", gap: 0 }}>
        {/* Sidebar */}
        <div style={{ width: 280, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.06)", minHeight: "100vh", paddingTop: 80, position: "sticky", top: 0, alignSelf: "flex-start", overflowY: "auto", maxHeight: "100vh" }}>
          <div style={{ padding: "16px 20px" }}>
            <div onClick={onBack} style={{ cursor: "pointer", fontSize: 13, color: "rgba(255,255,255,0.4)", marginBottom: 24, display: "flex", alignItems: "center", gap: 6 }}>
              ← View All Playbooks
            </div>
            {categories.map((cat) => (
              <div key={cat.id} style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "rgba(255,255,255,0.3)", marginBottom: 8, padding: "0 4px" }}>{cat.label}</div>
                {cat.playbooks.map((p) => (
                  <div
                    key={p.slug}
                    onClick={() => onBack(p)}
                    style={{
                      padding: "7px 12px",
                      borderRadius: 8,
                      fontSize: 13,
                      color: p.slug === playbook.slug ? "#fff" : "rgba(255,255,255,0.55)",
                      background: p.slug === playbook.slug ? "rgba(255,255,255,0.08)" : "transparent",
                      cursor: "pointer",
                      marginBottom: 2,
                      fontWeight: p.slug === playbook.slug ? 600 : 400,
                      lineHeight: 1.4,
                    }}
                  >
                    {p.title}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Content */}
        <div style={{ flex: 1, padding: "100px 48px 80px" }}>
          <h1 style={{ fontSize: 36, fontWeight: 800, color: "#fff", letterSpacing: "-0.03em", marginBottom: 6 }}>{playbook.title}</h1>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 32 }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.6)" }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: colorMap[playbook.color] }} />
              {playbook.time}
            </span>
            <span style={{ color: "rgba(255,255,255,0.25)" }}>·</span>
            <span style={{ fontSize: 13, color: "rgba(255,255,255,0.4)" }}>{playbook.desc}</span>
          </div>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", marginBottom: 24 }}>
            Replaces DGX Spark: <span style={{ color: "rgba(255,255,255,0.5)" }}>{playbook.spark}</span>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 0, borderBottom: "1px solid rgba(255,255,255,0.08)", marginBottom: 32 }}>
            {tabs.map((tab, idx) => (
              <div
                key={tab.id}
                onClick={() => setActiveTab(idx)}
                style={{
                  padding: "12px 20px",
                  fontSize: 14,
                  fontWeight: activeTab === idx ? 700 : 500,
                  color: activeTab === idx ? "#fff" : "rgba(255,255,255,0.4)",
                  borderBottom: activeTab === idx ? "2px solid #0A84FF" : "2px solid transparent",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {tab.label}
              </div>
            ))}
          </div>

          <MiniMarkdown text={tabs[activeTab]?.content || ""} />
        </div>
      </div>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────

export default function MacPlaybooks() {
  const [view, setView] = useState("home"); // "home" | "playbook"
  const [selectedPlaybook, setSelectedPlaybook] = useState(null);
  const [search, setSearch] = useState("");
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const openPlaybook = (pb) => {
    setSelectedPlaybook(pb);
    setView("playbook");
    window.scrollTo(0, 0);
  };

  const goHome = (maybePb) => {
    if (maybePb?.slug) {
      setSelectedPlaybook(maybePb);
      setView("playbook");
      window.scrollTo(0, 0);
    } else {
      setView("home");
      window.scrollTo(0, 0);
    }
  };

  const filtered = search.trim()
    ? allPlaybooks.filter((p) =>
        (p.title + " " + p.desc + " " + p.tags.join(" ") + " " + p.spark).toLowerCase().includes(search.toLowerCase())
      )
    : null;

  // ─── Nav ───
  const navBg = scrollY > 40 ? "rgba(10,10,15,0.85)" : "transparent";
  const navBorder = scrollY > 40 ? "rgba(255,255,255,0.06)" : "transparent";

  const Nav = (
    <nav style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
      background: navBg, backdropFilter: scrollY > 40 ? "blur(20px) saturate(180%)" : "none",
      borderBottom: `1px solid ${navBorder}`,
      transition: "all 0.3s ease",
    }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 32px", height: 56, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <div onClick={goHome} style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 22 }}>🍎</span>
            <span style={{ fontSize: 16, fontWeight: 800, color: "#fff", letterSpacing: "-0.03em" }}>mac-playbooks</span>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {["Playbooks", "Docs", "GitHub"].map((l) => (
              <span key={l} style={{ padding: "6px 12px", fontSize: 13, color: "rgba(255,255,255,0.5)", cursor: "pointer", borderRadius: 8, fontWeight: 500 }}>{l}</span>
            ))}
          </div>
        </div>
        <div style={{ position: "relative" }}>
          <input
            value={search}
            onChange={(e) => { setSearch(e.target.value); if (view !== "home") setView("home"); }}
            placeholder="Search playbooks..."
            style={{
              width: 220, padding: "7px 14px 7px 34px", borderRadius: 10,
              background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)",
              color: "#fff", fontSize: 13, outline: "none",
              fontFamily: "inherit",
            }}
          />
          <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "rgba(255,255,255,0.3)", fontSize: 13 }}>⌘</span>
        </div>
      </div>
    </nav>
  );

  if (view === "playbook" && selectedPlaybook) {
    return (
      <div style={{ fontFamily: "'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif", color: "#fff", background: "#0a0a0f" }}>
        {Nav}
        <PlaybookPage playbook={selectedPlaybook} onBack={goHome} />
      </div>
    );
  }

  // ─── Home ───
  const featuredPlaybooks = allPlaybooks.filter((p) => featured.includes(p.slug));
  const newPlaybooks = allPlaybooks.filter((p) => whatsNew.includes(p.slug));

  return (
    <div style={{ fontFamily: "'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif", color: "#fff", background: "#0a0a0f", minHeight: "100vh" }}>
      {Nav}

      {/* Hero */}
      <div style={{
        position: "relative", overflow: "hidden",
        padding: "120px 32px 64px", textAlign: "center",
        background: "radial-gradient(ellipse at 50% 0%, rgba(10,132,255,0.08) 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(48,209,88,0.05) 0%, transparent 50%)",
      }}>
        <div style={{ position: "absolute", inset: 0, background: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E\")", opacity: 0.5 }} />
        <h1 style={{ fontSize: 48, fontWeight: 900, letterSpacing: "-0.04em", lineHeight: 1.1, marginBottom: 16, position: "relative" }}>
          Start Building on{" "}
          <span style={{ background: "linear-gradient(135deg, #0A84FF, #30D158, #5AC8FA)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Mac</span>
        </h1>
        <p style={{ fontSize: 18, color: "rgba(255,255,255,0.5)", maxWidth: 600, margin: "0 auto", lineHeight: 1.6, position: "relative", fontWeight: 400 }}>
          Find instructions and examples to run AI workloads on Apple Silicon.
          <br />
          The macOS counterpart to NVIDIA DGX Spark Playbooks.
        </p>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 32px" }}>

        {/* Search results */}
        {filtered ? (
          <div style={{ marginBottom: 64 }}>
            <h2 style={{ fontSize: 14, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "rgba(255,255,255,0.3)", marginBottom: 20 }}>
              {filtered.length} result{filtered.length !== 1 ? "s" : ""} for "{search}"
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 16 }}>
              {filtered.map((p) => <PlaybookCard key={p.slug} playbook={p} onClick={openPlaybook} />)}
            </div>
            {filtered.length === 0 && <p style={{ color: "rgba(255,255,255,0.4)", marginTop: 20 }}>No playbooks match your search. Try different keywords.</p>}
          </div>
        ) : (
          <>
            {/* Quick starts + First time */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, marginBottom: 64 }}>
              <div>
                <h2 style={{ fontSize: 22, fontWeight: 800, color: "#fff", marginBottom: 6, letterSpacing: "-0.02em" }}>Quick Start</h2>
                <p style={{ fontSize: 14, color: "rgba(255,255,255,0.4)", marginBottom: 20 }}>Get up and running in 5 minutes</p>
                <div style={{
                  borderRadius: 20, overflow: "hidden",
                  background: "linear-gradient(135deg, rgba(10,132,255,0.12), rgba(48,209,88,0.08))",
                  border: "1px solid rgba(255,255,255,0.06)",
                  padding: 32,
                }}>
                  <pre style={{ fontSize: 13, lineHeight: 1.8, color: "rgba(255,255,255,0.75)", fontFamily: "'SF Mono', 'Fira Code', monospace", margin: 0, whiteSpace: "pre-wrap" }}>
{`# Install Ollama
brew install ollama && ollama serve &

# Pull a model and chat
ollama pull qwen2.5:7b
ollama run qwen2.5:7b

# Install MLX for max performance
pip install mlx-lm
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
  --prompt "Hello, Mac!"`}
                  </pre>
                </div>
              </div>
              <div>
                <h2 style={{ fontSize: 22, fontWeight: 800, color: "#fff", marginBottom: 6, letterSpacing: "-0.02em" }}>First Time Here?</h2>
                <p style={{ fontSize: 14, color: "rgba(255,255,255,0.4)", marginBottom: 20 }}>Try these developer quickstarts</p>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {featuredPlaybooks.map((p) => <QuickstartCard key={p.slug} playbook={p} onClick={openPlaybook} />)}
                </div>
              </div>
            </div>

            {/* What's New */}
            <div style={{ marginBottom: 64 }}>
              <h2 style={{ fontSize: 22, fontWeight: 800, color: "#fff", marginBottom: 6, letterSpacing: "-0.02em" }}>What's New</h2>
              <p style={{ fontSize: 14, color: "rgba(255,255,255,0.4)", marginBottom: 20 }}>Recently added playbooks</p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {newPlaybooks.map((p) => <PlaybookCard key={p.slug} playbook={p} onClick={openPlaybook} />)}
              </div>
            </div>

            {/* All Playbooks */}
            <div style={{ marginBottom: 80 }}>
              <h2 style={{ fontSize: 22, fontWeight: 800, color: "#fff", marginBottom: 6, letterSpacing: "-0.02em" }}>All Playbooks</h2>
              <p style={{ fontSize: 14, color: "rgba(255,255,255,0.4)", marginBottom: 24 }}>Detailed instructions to set up and run popular AI workflows on Mac</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
                {allPlaybooks.map((p) => <PlaybookCard key={p.slug} playbook={p} onClick={openPlaybook} />)}
              </div>
            </div>

            {/* Footer */}
            <div style={{
              padding: "32px 0", borderTop: "1px solid rgba(255,255,255,0.06)",
              display: "flex", justifyContent: "space-between", alignItems: "center",
              marginBottom: 32,
            }}>
              <span style={{ fontSize: 13, color: "rgba(255,255,255,0.25)" }}>
                mac-playbooks · macOS counterpart to NVIDIA DGX Spark Playbooks
              </span>
              <span style={{ fontSize: 13, color: "rgba(255,255,255,0.25)" }}>Apache-2.0</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
