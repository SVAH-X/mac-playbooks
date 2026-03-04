"use client";

import { useState } from "react";
import type { Playbook, CategoryGroup } from "@/lib/playbook-types";
import { groupByCategory } from "@/lib/playbook-types";
import Nav from "./Nav";
import PlaybookCard from "./PlaybookCard";
import QuickstartCard from "./QuickstartCard";
import Link from "next/link";

const CATEGORY_ICONS: Record<string, string> = {
  onboarding: "🚀",
  inference: "⚡",
  "fine-tuning": "🧪",
  "data-science": "📊",
  "image-gen": "🎨",
  applications: "🤖",
  robotics: "🦾",
  tools: "🔧",
};

interface MacPlaybooksHomeProps {
  playbooks: Playbook[];
}

export default function MacPlaybooksHome({ playbooks }: MacPlaybooksHomeProps) {
  const [search, setSearch] = useState("");
  const [activeCategory, setActiveCategory] = useState<string>("all");

  const categories = groupByCategory(playbooks);
  const featured = playbooks.filter((p) => p.featured);
  const whatsNew = playbooks.filter((p) => p.whatsNew);

  const filtered = search.trim()
    ? playbooks.filter((p) =>
        (p.title + " " + p.desc + " " + p.tags.join(" ") + " " + p.spark)
          .toLowerCase()
          .includes(search.toLowerCase())
      )
    : null;

  const filteredCategories: CategoryGroup[] =
    activeCategory === "all"
      ? categories
      : categories.filter((c) => c.id === activeCategory);

  const allCategoryTabs = [
    { id: "all", label: "All", icon: "🍎" },
    ...categories.map((c) => ({
      id: c.id,
      label: c.label,
      icon: CATEGORY_ICONS[c.id] ?? "📄",
    })),
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0f" }}>
      <Nav search={search} onSearchChange={setSearch} />

      {/* Hero */}
      <div
        style={{
          position: "relative",
          overflow: "hidden",
          padding: "110px 32px 56px",
          textAlign: "center",
          background:
            "radial-gradient(ellipse at 50% -10%, rgba(10,132,255,0.10) 0%, transparent 60%), radial-gradient(ellipse at 80% 30%, rgba(48,209,88,0.06) 0%, transparent 50%)",
        }}
      >
        {/* Badge above title */}
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            padding: "5px 14px",
            borderRadius: 20,
            background: "rgba(10,132,255,0.12)",
            border: "1px solid rgba(10,132,255,0.25)",
            fontSize: 12,
            fontWeight: 600,
            color: "#0A84FF",
            marginBottom: 20,
            letterSpacing: "0.02em",
          }}
        >
          🍎 macOS counterpart to NVIDIA DGX Spark Playbooks
        </div>
        <h1
          style={{
            fontSize: "clamp(32px, 5vw, 48px)",
            fontWeight: 900,
            letterSpacing: "-0.04em",
            lineHeight: 1.1,
            marginBottom: 16,
          }}
        >
          Start Building on{" "}
          <span
            style={{
              background: "linear-gradient(135deg, #0A84FF, #30D158, #5AC8FA)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Mac
          </span>
        </h1>
        <p
          style={{
            fontSize: "clamp(15px, 2vw, 18px)",
            color: "rgba(255,255,255,0.5)",
            maxWidth: 580,
            margin: "0 auto 28px",
            lineHeight: 1.6,
            fontWeight: 400,
          }}
        >
          Find instructions and examples to run AI workloads on Apple Silicon.
          <br />
          Optimized for M-series · Unified Memory · Metal GPU Acceleration.
        </p>

        {/* Stat chips */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            flexWrap: "wrap",
            gap: 10,
          }}
        >
          {[
            { label: "29 Playbooks", icon: "📚" },
            { label: "8 Categories", icon: "🗂️" },
            { label: "Apple Silicon", icon: "⚡" },
            { label: "100% Local", icon: "🔒" },
          ].map(({ label, icon }) => (
            <span
              key={label}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                padding: "5px 13px",
                borderRadius: 20,
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.09)",
                fontSize: 12,
                fontWeight: 500,
                color: "rgba(255,255,255,0.5)",
              }}
            >
              {icon} {label}
            </span>
          ))}
        </div>
      </div>

      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          padding: "0 24px",
        }}
      >
        {/* Search results */}
        {filtered ? (
          <div style={{ marginBottom: 64 }}>
            <h2
              style={{
                fontSize: 13,
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "rgba(255,255,255,0.3)",
                marginBottom: 20,
              }}
            >
              {filtered.length} result{filtered.length !== 1 ? "s" : ""} for
              &ldquo;{search}&rdquo;
            </h2>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
                gap: 16,
              }}
            >
              {filtered.map((p) => (
                <PlaybookCard key={p.slug} playbook={p} />
              ))}
            </div>
            {filtered.length === 0 && (
              <p style={{ color: "rgba(255,255,255,0.4)", marginTop: 20 }}>
                No playbooks match your search.
              </p>
            )}
          </div>
        ) : (
          <>
            {/* ── Access Remotely + First Time Here ── */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 28,
                marginBottom: 56,
              }}
            >
              {/* Access Remotely card */}
              <div>
                <h2
                  style={{
                    fontSize: 20,
                    fontWeight: 800,
                    color: "#fff",
                    marginBottom: 4,
                    letterSpacing: "-0.02em",
                  }}
                >
                  Access Remotely
                </h2>
                <p
                  style={{
                    fontSize: 13,
                    color: "rgba(255,255,255,0.4)",
                    marginBottom: 16,
                  }}
                >
                  Connect to your Mac from anywhere
                </p>
                <Link href="/playbooks/tailscale" style={{ textDecoration: "none" }}>
                  <div
                    style={{
                      borderRadius: 18,
                      background:
                        "linear-gradient(135deg, rgba(10,132,255,0.14), rgba(48,209,88,0.10))",
                      border: "1px solid rgba(255,255,255,0.08)",
                      padding: "28px 28px 24px",
                      cursor: "pointer",
                      transition: "border-color 0.2s",
                      height: "calc(100% - 56px)",
                      boxSizing: "border-box",
                      display: "flex",
                      flexDirection: "column",
                      gap: 16,
                    }}
                  >
                    <div style={{ fontSize: 36 }}>🌐</div>
                    <div>
                      <div
                        style={{
                          fontSize: 16,
                          fontWeight: 700,
                          color: "#fff",
                          marginBottom: 8,
                        }}
                      >
                        Set Up Remote Access
                      </div>
                      <div
                        style={{ fontSize: 13, color: "rgba(255,255,255,0.5)", lineHeight: 1.6 }}
                      >
                        Tailscale creates a secure mesh VPN to access your Mac and its AI models
                        from any device, anywhere.
                      </div>
                    </div>
                    <div
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                        fontSize: 13,
                        fontWeight: 600,
                        color: "#0A84FF",
                        marginTop: "auto",
                      }}
                    >
                      Configure Now →
                    </div>
                  </div>
                </Link>
              </div>

              {/* First Time Here */}
              <div>
                <h2
                  style={{
                    fontSize: 20,
                    fontWeight: 800,
                    color: "#fff",
                    marginBottom: 4,
                    letterSpacing: "-0.02em",
                  }}
                >
                  First Time Here?
                </h2>
                <p
                  style={{
                    fontSize: 13,
                    color: "rgba(255,255,255,0.4)",
                    marginBottom: 16,
                  }}
                >
                  Try these developer quickstarts
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {featured.map((p) => (
                    <QuickstartCard key={p.slug} playbook={p} />
                  ))}
                </div>
              </div>
            </div>

            {/* ── What's New ── */}
            {whatsNew.length > 0 && (
              <div style={{ marginBottom: 56 }}>
                <h2
                  style={{
                    fontSize: 20,
                    fontWeight: 800,
                    color: "#fff",
                    marginBottom: 4,
                    letterSpacing: "-0.02em",
                  }}
                >
                  {"What's New"}
                </h2>
                <p
                  style={{
                    fontSize: 13,
                    color: "rgba(255,255,255,0.4)",
                    marginBottom: 20,
                  }}
                >
                  Recently added playbooks
                </p>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
                    gap: 16,
                  }}
                >
                  {whatsNew.map((p) => (
                    <PlaybookCard key={p.slug} playbook={p} isNew />
                  ))}
                </div>
              </div>
            )}

            {/* ── All Playbooks ── */}
            <div style={{ marginBottom: 80 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  justifyContent: "space-between",
                  marginBottom: 4,
                }}
              >
                <h2
                  style={{
                    fontSize: 20,
                    fontWeight: 800,
                    color: "#fff",
                    letterSpacing: "-0.02em",
                  }}
                >
                  All Playbooks
                </h2>
                <span style={{ fontSize: 13, color: "rgba(255,255,255,0.25)" }}>
                  {playbooks.length} playbooks
                </span>
              </div>
              <p
                style={{
                  fontSize: 13,
                  color: "rgba(255,255,255,0.4)",
                  marginBottom: 20,
                }}
              >
                Step-by-step instructions for AI workloads on Apple Silicon
              </p>

              {/* Category filter tabs */}
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 8,
                  marginBottom: 28,
                  paddingBottom: 20,
                  borderBottom: "1px solid rgba(255,255,255,0.06)",
                }}
              >
                {allCategoryTabs.map((tab) => {
                  const active = activeCategory === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveCategory(tab.id)}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                        padding: "7px 14px",
                        borderRadius: 20,
                        fontSize: 12,
                        fontWeight: 600,
                        cursor: "pointer",
                        transition: "all 0.15s ease",
                        background: active
                          ? "rgba(10,132,255,0.15)"
                          : "rgba(255,255,255,0.04)",
                        border: active
                          ? "1px solid rgba(10,132,255,0.4)"
                          : "1px solid rgba(255,255,255,0.08)",
                        color: active ? "#0A84FF" : "rgba(255,255,255,0.55)",
                        fontFamily: "inherit",
                      }}
                    >
                      <span style={{ fontSize: 14 }}>{tab.icon}</span>
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              {/* Cards grid grouped by category */}
              {filteredCategories.map((cat) => (
                <div key={cat.id} style={{ marginBottom: 40 }}>
                  {activeCategory === "all" && (
                    <div
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        textTransform: "uppercase",
                        letterSpacing: "0.1em",
                        color: "rgba(255,255,255,0.28)",
                        marginBottom: 14,
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                      }}
                    >
                      <span style={{ fontSize: 14 }}>{CATEGORY_ICONS[cat.id] ?? "📄"}</span>
                      {cat.label}
                    </div>
                  )}
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
                      gap: 14,
                    }}
                  >
                    {cat.playbooks.map((p) => (
                      <PlaybookCard key={p.slug} playbook={p} />
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Footer */}
            <div
              style={{
                padding: "32px 0 40px",
                borderTop: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  flexWrap: "wrap",
                  gap: 24,
                  marginBottom: 24,
                }}
              >
                <div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 8,
                    }}
                  >
                    <span style={{ fontSize: 18 }}>🍎</span>
                    <span
                      style={{
                        fontSize: 15,
                        fontWeight: 800,
                        color: "#fff",
                        letterSpacing: "-0.02em",
                      }}
                    >
                      mac-playbooks
                    </span>
                  </div>
                  <p
                    style={{
                      fontSize: 12,
                      color: "rgba(255,255,255,0.3)",
                      maxWidth: 320,
                      lineHeight: 1.6,
                    }}
                  >
                    macOS counterpart to NVIDIA DGX Spark Playbooks. Step-by-step
                    AI/ML playbooks for Apple Silicon.
                  </p>
                </div>
                <div style={{ display: "flex", gap: 32 }}>
                  <div>
                    <div
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        textTransform: "uppercase",
                        letterSpacing: "0.08em",
                        color: "rgba(255,255,255,0.25)",
                        marginBottom: 10,
                      }}
                    >
                      Resources
                    </div>
                    {[
                      { label: "GitHub", href: "https://github.com/SVAH-X/mac-playbooks" },
                      { label: "MLX Docs", href: "https://ml-explore.github.io/mlx/" },
                      { label: "Ollama", href: "https://ollama.com" },
                      { label: "DGX Spark Playbooks", href: "https://github.com/NVIDIA/dgx-spark-playbooks" },
                    ].map(({ label, href }) => (
                      <a
                        key={label}
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          display: "block",
                          fontSize: 13,
                          color: "rgba(255,255,255,0.4)",
                          textDecoration: "none",
                          marginBottom: 6,
                        }}
                      >
                        {label}
                      </a>
                    ))}
                  </div>
                </div>
              </div>
              <div
                style={{
                  paddingTop: 20,
                  borderTop: "1px solid rgba(255,255,255,0.04)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.18)" }}>
                  © 2025 mac-playbooks contributors
                </span>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.18)" }}>
                  Apache-2.0 License
                </span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
