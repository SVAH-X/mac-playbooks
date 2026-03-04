"use client";

import { useState } from "react";
import Link from "next/link";
import type { Playbook } from "@/lib/playbook-types";
import { groupByCategory } from "@/lib/playbook-types";
import Nav from "./Nav";
import MiniMarkdown from "./MiniMarkdown";

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

const BADGE_COLORS: Record<string, { bg: string; color: string }> = {
  green: { bg: "rgba(45,212,191,0.15)", color: "#2dd4bf" },
  orange: { bg: "rgba(245,158,11,0.15)", color: "#f59e0b" },
  red: { bg: "rgba(239,68,68,0.15)", color: "#ef4444" },
};

interface PlaybookPageProps {
  playbook: Playbook;
  allPlaybooks: Playbook[];
}

export default function PlaybookPage({ playbook, allPlaybooks }: PlaybookPageProps) {
  const [activeTab, setActiveTab] = useState(0);
  const categories = groupByCategory(allPlaybooks);
  const badge = BADGE_COLORS[playbook.color] ?? BADGE_COLORS.green;
  const icon = CATEGORY_ICONS[playbook.category] ?? "📄";

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0f" }}>
      <Nav />

      <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex" }}>
        {/* ── Sidebar ── */}
        <aside
          style={{
            width: 260,
            flexShrink: 0,
            borderRight: "1px solid rgba(255,255,255,0.06)",
            minHeight: "100vh",
            paddingTop: 72,
            position: "sticky",
            top: 0,
            alignSelf: "flex-start",
            overflowY: "auto",
            maxHeight: "100vh",
          }}
        >
          <div style={{ padding: "16px 16px 32px" }}>
            <Link
              href="/"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                fontSize: 12,
                color: "rgba(255,255,255,0.35)",
                marginBottom: 24,
                textDecoration: "none",
                padding: "6px 10px",
                borderRadius: 8,
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.07)",
              }}
            >
              ← All Playbooks
            </Link>

            {categories.map((cat) => (
              <div key={cat.id} style={{ marginBottom: 22 }}>
                <div
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    color: "rgba(255,255,255,0.28)",
                    marginBottom: 6,
                    padding: "0 6px",
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}
                >
                  <span style={{ fontSize: 12 }}>{CATEGORY_ICONS[cat.id] ?? ""}</span>
                  {cat.label}
                </div>
                {cat.playbooks.map((p) => {
                  const active = p.slug === playbook.slug;
                  return (
                    <Link
                      key={p.slug}
                      href={`/playbooks/${p.slug}`}
                      style={{
                        display: "block",
                        padding: "6px 10px",
                        borderRadius: 8,
                        fontSize: 12,
                        color: active ? "#fff" : "rgba(255,255,255,0.48)",
                        background: active ? "rgba(10,132,255,0.12)" : "transparent",
                        borderLeft: active ? "2px solid #0A84FF" : "2px solid transparent",
                        textDecoration: "none",
                        marginBottom: 1,
                        fontWeight: active ? 600 : 400,
                        lineHeight: 1.45,
                        transition: "all 0.15s ease",
                      }}
                    >
                      {p.title}
                    </Link>
                  );
                })}
              </div>
            ))}
          </div>
        </aside>

        {/* ── Main content ── */}
        <main style={{ flex: 1, padding: "96px 44px 80px", minWidth: 0 }}>
          {/* Header */}
          <div style={{ marginBottom: 32 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 16,
              }}
            >
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: 13,
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 24,
                  flexShrink: 0,
                }}
              >
                {icon}
              </div>
              <div>
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    color: "rgba(255,255,255,0.35)",
                    marginBottom: 3,
                  }}
                >
                  Mac Playbook
                </div>
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 5,
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: "0.04em",
                    padding: "3px 9px",
                    borderRadius: 20,
                    background: badge.bg,
                    color: badge.color,
                    border: `1px solid ${badge.color}33`,
                    textTransform: "uppercase",
                  }}
                >
                  ⏱ {playbook.time}
                </span>
              </div>
            </div>

            <h1
              style={{
                fontSize: "clamp(24px, 3vw, 34px)",
                fontWeight: 800,
                color: "#fff",
                letterSpacing: "-0.03em",
                marginBottom: 8,
                lineHeight: 1.2,
              }}
            >
              {playbook.title}
            </h1>
            <p style={{ fontSize: 15, color: "rgba(255,255,255,0.45)", lineHeight: 1.6 }}>
              {playbook.desc}
            </p>

            <div
              style={{
                fontSize: 11,
                color: "rgba(255,255,255,0.25)",
                marginTop: 12,
              }}
            >
              Replaces DGX Spark:{" "}
              <span style={{ color: "rgba(255,255,255,0.45)" }}>{playbook.spark}</span>
            </div>

            {/* Tags */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 16 }}>
              {playbook.tags.map((t) => (
                <span
                  key={t}
                  style={{
                    fontSize: 11,
                    padding: "3px 10px",
                    borderRadius: 20,
                    background: "rgba(255,255,255,0.05)",
                    color: "rgba(255,255,255,0.4)",
                    border: "1px solid rgba(255,255,255,0.08)",
                  }}
                >
                  {t}
                </span>
              ))}
            </div>
          </div>

          {/* Tabs */}
          <div
            style={{
              display: "flex",
              gap: 2,
              borderBottom: "1px solid rgba(255,255,255,0.07)",
              marginBottom: 32,
            }}
          >
            {playbook.tabs.map((tab, idx) => (
              <button
                key={tab.label}
                onClick={() => setActiveTab(idx)}
                style={{
                  padding: "10px 18px",
                  fontSize: 13,
                  fontWeight: activeTab === idx ? 700 : 500,
                  color: activeTab === idx ? "#fff" : "rgba(255,255,255,0.38)",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                  background: "transparent",
                  border: "none",
                  borderBottom: activeTab === idx
                    ? "2px solid #0A84FF"
                    : "2px solid transparent",
                  fontFamily: "inherit",
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <MiniMarkdown text={playbook.tabs[activeTab]?.content ?? ""} />
        </main>
      </div>
    </div>
  );
}
