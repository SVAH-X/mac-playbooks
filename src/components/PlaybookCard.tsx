"use client";

import { useState } from "react";
import Link from "next/link";
import type { Playbook } from "@/lib/playbook-types";

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

const CATEGORY_COLORS: Record<string, string> = {
  onboarding: "rgba(10,132,255,0.15)",
  inference: "rgba(48,209,88,0.15)",
  "fine-tuning": "rgba(255,159,10,0.15)",
  "data-science": "rgba(90,200,250,0.15)",
  "image-gen": "rgba(191,90,242,0.15)",
  applications: "rgba(255,149,0,0.15)",
  robotics: "rgba(255,69,58,0.15)",
  tools: "rgba(99,99,102,0.2)",
};

const BADGE_COLORS: Record<string, { bg: string; color: string }> = {
  green: { bg: "rgba(45,212,191,0.15)", color: "#2dd4bf" },
  orange: { bg: "rgba(245,158,11,0.15)", color: "#f59e0b" },
  red: { bg: "rgba(239,68,68,0.15)", color: "#ef4444" },
};

interface PlaybookCardProps {
  playbook: Playbook;
  isNew?: boolean;
}

export default function PlaybookCard({ playbook, isNew }: PlaybookCardProps) {
  const [hovered, setHovered] = useState(false);
  const icon = CATEGORY_ICONS[playbook.category] ?? "📄";
  const iconBg = CATEGORY_COLORS[playbook.category] ?? "rgba(255,255,255,0.08)";
  const badge = BADGE_COLORS[playbook.color] ?? BADGE_COLORS.green;

  return (
    <Link href={`/playbooks/${playbook.slug}`} style={{ textDecoration: "none" }}>
      <div
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          background: hovered ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.03)",
          border: "1px solid",
          borderColor: hovered ? "rgba(255,255,255,0.14)" : "rgba(255,255,255,0.07)",
          borderRadius: 16,
          padding: "24px",
          cursor: "pointer",
          transition: "all 0.2s cubic-bezier(0.4,0,0.2,1)",
          transform: hovered ? "translateY(-2px)" : "none",
          display: "flex",
          flexDirection: "column",
          gap: 16,
          height: "100%",
          boxSizing: "border-box",
        }}
      >
        {/* Icon + badge row */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: 14,
              background: iconBg,
              border: "1px solid rgba(255,255,255,0.08)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 26,
              flexShrink: 0,
            }}
          >
            {icon}
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 5 }}>
            {isNew && (
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.06em",
                  padding: "2px 8px",
                  borderRadius: 20,
                  background: "rgba(10,132,255,0.18)",
                  color: "#0A84FF",
                  border: "1px solid rgba(10,132,255,0.35)",
                  textTransform: "uppercase",
                }}
              >
                New
              </span>
            )}
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 5,
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: "0.04em",
                padding: "4px 10px",
                borderRadius: 20,
                background: badge.bg,
                color: badge.color,
                border: `1px solid ${badge.color}33`,
                textTransform: "uppercase",
                whiteSpace: "nowrap",
              }}
            >
              ⏱ {playbook.time}
            </span>
          </div>
        </div>

        {/* Title + desc */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: "#fff",
              letterSpacing: "-0.02em",
              lineHeight: 1.3,
            }}
          >
            {playbook.title}
          </div>
          <div
            style={{
              fontSize: 13,
              color: "rgba(255,255,255,0.5)",
              lineHeight: 1.55,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {playbook.desc}
          </div>
        </div>

        {/* Tags */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
          {playbook.tags.slice(0, 3).map((t) => (
            <span
              key={t}
              style={{
                fontSize: 11,
                padding: "3px 9px",
                borderRadius: 20,
                background: "rgba(255,255,255,0.05)",
                color: "rgba(255,255,255,0.4)",
                border: "1px solid rgba(255,255,255,0.07)",
              }}
            >
              {t}
            </span>
          ))}
        </div>
      </div>
    </Link>
  );
}
