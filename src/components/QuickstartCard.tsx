"use client";

import { useState } from "react";
import Link from "next/link";
import type { Playbook } from "@/lib/playbook-types";

const BADGE_COLORS: Record<string, { bg: string; color: string }> = {
  green: { bg: "rgba(45,212,191,0.15)", color: "#2dd4bf" },
  orange: { bg: "rgba(245,158,11,0.15)", color: "#f59e0b" },
  red: { bg: "rgba(239,68,68,0.15)", color: "#ef4444" },
};

interface QuickstartCardProps {
  playbook: Playbook;
}

export default function QuickstartCard({ playbook }: QuickstartCardProps) {
  const [hovered, setHovered] = useState(false);
  const badge = BADGE_COLORS[playbook.color] ?? BADGE_COLORS.green;

  return (
    <Link href={`/playbooks/${playbook.slug}`} style={{ textDecoration: "none" }}>
      <div
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 14,
          padding: "14px 18px",
          background: hovered ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 14,
          cursor: "pointer",
          transition: "all 0.2s ease",
        }}
      >
        <div
          style={{
            width: 44,
            height: 44,
            borderRadius: 12,
            background: "linear-gradient(135deg, rgba(10,132,255,0.2), rgba(48,209,88,0.15))",
            border: "1px solid rgba(255,255,255,0.08)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 22,
            flexShrink: 0,
          }}
        >
          ⚡
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span style={{ fontSize: 14, fontWeight: 700, color: "#fff" }}>
              {playbook.title}
            </span>
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: "0.04em",
                padding: "2px 8px",
                borderRadius: 20,
                background: badge.bg,
                color: badge.color,
                border: `1px solid ${badge.color}33`,
                textTransform: "uppercase",
              }}
            >
              {playbook.time}
            </span>
          </div>
          <div
            style={{
              fontSize: 12,
              color: "rgba(255,255,255,0.4)",
              marginTop: 2,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {playbook.desc}
          </div>
        </div>
        <div style={{ color: "rgba(255,255,255,0.2)", fontSize: 16 }}>→</div>
      </div>
    </Link>
  );
}
