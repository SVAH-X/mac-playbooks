"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

interface NavProps {
  search?: string;
  onSearchChange?: (val: string) => void;
}

export default function Nav({ search = "", onSearchChange }: NavProps) {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const navBg =
    scrollY > 40 ? "rgba(10,10,15,0.85)" : "transparent";
  const navBorder =
    scrollY > 40 ? "rgba(255,255,255,0.06)" : "transparent";

  return (
    <nav
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        background: navBg,
        backdropFilter: scrollY > 40 ? "blur(20px) saturate(180%)" : "none",
        borderBottom: `1px solid ${navBorder}`,
        transition: "all 0.3s ease",
      }}
    >
      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          padding: "0 32px",
          height: 56,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <Link
            href="/"
            style={{
              textDecoration: "none",
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <span style={{ fontSize: 22 }}>🍎</span>
            <span
              style={{
                fontSize: 16,
                fontWeight: 800,
                color: "#fff",
                letterSpacing: "-0.03em",
              }}
            >
              mac-playbooks
            </span>
          </Link>
          <div style={{ display: "flex", gap: 4 }}>
            <Link
              href="/"
              style={{
                padding: "6px 12px",
                fontSize: 13,
                color: "rgba(255,255,255,0.5)",
                textDecoration: "none",
                borderRadius: 8,
                fontWeight: 500,
              }}
            >
              Playbooks
            </Link>
            <a
              href="https://github.com/SVAH-X/mac-playbooks"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                padding: "6px 12px",
                fontSize: 13,
                color: "rgba(255,255,255,0.5)",
                textDecoration: "none",
                borderRadius: 8,
                fontWeight: 500,
              }}
            >
              GitHub
            </a>
          </div>
        </div>
        {onSearchChange && (
          <div style={{ position: "relative" }}>
            <input
              value={search}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search playbooks..."
              style={{
                width: 220,
                padding: "7px 14px 7px 34px",
                borderRadius: 10,
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#fff",
                fontSize: 13,
                outline: "none",
                fontFamily: "inherit",
              }}
            />
            <span
              style={{
                position: "absolute",
                left: 12,
                top: "50%",
                transform: "translateY(-50%)",
                color: "rgba(255,255,255,0.3)",
                fontSize: 13,
              }}
            >
              ⌘
            </span>
          </div>
        )}
      </div>
    </nav>
  );
}
