"use client";

interface MiniMarkdownProps {
  text: string;
}

export default function MiniMarkdown({ text }: MiniMarkdownProps) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let inCode = false;
  let codeLines: string[] = [];
  let inTable = false;
  let tableRows: string[][] = [];

  const flushTable = () => {
    if (tableRows.length > 0) {
      elements.push(
        <div
          key={`t-${elements.length}`}
          style={{ overflowX: "auto", margin: "16px 0" }}
        >
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: 14,
            }}
          >
            <thead>
              <tr>
                {tableRows[0].map((cell, i) => (
                  <th
                    key={i}
                    style={{
                      padding: "8px 12px",
                      borderBottom: "2px solid rgba(255,255,255,0.15)",
                      textAlign: "left",
                      color: "rgba(255,255,255,0.7)",
                      fontWeight: 600,
                    }}
                  >
                    {cell}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.slice(2).map((row, ri) => (
                <tr key={ri}>
                  {row.map((cell, ci) => (
                    <td
                      key={ci}
                      style={{
                        padding: "8px 12px",
                        borderBottom: "1px solid rgba(255,255,255,0.08)",
                        color: "rgba(255,255,255,0.85)",
                      }}
                    >
                      {cell}
                    </td>
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
          <pre
            key={`c-${elements.length}`}
            style={{
              background: "rgba(0,0,0,0.4)",
              borderRadius: 10,
              padding: "16px 20px",
              margin: "12px 0",
              overflowX: "auto",
              fontSize: 13,
              lineHeight: 1.6,
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <code
              style={{
                color: "#e0e0e0",
                fontFamily:
                  "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
              }}
            >
              {codeLines.join("\n")}
            </code>
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
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    if (line.startsWith("|")) {
      inTable = true;
      tableRows.push(
        line
          .split("|")
          .filter(Boolean)
          .map((c) => c.trim())
      );
      continue;
    }
    if (inTable) flushTable();
    if (line.startsWith("## ")) {
      elements.push(
        <h2
          key={`h-${elements.length}`}
          style={{
            fontSize: 20,
            fontWeight: 700,
            color: "#fff",
            marginTop: 28,
            marginBottom: 12,
            letterSpacing: "-0.02em",
          }}
        >
          {line.slice(3)}
        </h2>
      );
    } else if (line.startsWith("- **")) {
      const m = line.match(/- \*\*(.+?)\*\*:?\s*(.*)/);
      if (m)
        elements.push(
          <div
            key={`li-${elements.length}`}
            style={{
              padding: "4px 0 4px 16px",
              color: "rgba(255,255,255,0.85)",
              fontSize: 15,
              lineHeight: 1.6,
            }}
          >
            <span style={{ color: "#fff", fontWeight: 600 }}>{m[1]}</span>
            {m[2] ? ": " + m[2] : ""}
          </div>
        );
    } else if (line.startsWith("- ")) {
      elements.push(
        <div
          key={`li-${elements.length}`}
          style={{
            padding: "3px 0 3px 16px",
            color: "rgba(255,255,255,0.8)",
            fontSize: 15,
            lineHeight: 1.6,
          }}
        >
          • {line.slice(2)}
        </div>
      );
    } else if (line.trim() === "") {
      elements.push(<div key={`sp-${elements.length}`} style={{ height: 8 }} />);
    } else {
      const processed = line.replace(
        /`([^`]+)`/g,
        '<code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-size:13px;font-family:SF Mono,monospace">$1</code>'
      );
      elements.push(
        <p
          key={`p-${elements.length}`}
          style={{
            color: "rgba(255,255,255,0.8)",
            fontSize: 15,
            lineHeight: 1.7,
            margin: "6px 0",
          }}
          dangerouslySetInnerHTML={{ __html: processed }}
        />
      );
    }
  }
  flushTable();
  return <>{elements}</>;
}
