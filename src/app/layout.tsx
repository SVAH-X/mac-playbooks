import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "mac-playbooks",
  description:
    "Step-by-step playbooks for AI/ML workloads on Apple Silicon — the macOS counterpart to NVIDIA DGX Spark Playbooks.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body
        style={{
          margin: 0,
          padding: 0,
          background: "#0a0a0f",
          fontFamily:
            "'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif",
          color: "#fff",
        }}
      >
        {children}
      </body>
    </html>
  );
}
