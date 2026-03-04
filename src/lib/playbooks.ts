import fs from "fs";
import path from "path";
import matter from "gray-matter";
import type { Playbook, PlaybookFrontmatter } from "./playbook-types";

export type { Playbook, PlaybookTab, PlaybookFrontmatter, CategoryGroup } from "./playbook-types";
export { groupByCategory } from "./playbook-types";

const CONTENT_DIR = path.join(process.cwd(), "content", "playbooks");

function parsePlaybook(filePath: string): Playbook {
  const raw = fs.readFileSync(filePath, "utf-8");
  const { data, content } = matter(raw);

  const tabRegex = /<!--\s*tab:\s*([^-]+?)\s*-->/g;
  const tabs: { label: string; content: string }[] = [];
  let lastIndex = 0;
  let lastLabel = "";
  let match: RegExpExecArray | null;

  while ((match = tabRegex.exec(content)) !== null) {
    if (lastLabel) {
      tabs.push({
        label: lastLabel,
        content: content.slice(lastIndex, match.index).trim(),
      });
    }
    lastLabel = match[1].trim();
    lastIndex = match.index + match[0].length;
  }

  if (lastLabel) {
    tabs.push({
      label: lastLabel,
      content: content.slice(lastIndex).trim(),
    });
  }

  if (tabs.length === 0) {
    tabs.push({ label: "Overview", content: content.trim() });
  }

  return {
    ...(data as PlaybookFrontmatter),
    tabs,
  };
}

export function getAllPlaybooks(): Playbook[] {
  const files = fs.readdirSync(CONTENT_DIR).filter((f) => f.endsWith(".md"));
  return files.map((f) => parsePlaybook(path.join(CONTENT_DIR, f)));
}

export function getPlaybookBySlug(slug: string): Playbook | null {
  const filePath = path.join(CONTENT_DIR, `${slug}.md`);
  if (!fs.existsSync(filePath)) return null;
  return parsePlaybook(filePath);
}
