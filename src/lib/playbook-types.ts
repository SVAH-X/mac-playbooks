export interface PlaybookTab {
  label: string;
  content: string;
}

export interface PlaybookFrontmatter {
  slug: string;
  title: string;
  time: string;
  color: "green" | "orange" | "red";
  desc: string;
  tags: string[];
  spark: string;
  category: string;
  featured: boolean;
  whatsNew: boolean;
}

export interface Playbook extends PlaybookFrontmatter {
  tabs: PlaybookTab[];
}

export interface CategoryGroup {
  id: string;
  label: string;
  playbooks: Playbook[];
}

const CATEGORY_ORDER = [
  "onboarding",
  "inference",
  "fine-tuning",
  "data-science",
  "image-gen",
  "applications",
  "robotics",
  "tools",
];

const CATEGORY_LABELS: Record<string, string> = {
  onboarding: "onboarding",
  inference: "inference",
  "fine-tuning": "fine tuning",
  "data-science": "data science",
  "image-gen": "image generation",
  applications: "applications",
  robotics: "robotics",
  tools: "tools",
};

export function groupByCategory(playbooks: Playbook[]): CategoryGroup[] {
  const map = new Map<string, Playbook[]>();
  for (const p of playbooks) {
    const arr = map.get(p.category) ?? [];
    arr.push(p);
    map.set(p.category, arr);
  }

  return CATEGORY_ORDER.filter((id) => map.has(id)).map((id) => ({
    id,
    label: CATEGORY_LABELS[id] ?? id,
    playbooks: map.get(id)!,
  }));
}
