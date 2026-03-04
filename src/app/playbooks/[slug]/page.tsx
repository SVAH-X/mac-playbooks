import { notFound } from "next/navigation";
import { getAllPlaybooks, getPlaybookBySlug } from "@/lib/playbooks";
import PlaybookPage from "@/components/PlaybookPage";

export async function generateStaticParams() {
  const playbooks = getAllPlaybooks();
  return playbooks.map((p) => ({ slug: p.slug }));
}

export default async function PlaybookRoute({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const playbook = getPlaybookBySlug(slug);
  if (!playbook) notFound();

  const allPlaybooks = getAllPlaybooks();
  return <PlaybookPage playbook={playbook} allPlaybooks={allPlaybooks} />;
}
