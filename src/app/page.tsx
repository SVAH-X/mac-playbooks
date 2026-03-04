import { getAllPlaybooks } from "@/lib/playbooks";
import MacPlaybooksHome from "@/components/MacPlaybooksHome";

export default function Home() {
  const playbooks = getAllPlaybooks();
  return <MacPlaybooksHome playbooks={playbooks} />;
}
