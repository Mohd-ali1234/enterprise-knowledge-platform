import { Folder } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function Collections() {
  return (
    <div>
      <PageHeader
        eyebrow="Curated Knowledge"
        title="Collections"
        subtitle="Group documents by team, project or topic — with governance, ownership and access scoped per collection."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={Folder} feature="Collections" />
      </div>
    </div>
  );
}
