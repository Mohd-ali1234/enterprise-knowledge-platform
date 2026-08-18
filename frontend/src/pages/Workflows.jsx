import { Workflow } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function Workflows() {
  return (
    <div>
      <PageHeader
        eyebrow="Automation"
        title="Workflows"
        subtitle="Compose multi-step AI workflows that trigger from your enterprise systems."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={Workflow} feature="Workflow automation" />
      </div>
    </div>
  );
}
