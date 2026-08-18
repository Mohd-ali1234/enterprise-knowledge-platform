import { Cpu } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function ModelManagement() {
  return (
    <div>
      <PageHeader
        eyebrow="Models"
        title="Model Management"
        subtitle="Govern the LLMs, embedders and rerankers powering retrieval and generation."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={Cpu} feature="Model management" />
      </div>
    </div>
  );
}
