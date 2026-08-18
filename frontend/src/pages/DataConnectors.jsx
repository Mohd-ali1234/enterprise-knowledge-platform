import { Database } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function DataConnectors() {
  return (
    <div>
      <PageHeader
        eyebrow="Sources"
        title="Data Connectors"
        subtitle="Live, governed ingestion from your enterprise systems with continuous incremental sync."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={Database} feature="Data connectors" />
      </div>
    </div>
  );
}
