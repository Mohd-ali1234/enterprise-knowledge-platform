import { ShieldCheck } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function DataGovernance() {
  return (
    <div>
      <PageHeader
        eyebrow="Governance"
        title="Data Governance"
        subtitle="Policies, classifications, retention and data-residency controls applied across the knowledge layer."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={ShieldCheck} feature="Data governance" />
      </div>
    </div>
  );
}
