import { ScrollText } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function AuditLogs() {
  return (
    <div>
      <PageHeader
        eyebrow="Compliance"
        title="Audit Logs"
        subtitle="Immutable record of every action taken across the platform."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={ScrollText} feature="Audit logging" />
      </div>
    </div>
  );
}
