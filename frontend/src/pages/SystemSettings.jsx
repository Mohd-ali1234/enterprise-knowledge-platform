import { Settings } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

// Backend configuration lives in backend/app/core/config.py and is set through
// environment variables; there is no endpoint to read or change it at runtime.
export default function SystemSettings() {
  return (
    <div>
      <PageHeader
        eyebrow="Platform"
        title="System Settings"
        subtitle="Foundational configuration for retrieval, compliance and your organization."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={Settings} feature="Runtime configuration" />
      </div>
    </div>
  );
}
