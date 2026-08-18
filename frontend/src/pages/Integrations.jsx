import { Plug } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function Integrations() {
  return (
    <div>
      <PageHeader
        eyebrow="Connected Apps"
        title="Integrations"
        subtitle="First-party connectors stream content into the knowledge index with respect to source-level permissions."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={Plug} feature="Integrations" />
      </div>
    </div>
  );
}
