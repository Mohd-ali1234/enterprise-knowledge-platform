import { Lock } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function Permissions() {
  return (
    <div>
      <PageHeader
        eyebrow="Access Control"
        title="Permissions"
        subtitle="Granular, role-based permissions across the knowledge platform."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={Lock} feature="Role-based permissions" />
      </div>
    </div>
  );
}
