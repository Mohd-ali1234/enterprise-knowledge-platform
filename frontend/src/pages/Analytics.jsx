import { BarChart3 } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function Analytics() {
  return (
    <div>
      <PageHeader
        eyebrow="Insights"
        title="Reports & Analytics"
        subtitle="Track adoption, retrieval quality, and how knowledge flows through your organization."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={BarChart3} feature="Reporting and analytics" />
      </div>
    </div>
  );
}
