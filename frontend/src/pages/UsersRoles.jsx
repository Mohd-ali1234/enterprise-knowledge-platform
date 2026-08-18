import { Users } from "lucide-react";
import { PageHeader, NotImplemented } from "@/components/ui/Primitives.jsx";

export default function UsersRoles() {
  return (
    <div>
      <PageHeader
        eyebrow="Identity"
        title="Users & Roles"
        subtitle="Provision people, scope their access, and align permissions with your enterprise identity provider."
      />
      <div className="px-8 pb-8">
        <NotImplemented icon={Users} feature="User management" />
      </div>
    </div>
  );
}
