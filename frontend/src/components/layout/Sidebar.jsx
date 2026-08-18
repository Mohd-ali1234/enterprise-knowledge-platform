import { NavLink, useLocation } from "react-router-dom";
import {
  Home,
  Search,
  FileText,
  Folder,
  Network,
  BarChart3,
  Workflow,
  Plug,
  Users,
  Lock,
  ScrollText,
  ShieldCheck,
  Cpu,
  Database,
  Settings,
  ChevronsLeft,
  ChevronsRight,
  Sparkles,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { classNames } from "@/lib/utils";

const SECTIONS = [
  {
    label: "Main Navigation",
    items: [
      { to: "/", label: "Home", icon: Home },
      { to: "/ask", label: "Ask Knowledge", icon: Search },
      { to: "/documents", label: "Documents", icon: FileText },
      { to: "/collections", label: "Collections", icon: Folder },
      { to: "/graph", label: "Knowledge Graph", icon: Network },
      { to: "/analytics", label: "Reports & Analytics", icon: BarChart3 },
      { to: "/workflows", label: "Workflows", icon: Workflow },
      { to: "/integrations", label: "Integrations", icon: Plug },
    ],
  },
  {
    label: "Admin & Security",
    items: [
      { to: "/users", label: "Users & Roles", icon: Users },
      { to: "/permissions", label: "Permissions", icon: Lock },
      { to: "/audit", label: "Audit Logs", icon: ScrollText },
      { to: "/governance", label: "Data Governance", icon: ShieldCheck },
    ],
  },
  {
    label: "System",
    items: [
      { to: "/models", label: "Model Management", icon: Cpu },
      { to: "/connectors", label: "Data Connectors", icon: Database },
      { to: "/settings", label: "System Settings", icon: Settings },
    ],
  },
];

export default function Sidebar() {
  const collapsed = useAppStore((s) => s.sidebarCollapsed);
  const toggle = useAppStore((s) => s.toggleSidebar);
  const loc = useLocation();

  return (
    <aside
      data-testid="app-sidebar"
      className={classNames(
        "shrink-0 h-screen sticky top-0 bg-ink-950 text-ink-100 border-r border-white/[0.06] flex flex-col scroll-dark transition-[width] duration-300",
        collapsed ? "w-[76px]" : "w-[268px]"
      )}
    >
{/* Brand */}
<div className="px-5 pt-5 pb-4 flex items-center gap-3">
  {/* Logo Container - Removed overflow-hidden so the green badge can pop out */}
  <div className="relative h-7 w-7 grid place-items-center">
    <img 
      src="/brandLogo.png" 
      alt="Enterprise Knowledge Logo" 
      className="h-full w-full object-contain drop-shadow-md" 
    />
    {/* Status dot positioned exactly on the bottom right edge of the hexagon */}
    <span className="absolute bottom-0 right-0 h-2 w-2 rounded-full bg-emerald-400 ring-[2.5px] ring-ink-950" />
  </div>
  
  {!collapsed && (
    <div className="leading-tight select-none">
      <div className="text-[13px] font-semibold tracking-tight text-white">
        Enterprise Knowledge
      </div>
      <div className="text-[13px] font-semibold tracking-tight text-ink-200">
        Intelligence Platform
      </div>
    </div>
  )}
</div>

<div className="px-3">
  <div className="h-px bg-white/[0.06]" />
</div>

      {/* Nav scroll */}
      <nav className="flex-1 overflow-y-auto scroll-dark px-3 py-4 space-y-6">
        {SECTIONS.map((sec) => (
          <div key={sec.label}>
            {!collapsed && (
              <div className="px-2 pb-2 text-[10.5px] font-semibold tracking-[0.14em] uppercase text-ink-400">
                {sec.label}
              </div>
            )}
            <ul className="space-y-0.5">
              {sec.items.map((it) => {
                const active = loc.pathname === it.to;
                const Icon = it.icon;
                return (
                  <li key={it.to}>
                    <NavLink
                      to={it.to}
                      data-testid={`nav-${it.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`}
                      className={classNames(
                        "group relative flex items-center gap-3 px-2.5 py-2 rounded-lg text-[13.5px] transition-colors",
                        active
                          ? "bg-white/[0.08] text-white"
                          : "text-ink-200 hover:text-white hover:bg-white/[0.04]"
                      )}
                    >
                      {active && (
                        <span className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-r-full bg-gradient-to-b from-brand-300 to-brand-500" />
                      )}
                      <Icon
                        className={classNames(
                          "h-[17px] w-[17px] shrink-0",
                          active ? "text-brand-300" : "text-ink-300 group-hover:text-ink-100"
                        )}
                        strokeWidth={1.8}
                      />
                      {!collapsed && <span className="truncate">{it.label}</span>}
                    </NavLink>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* There is no tenancy or org model in the backend, so no org switcher. */}
      <div className="border-t border-white/[0.06] p-3">
        <button
          onClick={toggle}
          data-testid="sidebar-collapse-toggle"
          className="mt-2 w-full flex items-center justify-center gap-2 text-[11.5px] text-ink-300 hover:text-white py-1.5 rounded-md hover:bg-white/[0.04]"
        >
          {collapsed ? <ChevronsRight className="h-4 w-4" /> : <><ChevronsLeft className="h-4 w-4" /> Collapse</>}
        </button>
      </div>
    </aside>
  );
}
