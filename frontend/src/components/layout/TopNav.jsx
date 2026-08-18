import { Search, HelpCircle, PanelLeft } from "lucide-react";
import { useAppStore } from "@/lib/store";

export default function TopNav() {
  const toggleSidebar = useAppStore((s) => s.toggleSidebar);

  return (
    <header
      data-testid="top-nav"
      className="sticky top-0 z-30 bg-white/85 backdrop-blur-xl border-b border-ink-100/80"
    >
      <div className="h-[64px] px-6 flex items-center gap-4">
        <button
          onClick={toggleSidebar}
          data-testid="topnav-sidebar-toggle"
          className="h-9 w-9 grid place-items-center rounded-lg hover:bg-ink-50 text-ink-500 focus-ring"
        >
          <PanelLeft className="h-[18px] w-[18px]" strokeWidth={1.8} />
        </button>

        {/* Global search */}
        <div className="flex-1 max-w-[640px]">
          <div className="group relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-[16px] w-[16px] text-ink-400" />
            <input
              data-testid="global-search-input"
              placeholder="Search across documents, teams, and knowledge..."
              className="w-full h-10 pl-9 pr-20 rounded-xl border border-ink-100 bg-ink-50/60 hover:bg-white focus:bg-white focus:border-brand-400 focus:ring-4 focus:ring-brand-400/15 outline-none text-[13.5px] placeholder:text-ink-400 transition-all"
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
              <kbd className="text-[10.5px] font-mono px-1.5 py-0.5 rounded-md border border-ink-200 bg-white text-ink-500">⌘</kbd>
              <kbd className="text-[10.5px] font-mono px-1.5 py-0.5 rounded-md border border-ink-200 bg-white text-ink-500">K</kbd>
            </div>
          </div>
        </div>

        <div className="flex-1" />

        {/* No auth, notifications or app directory in the backend, so the
            signed-in user, the notification count and the app launcher are
            gone rather than invented. */}
        <div className="flex items-center gap-1">
          <button data-testid="help-btn" className="h-9 px-2.5 inline-flex items-center gap-1.5 rounded-lg hover:bg-ink-50 text-ink-600 text-[13px]">
            <HelpCircle className="h-[17px] w-[17px]" strokeWidth={1.8} /> Help
          </button>
        </div>
      </div>
    </header>
  );
}
