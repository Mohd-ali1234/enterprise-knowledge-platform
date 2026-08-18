import { Outlet } from "react-router-dom"
import Sidebar from "@/components/layout/Sidebar.jsx";
import TopNav from "@/components/layout/TopNav.jsx";

export default function AppLayout() {
  return (
    <div className="min-h-screen flex bg-ink-50">
      <Sidebar />
      <div className="flex-1 min-w-0 flex flex-col">
        <TopNav />
        <main className="flex-1 min-w-0">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
