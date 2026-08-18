import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Sparkles,
  Search,
  FileText,
  Folder,
  Network,
  ArrowUpRight,
  ShieldCheck,
  Loader2,
} from "lucide-react";
import { Card, StatusPill } from "@/components/ui/Primitives.jsx";
import { checkHealth, BASE_URL } from "@/lib/api";

const TILES = [
  { to: "/ask", label: "Ask Knowledge", icon: Search, desc: "Query your indexed corpus", accent: "from-brand-400 to-brand-700" },
  { to: "/documents", label: "Browse Documents", icon: FileText, desc: "Upload and index a PDF", accent: "from-sky-400 to-blue-700" },
  { to: "/collections", label: "Collections", icon: Folder, desc: "Curated knowledge sets", accent: "from-emerald-400 to-emerald-700" },
  { to: "/graph", label: "Knowledge Graph", icon: Network, desc: "Entity & concept map", accent: "from-violet-400 to-violet-700" },
];

export default function Home() {
  // The only system-level endpoint the backend exposes is the liveness probe,
  // so that is the only status this page reports.
  const [health, setHealth] = useState({ state: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    checkHealth(controller.signal)
      .then((data) => setHealth({ state: "ok", data }))
      .catch((e) => {
        if (e.name !== "AbortError") setHealth({ state: "error", message: e.message });
      });
    return () => controller.abort();
  }, []);

  return (
    <div className="px-8 py-8 grid grid-cols-[minmax(0,1fr)_360px] gap-6">
      <div className="min-w-0 space-y-6">
        {/* Hero */}
        <div className="relative overflow-hidden rounded-2xl bg-ink-950 text-white p-8 bg-grid-dark elev-3">
          <div className="absolute -top-24 -right-24 h-64 w-64 rounded-full bg-brand-500/30 blur-3xl" />
          <div className="relative">
            <div className="inline-flex items-center gap-2 text-[11px] font-semibold tracking-[0.14em] uppercase text-brand-300 mb-3">
              <Sparkles className="h-3.5 w-3.5" /> Enterprise Knowledge
            </div>
            <h1 className="text-[34px] leading-[1.1] font-semibold tracking-tight max-w-2xl">
              Your organization's knowledge, intelligently retrieved.
            </h1>
            <p className="mt-3 text-ink-200 max-w-xl text-[14px]">
              Upload documents, index them into a hybrid search corpus, and ask
              questions that come back with grounded, cited answers.
            </p>
            <div className="mt-6 flex items-center gap-3">
              <Link
                to="/ask"
                data-testid="home-cta-ask"
                className="h-10 px-4 inline-flex items-center gap-2 rounded-xl bg-white text-ink-900 hover:bg-ink-100 text-[13.5px] font-semibold"
              >
                <Search className="h-4 w-4" /> Open Ask Knowledge
              </Link>
              <Link to="/documents" className="h-10 px-4 inline-flex items-center gap-2 rounded-xl bg-white/10 text-white hover:bg-white/15 text-[13.5px]">
                <FileText className="h-4 w-4" /> Upload a Document
              </Link>
            </div>
          </div>
        </div>

        {/* Tiles */}
        <div className="grid grid-cols-2 gap-4">
          {TILES.map((t) => {
            const Icon = t.icon;
            return (
              <Link
                key={t.to}
                to={t.to}
                data-testid={`home-tile-${t.label.toLowerCase().replace(/\s+/g, "-")}`}
                className="group rounded-2xl bg-white p-5 ring-1 ring-ink-100 hover:ring-brand-200 hover:elev-2 transition-all"
              >
                <div className={`h-11 w-11 rounded-xl bg-gradient-to-br ${t.accent} grid place-items-center text-white`}>
                  <Icon className="h-5 w-5" strokeWidth={2} />
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <div className="text-[14.5px] font-semibold text-ink-900">{t.label}</div>
                  <ArrowUpRight className="h-4 w-4 text-ink-300 group-hover:text-brand-600 group-hover:-translate-y-0.5 transition-all" />
                </div>
                <div className="text-[12.5px] text-ink-500 mt-1">{t.desc}</div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Right rail */}
      <aside className="space-y-4">
        <Card>
          <div className="p-5">
            <div className="text-[13px] font-semibold text-ink-700 mb-3 flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-emerald-600" /> System Health
            </div>
            {health.state === "loading" && (
              <div className="flex items-center gap-2 py-1.5 text-[12.5px] text-ink-400">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking…
              </div>
            )}
            {health.state === "error" && (
              <>
                <Row label="API" value={<StatusPill status="Unreachable" />} />
                <p className="mt-2 text-[11.5px] text-rose-600 break-words">{health.message}</p>
                <p className="mt-1 text-[11.5px] text-ink-400 break-all">{BASE_URL}</p>
              </>
            )}
            {health.state === "ok" && (
              <>
                <Row label="API" value={<StatusPill status="Healthy" />} />
                <Row label="Service" value={health.data.app} />
                <Row label="Version" value={health.data.version} />
              </>
            )}
          </div>
        </Card>
      </aside>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-[12.5px]">
      <span className="text-ink-500">{label}</span>
      <span className="text-ink-900 font-medium">{value}</span>
    </div>
  );
}
