import { classNames } from "@/lib/utils";

export function PageHeader({ title, subtitle, actions, eyebrow }) {
  return (
    <div className="px-8 pt-8 pb-6 flex items-start gap-6">
      <div className="flex-1 min-w-0">
        {eyebrow && (
          <div className="text-[11px] font-semibold tracking-[0.14em] uppercase text-brand-600 mb-2">
            {eyebrow}
          </div>
        )}
        <h1 className="text-[28px] leading-[1.15] font-semibold tracking-tight text-ink-900">
          {title}
        </h1>
        {subtitle && (
          <p className="mt-1.5 text-[14px] text-ink-400 max-w-2xl">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function StatusPill({ status }) {
  const map = {
    Active: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    Connected: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    Live: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    Healthy: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    Operational: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    Syncing: "bg-amber-50 text-amber-700 ring-amber-200",
    Idle: "bg-ink-100 text-ink-500 ring-ink-200",
    Paused: "bg-ink-100 text-ink-500 ring-ink-200",
    Draft: "bg-ink-100 text-ink-500 ring-ink-200",
    Suspended: "bg-rose-50 text-rose-700 ring-rose-200",
  };
  return (
    <span
      className={classNames(
        "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium ring-1 ring-inset",
        map[status] || "bg-ink-100 text-ink-600 ring-ink-200"
      )}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
      {status}
    </span>
  );
}

export function Card({ children, className }) {
  return (
    <div className={classNames("bg-white rounded-2xl ring-1 ring-ink-100 elev-1", className)}>
      {children}
    </div>
  );
}

export function FileTypeBadge({ type }) {
  const colors = {
    PDF: "bg-rose-50 text-rose-600 ring-rose-200",
    DOCX: "bg-blue-50 text-blue-600 ring-blue-200",
    XLSX: "bg-emerald-50 text-emerald-600 ring-emerald-200",
    PPTX: "bg-amber-50 text-amber-600 ring-amber-200",
  };
  return (
    <span
      className={classNames(
        "inline-flex items-center justify-center min-w-[40px] h-[40px] rounded-lg text-[10.5px] font-bold tracking-wider ring-1 ring-inset",
        colors[type] || "bg-ink-100 text-ink-600 ring-ink-200"
      )}
    >
      {type}
    </span>
  );
}

/**
 * Placeholder for a page whose feature has no backend endpoint yet.
 *
 * The API currently exposes only document ingestion and knowledge retrieval
 * (see lib/api.js). Pages for identity, governance, automation and analytics
 * have nothing to read from, so they say so plainly rather than rendering
 * invented data.
 */
export function NotImplemented({ icon: Icon, feature }) {
  return (
    <EmptyState
      icon={Icon}
      title={`${feature} is not available yet`}
      description="The backend does not expose an endpoint for this feature, so there is nothing to show. This page is a placeholder until one exists."
    />
  );
}

export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="rounded-2xl border border-dashed border-ink-200 bg-white p-10 text-center">
      {Icon && (
        <div className="mx-auto h-12 w-12 rounded-xl bg-brand-50 grid place-items-center text-brand-600 mb-3">
          <Icon className="h-5 w-5" />
        </div>
      )}
      <div className="text-[15px] font-semibold text-ink-900">{title}</div>
      {description && <p className="mt-1 text-[13px] text-ink-400 max-w-md mx-auto">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
