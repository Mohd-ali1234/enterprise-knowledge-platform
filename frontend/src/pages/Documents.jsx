import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  FileText,
  Search,
  Download,
  Upload,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { PageHeader, Card, FileTypeBadge, EmptyState } from "@/components/ui/Primitives.jsx";
import { uploadAndIndex, listDocuments, deleteDocuments } from "@/lib/api";

/** Best-effort file type for the badge, from the recorded type or the name. */
function badgeType(document) {
  const type = document.file_type || document.file_name.split(".").pop() || "";
  return type.toUpperCase().slice(0, 4) || "DOC";
}

function formatIndexedAt(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}

export default function Documents() {
  const fileInput = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null); // { kind: "success" | "error", text }
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(() => new Set());
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Split from `refresh` so the mount effect never touches state synchronously:
  // every setState here happens after the first await.
  const load = useCallback(async (signal) => {
    try {
      const data = await listDocuments(signal);
      setDocuments(data.documents);
      // Drop selections for documents that no longer exist.
      const live = new Set(data.documents.map((d) => d.document_id));
      setSelected((current) => new Set([...current].filter((id) => live.has(id))));
      setLoadError(null);
    } catch (e) {
      if (e.name !== "AbortError") setLoadError(e.message || "Could not load documents.");
    } finally {
      setLoading(false);
    }
  }, []);

  /** Re-read the list, showing the spinner. `loading` already starts true. */
  const refresh = useCallback(
    async (signal) => {
      setLoading(true);
      await load(signal);
    },
    [load],
  );

  useEffect(() => {
    const controller = new AbortController();
    // Wrapped so no state update is reachable synchronously from the effect
    // body; every setState in `load` happens after its first await.
    (async () => {
      await load(controller.signal);
    })();
    return () => controller.abort();
  }, [load]);

  async function handleFiles(files) {
    const file = files?.[0];
    if (!file) return;
    setUploading(true);
    setStatus(null);
    try {
      const res = await uploadAndIndex(file);
      setStatus({
        kind: "success",
        text: `"${res.file_name}" indexed — ${res.chunks_indexed} chunks, ${res.total_entities} entities.`,
      });
      await refresh();
    } catch (e) {
      setStatus({ kind: "error", text: e.message || "Upload failed." });
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function handleDelete() {
    setDeleting(true);
    setStatus(null);
    try {
      const res = await deleteDocuments([...selected]);
      setStatus({ kind: "success", text: res.message });
      setSelected(new Set());
      setConfirming(false);
      await refresh();
    } catch (e) {
      setStatus({ kind: "error", text: e.message || "Delete failed." });
    } finally {
      setDeleting(false);
    }
  }

  const term = query.trim().toLowerCase();
  const visible = useMemo(
    () =>
      term ? documents.filter((d) => d.file_name.toLowerCase().includes(term)) : documents,
    [documents, term],
  );

  const totalChunks = documents.reduce((sum, d) => sum + d.chunk_count, 0);
  const allVisibleSelected = visible.length > 0 && visible.every((d) => selected.has(d.document_id));
  const selectedDocs = documents.filter((d) => selected.has(d.document_id));
  const selectedChunks = selectedDocs.reduce((sum, d) => sum + d.chunk_count, 0);

  const toggleOne = (id) =>
    setSelected((current) => {
      const next = new Set(current);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  // Select-all acts on what is visible, so it never silently selects rows
  // hidden behind the search filter.
  const toggleAllVisible = () =>
    setSelected((current) => {
      const next = new Set(current);
      visible.forEach((d) =>
        allVisibleSelected ? next.delete(d.document_id) : next.add(d.document_id),
      );
      return next;
    });

  return (
    <div>
      <input
        ref={fileInput}
        type="file"
        data-testid="docs-file-input"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <PageHeader
        eyebrow="Knowledge Library"
        title="Documents"
        subtitle="Every document indexed into the searchable corpus."
        actions={
          <>
            <button
              data-testid="docs-upload"
              onClick={() => fileInput.current?.click()}
              disabled={uploading}
              className="h-10 px-4 inline-flex items-center gap-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-[13px] font-medium disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              {uploading ? "Indexing…" : "Upload"}
            </button>
            <button
              data-testid="docs-refresh"
              onClick={() => refresh()}
              className="h-10 px-3 inline-flex items-center gap-2 rounded-lg ring-1 ring-ink-200 text-[13px] text-ink-700 hover:bg-white"
            >
              <RefreshCw className={loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} /> Refresh
            </button>
          </>
        }
      />
      <div className="px-8 pb-8">
        {status && (
          <div
            data-testid="docs-status"
            className={
              "mb-4 rounded-xl p-3.5 flex items-start gap-3 ring-1 " +
              (status.kind === "success" ? "bg-emerald-50 ring-emerald-200" : "bg-rose-50 ring-rose-200")
            }
          >
            {status.kind === "success" ? (
              <CheckCircle2 className="h-[18px] w-[18px] text-emerald-500 shrink-0 mt-0.5" />
            ) : (
              <AlertTriangle className="h-[18px] w-[18px] text-rose-500 shrink-0 mt-0.5" />
            )}
            <div className={"text-[13px] " + (status.kind === "success" ? "text-emerald-800" : "text-rose-700")}>
              {status.text}
            </div>
          </div>
        )}

        {loadError && (
          <div className="mb-4 rounded-xl bg-rose-50 ring-1 ring-rose-200 p-3.5 text-[13px] text-rose-700">
            {loadError}
          </div>
        )}

        {confirming && (
          <ConfirmDelete
            documents={selectedDocs}
            chunks={selectedChunks}
            deleting={deleting}
            onCancel={() => setConfirming(false)}
            onConfirm={handleDelete}
          />
        )}

        {loading && documents.length === 0 ? (
          <Card>
            <div className="p-8 flex items-center justify-center gap-2 text-[13px] text-ink-400">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading indexed documents…
            </div>
          </Card>
        ) : documents.length === 0 ? (
          <EmptyState
            icon={FileText}
            title="No documents indexed yet"
            description="Upload a PDF, DOCX, HTML, Markdown, CSV, TXT or a Notion/Confluence export to add it to the searchable corpus."
          />
        ) : (
          <Card>
            {/* The toolbar swaps to a selection bar so the destructive action
                only ever appears alongside what it would destroy. */}
            {selected.size > 0 ? (
              <div
                data-testid="selection-bar"
                className="p-4 border-b border-ink-100 flex items-center gap-3 bg-brand-50/60"
              >
                <span className="text-[13px] font-medium text-ink-900">
                  {selected.size} selected
                  <span className="text-ink-500 font-normal"> · {selectedChunks} chunks</span>
                </span>
                <button
                  data-testid="clear-selection"
                  onClick={() => setSelected(new Set())}
                  className="h-8 px-2.5 inline-flex items-center gap-1.5 rounded-lg text-[12.5px] text-ink-600 hover:bg-white"
                >
                  <X className="h-3.5 w-3.5" /> Clear
                </button>
                <button
                  data-testid="delete-selected"
                  onClick={() => setConfirming(true)}
                  className="ml-auto h-9 px-3.5 inline-flex items-center gap-2 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-[13px] font-medium"
                >
                  <Trash2 className="h-4 w-4" /> Delete
                </button>
              </div>
            ) : (
              <div className="p-4 border-b border-ink-100 flex items-center gap-3">
                <div className="relative flex-1 max-w-md">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-ink-400" />
                  <input
                    data-testid="docs-search"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={`Search ${documents.length} documents...`}
                    className="w-full h-10 pl-9 pr-3 rounded-lg ring-1 ring-ink-100 focus:ring-2 focus:ring-brand-400 outline-none text-[13px] bg-ink-50/40"
                  />
                </div>
                <div className="ml-auto text-[12px] text-ink-400">
                  {visible.length === documents.length
                    ? `${documents.length} documents · ${totalChunks} chunks`
                    : `${visible.length} of ${documents.length} documents`}
                </div>
              </div>
            )}

            <div className="grid grid-cols-[40px_1fr_120px_160px_44px] text-[11.5px] uppercase font-semibold tracking-wider text-ink-400 px-5 py-3 border-b border-ink-100 bg-ink-50/30">
              <div className="flex items-center">
                <Checkbox
                  checked={allVisibleSelected}
                  indeterminate={!allVisibleSelected && visible.some((d) => selected.has(d.document_id))}
                  onChange={toggleAllVisible}
                  testId="select-all"
                  label="Select all documents"
                />
              </div>
              <div>Name</div><div>Chunks</div><div>Indexed</div><div></div>
            </div>

            <ul className="divide-y divide-ink-100">
              {visible.map((d) => {
                const isSelected = selected.has(d.document_id);
                return (
                  <li
                    key={d.document_id}
                    data-testid={`doc-${d.document_id}`}
                    className={
                      "grid grid-cols-[40px_1fr_120px_160px_44px] items-center px-5 py-3 " +
                      (isSelected ? "bg-brand-50/50" : "hover:bg-ink-50/60")
                    }
                  >
                    <div className="flex items-center">
                      <Checkbox
                        checked={isSelected}
                        onChange={() => toggleOne(d.document_id)}
                        testId={`select-${d.document_id}`}
                        label={`Select ${d.file_name}`}
                      />
                    </div>
                    <div className="flex items-center gap-3 min-w-0">
                      <FileTypeBadge type={badgeType(d)} />
                      <div className="min-w-0">
                        <div className="text-[13.5px] font-medium text-ink-900 truncate">{d.file_name}</div>
                        <div className="text-[11.5px] text-ink-400 font-mono truncate">{d.document_id}</div>
                      </div>
                    </div>
                    <div className="text-[12.5px] text-ink-700 font-mono">{d.chunk_count}</div>
                    <div className="text-[12.5px] text-ink-500">{formatIndexedAt(d.indexed_at)}</div>
                    <div className="flex items-center justify-end">
                      <button className="h-8 w-8 grid place-items-center rounded-md hover:bg-ink-100 text-ink-500">
                        <Download className="h-4 w-4" />
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          </Card>
        )}
      </div>
    </div>
  );
}

function ConfirmDelete({ documents, chunks, deleting, onCancel, onConfirm }) {
  return (
    <div
      data-testid="confirm-delete"
      className="mb-4 rounded-xl bg-white ring-1 ring-rose-200 p-4 elev-1"
    >
      <div className="flex items-start gap-3">
        <div className="h-9 w-9 shrink-0 rounded-lg bg-rose-50 grid place-items-center text-rose-600">
          <Trash2 className="h-[18px] w-[18px]" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[14px] font-semibold text-ink-900">
            Delete {documents.length} document{documents.length === 1 ? "" : "s"}?
          </div>
          <p className="mt-1 text-[13px] text-ink-500">
            This removes {chunks} chunk{chunks === 1 ? "" : "s"} from the index permanently.
            The content stops being retrievable and cannot be recovered without
            re-uploading the original file{documents.length === 1 ? "" : "s"}.
          </p>
          {/* Naming what will go makes an accidental select-all recoverable
              before it is irreversible, not after. */}
          <ul className="mt-3 max-h-32 overflow-y-auto text-[12.5px] text-ink-700 space-y-1">
            {documents.map((d) => (
              <li key={d.document_id} className="flex items-center gap-2">
                <span className="truncate">{d.file_name}</span>
                <span className="text-ink-400 font-mono text-[11.5px] shrink-0">
                  {d.chunk_count} chunks
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            data-testid="cancel-delete"
            onClick={onCancel}
            disabled={deleting}
            className="h-9 px-3.5 rounded-lg ring-1 ring-ink-200 text-[13px] text-ink-700 hover:bg-ink-50 disabled:opacity-60"
          >
            Cancel
          </button>
          <button
            data-testid="confirm-delete-btn"
            onClick={onConfirm}
            disabled={deleting}
            className="h-9 px-3.5 inline-flex items-center gap-2 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-[13px] font-medium disabled:opacity-60"
          >
            {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            {deleting ? "Deleting…" : "Delete permanently"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Checkbox({ checked, indeterminate = false, onChange, testId, label }) {
  const ref = useRef(null);

  useEffect(() => {
    // `indeterminate` is a DOM property with no HTML attribute, so React
    // cannot set it declaratively.
    if (ref.current) ref.current.indeterminate = indeterminate && !checked;
  }, [indeterminate, checked]);

  return (
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      onChange={onChange}
      data-testid={testId}
      aria-label={label}
      className="h-4 w-4 rounded border-ink-300 text-brand-600 focus:ring-2 focus:ring-brand-400 cursor-pointer accent-brand-600"
    />
  );
}
