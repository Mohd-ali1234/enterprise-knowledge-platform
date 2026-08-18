import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  Handle,
  Position,
  MarkerType,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Network,
  Search,
  Loader2,
  AlertTriangle,
  X,
  Maximize2,
  FileText,
  Layers,
  Quote,
  SlidersHorizontal,
} from "lucide-react";
import { PageHeader, Card, EmptyState } from "@/components/ui/Primitives.jsx";
import {
  fetchGraph,
  fetchGraphStats,
  fetchEntity,
  fetchNeighbors,
  listDocuments,
} from "@/lib/api";
import { layoutGraph } from "@/lib/graphLayout";
import { classNames } from "@/lib/utils";

/** One colour per entity type, so a type is readable without the legend. */
const TYPE_COLORS = {
  PERSON: "#4f5bf2",
  ORG: "#16a571",
  DEPARTMENT: "#0ea5e9",
  ROLE: "#d18a1c",
  LOCATION: "#d44343",
  PROJECT: "#8b5cf6",
  TECHNOLOGY: "#0891b2",
  PRODUCT: "#db2777",
  EVENT: "#65a30d",
  DATE: "#6b7280",
  MONEY: "#059669",
  OTHER: "#94a3b8",
};

const RELATIONSHIP_TYPES = [
  "WORKS_IN", "WORKS_FOR", "REPORTS_TO", "MANAGES", "MANAGED_BY", "OWNS",
  "OWNED_BY", "PART_OF", "BELONGS_TO", "RESPONSIBLE_FOR", "LOCATED_IN",
  "USES", "DEVELOPS", "PROVIDES", "ACQUIRED", "FOUNDED", "CREATED", "LEADS",
  "MEMBER_OF", "RELATED_TO",
];

const colorFor = (type) => TYPE_COLORS[type] || TYPE_COLORS.OTHER;

/* ── Custom node ─────────────────────────────────────────────── */

function EntityNode({ data, selected }) {
  const color = colorFor(data.type);
  // Size carries degree: the busiest entities read as the hubs they are.
  const scale = Math.min(1.5, 1 + (data.degree || 0) * 0.06);

  return (
    <div
      className={classNames(
        "rounded-xl bg-white px-3 py-2 transition-all",
        selected ? "ring-2 elev-2" : "ring-1 hover:ring-2",
        data.dimmed && "opacity-25",
      )}
      style={{
        borderLeft: `4px solid ${color}`,
        boxShadow: selected ? `0 0 0 2px ${color}` : undefined,
        ringColor: color,
        minWidth: 110 * scale,
      }}
    >
      <Handle type="target" position={Position.Top} className="!opacity-0" />
      <div className="text-[12.5px] font-semibold text-ink-900 leading-tight max-w-[190px] truncate">
        {data.label}
      </div>
      <div className="mt-0.5 flex items-center gap-1.5">
        <span
          className="text-[10px] font-semibold uppercase tracking-wider"
          style={{ color }}
        >
          {data.type}
        </span>
        {data.mentions > 1 && (
          <span className="text-[10px] text-ink-400">· {data.mentions} mentions</span>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="!opacity-0" />
    </div>
  );
}

const NODE_TYPES = { entity: EntityNode };

/* ── Canvas ──────────────────────────────────────────────────── */

function GraphCanvas({ graph, selectedId, highlighted, onSelect, onExpand }) {
  const { fitView } = useReactFlow();

  const positions = useMemo(
    () => layoutGraph(graph.nodes, graph.edges),
    [graph.nodes, graph.edges],
  );

  const nodes = useMemo(
    () =>
      graph.nodes.map((node) => ({
        id: node.id,
        type: "entity",
        position: positions[node.id] || { x: 0, y: 0 },
        data: {
          label: node.label,
          type: node.type,
          degree: node.degree,
          mentions: node.mentions_count,
          dimmed: highlighted.size > 0 && !highlighted.has(node.id),
        },
        selected: node.id === selectedId,
      })),
    [graph.nodes, positions, selectedId, highlighted],
  );

  const edges = useMemo(
    () =>
      graph.edges.map((edge) => {
        const active =
          edge.source === selectedId ||
          edge.target === selectedId ||
          (highlighted.has(edge.source) && highlighted.has(edge.target));
        const dimmed = highlighted.size > 0 && !active;

        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.type.replace(/_/g, " ").toLowerCase(),
          animated: active && selectedId != null,
          style: {
            stroke: active ? "#4f5bf2" : "#cbd5e1",
            strokeWidth: active ? 2 : 1.2,
            opacity: dimmed ? 0.15 : 1,
          },
          labelStyle: {
            fontSize: 10,
            fill: active ? "#4f5bf2" : "#64748b",
            opacity: dimmed ? 0.15 : 1,
          },
          labelBgStyle: { fill: "#ffffff", fillOpacity: dimmed ? 0.15 : 0.85 },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            width: 16,
            height: 16,
            color: active ? "#4f5bf2" : "#cbd5e1",
          },
        };
      }),
    [graph.edges, selectedId, highlighted],
  );

  // Re-fit whenever the node set changes, so an expansion stays in view.
  useEffect(() => {
    const timer = setTimeout(() => fitView({ padding: 0.2, duration: 400 }), 60);
    return () => clearTimeout(timer);
  }, [graph.nodes.length, fitView]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodeClick={(_, node) => onSelect(node.id)}
      onNodeDoubleClick={(_, node) => onExpand(node.id)}
      onPaneClick={() => onSelect(null)}
      fitView
      minZoom={0.1}
      maxZoom={2.5}
      proOptions={{ hideAttribution: true }}
      className="bg-ink-50/40"
    >
      <Background color="#cbd5e1" gap={20} size={1} />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

/* ── Side panel ──────────────────────────────────────────────── */

function EntityPanel({ detail, loading, documents, onClose, onExpand, onSelect }) {
  if (loading) {
    return (
      <aside className="w-[340px] shrink-0 border-l border-ink-100 bg-white p-5">
        <div className="flex items-center gap-2 text-[13px] text-ink-400">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading entity…
        </div>
      </aside>
    );
  }
  if (!detail) return null;

  const nameFor = (id) => documents[id] || id;

  return (
    <aside
      data-testid="entity-panel"
      className="w-[340px] shrink-0 border-l border-ink-100 bg-white overflow-y-auto"
    >
      <div className="p-5 border-b border-ink-100">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div
              className="text-[10px] font-semibold uppercase tracking-wider"
              style={{ color: colorFor(detail.type) }}
            >
              {detail.type}
            </div>
            <h2 className="text-[17px] font-semibold text-ink-900 mt-0.5 break-words">
              {detail.label}
            </h2>
            <div className="text-[12px] text-ink-400 mt-1">
              {detail.mentions_count} mention{detail.mentions_count === 1 ? "" : "s"} ·{" "}
              {detail.relationships.length} relationship
              {detail.relationships.length === 1 ? "" : "s"}
            </div>
          </div>
          <button
            onClick={onClose}
            data-testid="close-panel"
            className="h-7 w-7 grid place-items-center rounded-md text-ink-400 hover:bg-ink-100 shrink-0"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <button
          onClick={() => onExpand(detail.id)}
          data-testid="expand-node"
          className="mt-3 w-full h-9 inline-flex items-center justify-center gap-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-[13px] font-medium"
        >
          <Maximize2 className="h-3.5 w-3.5" /> Expand connections
        </button>
      </div>

      <Section icon={Layers} title={`Relationships (${detail.relationships.length})`}>
        {detail.relationships.length === 0 && (
          <p className="text-[12.5px] text-ink-400">No relationships recorded.</p>
        )}
        <ul className="space-y-2.5">
          {detail.relationships.map((relationship) => {
            const outgoing = relationship.source_id === detail.id;
            const otherId = outgoing ? relationship.target_id : relationship.source_id;
            const otherLabel =
              (outgoing ? relationship.target_label : relationship.source_label) || otherId;
            // A negated statement is evidence that the relation does NOT hold.
            // Rendering it like any other row would assert the opposite of the
            // source sentence, which is the bug this graph exists to avoid.
            const negative = relationship.polarity === "negative";

            return (
              <li
                key={relationship.id}
                className={classNames(
                  "rounded-lg ring-1 p-2.5",
                  negative ? "ring-rose-200 bg-rose-50/40" : "ring-ink-100",
                )}
              >
                <div className="flex items-center gap-1.5 text-[12px]">
                  <span className="text-ink-400">{outgoing ? "→" : "←"}</span>
                  <span
                    className={classNames(
                      "font-semibold",
                      negative ? "text-rose-700 line-through" : "text-brand-700",
                    )}
                  >
                    {relationship.type.replace(/_/g, " ").toLowerCase()}
                  </span>
                  {negative && (
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-rose-600 bg-rose-100 rounded px-1.5 py-0.5">
                      negated
                    </span>
                  )}
                </div>
                <button
                  onClick={() => onSelect(otherId)}
                  className="mt-1 text-[13px] text-ink-900 hover:text-brand-700 text-left break-words"
                >
                  {otherLabel}
                </button>
                <div className="mt-1.5 flex items-center gap-2 text-[11px] text-ink-400">
                  <span className="font-mono">{relationship.confidence.toFixed(2)}</span>
                  {relationship.predicate && <span>· “{relationship.predicate}”</span>}
                </div>
                {/* Provenance: the sentence the relationship was read from. */}
                {relationship.evidence.slice(0, 2).map((item, i) => (
                  <div
                    key={i}
                    className="mt-2 rounded-md bg-ink-50/80 p-2 text-[11.5px] text-ink-600 leading-snug"
                  >
                    <Quote className="h-3 w-3 inline mr-1 text-ink-300" />
                    {item.sentence}
                    <div className="mt-1 text-[10.5px] text-ink-400 truncate">
                      {nameFor(item.document_id)}
                    </div>
                  </div>
                ))}
              </li>
            );
          })}
        </ul>
      </Section>

      <Section icon={FileText} title="Sources">
        <ul className="space-y-1.5 text-[12.5px] text-ink-700">
          {detail.source_documents.length === 0 && (
            <li className="text-ink-400">No source documents recorded.</li>
          )}
          {detail.source_documents.map((id) => (
            <li key={id} className="truncate">{nameFor(id)}</li>
          ))}
        </ul>
      </Section>
    </aside>
  );
}

function Section({ icon: Icon, title, children }) {
  return (
    <div className="p-5 border-b border-ink-100 last:border-0">
      <div className="text-[12.5px] font-semibold text-ink-700 flex items-center gap-2 mb-3">
        <Icon className="h-[14px] w-[14px] text-brand-600" /> {title}
      </div>
      {children}
    </div>
  );
}

/* ── Page ────────────────────────────────────────────────────── */

export default function KnowledgeGraph() {
  const [graph, setGraph] = useState({ nodes: [], edges: [], truncated: false });
  const [stats, setStats] = useState(null);
  const [documents, setDocuments] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [search, setSearch] = useState("");
  const [entityType, setEntityType] = useState("");
  const [relationshipType, setRelationshipType] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [minConfidence, setMinConfidence] = useState(0);

  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const expansion = useRef({ nodes: [], edges: [] });

  const filters = useMemo(
    () => ({
      search: search.trim() || undefined,
      entity_type: entityType || undefined,
      relationship_type: relationshipType || undefined,
      document_id: documentId || undefined,
      min_confidence: minConfidence || undefined,
    }),
    [search, entityType, relationshipType, documentId, minConfidence],
  );

  const load = useCallback(async (activeFilters, signal) => {
    try {
      const data = await fetchGraph(activeFilters, signal);
      expansion.current = { nodes: [], edges: [] };
      setGraph(data);
      setError(null);
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message || "Could not load the graph.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Filters are debounced so typing in the search box does not fire a request
  // per keystroke.
  useEffect(() => {
    const controller = new AbortController();
    // Everything runs inside the timeout, so no state update is reachable
    // synchronously from the effect body.
    const timer = setTimeout(() => {
      (async () => {
        setLoading(true);
        await load(filters, controller.signal);
      })();
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [filters, load]);

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const [statsData, documentData] = await Promise.all([
          fetchGraphStats(controller.signal),
          listDocuments(controller.signal).catch(() => ({ documents: [] })),
        ]);
        setStats(statsData);
        setDocuments(
          Object.fromEntries(documentData.documents.map((d) => [d.document_id, d.file_name])),
        );
      } catch (e) {
        if (e.name !== "AbortError") setStats(null);
      }
    })();
    return () => controller.abort();
  }, []);

  // Selecting a node loads its detail panel. Deselection is handled at render
  // time rather than here, so the effect body updates no state synchronously.
  useEffect(() => {
    if (!selectedId) return undefined;

    const controller = new AbortController();
    (async () => {
      setDetailLoading(true);
      setDetail(null); // drop the previous entity while the next one loads
      try {
        setDetail(await fetchEntity(selectedId, controller.signal));
      } catch (e) {
        if (e.name !== "AbortError") setDetail(null);
      } finally {
        setDetailLoading(false);
      }
    })();
    return () => controller.abort();
  }, [selectedId]);

  /** Merge an entity's neighbourhood into the current view. */
  const expand = useCallback(async (entityId) => {
    try {
      const neighbourhood = await fetchNeighbors(entityId, 1);
      setGraph((current) => {
        const nodeIds = new Set(current.nodes.map((n) => n.id));
        const edgeIds = new Set(current.edges.map((e) => e.id));
        return {
          ...current,
          nodes: [...current.nodes, ...neighbourhood.nodes.filter((n) => !nodeIds.has(n.id))],
          edges: [...current.edges, ...neighbourhood.edges.filter((e) => !edgeIds.has(e.id))],
        };
      });
    } catch {
      setError("Could not expand this entity.");
    }
  }, []);

  // Searching focuses the match and its immediate neighbourhood.
  const highlighted = useMemo(() => {
    const term = search.trim().toLowerCase();
    if (!term) return new Set();

    const matches = graph.nodes
      .filter((node) => node.label.toLowerCase().includes(term))
      .map((node) => node.id);
    const focus = new Set(matches);
    graph.edges.forEach((edge) => {
      if (focus.has(edge.source)) focus.add(edge.target);
      else if (focus.has(edge.target)) focus.add(edge.source);
    });
    return focus;
  }, [search, graph.nodes, graph.edges]);

  const hasFilters = Boolean(
    search || entityType || relationshipType || documentId || minConfidence,
  );

  return (
    <div className="flex flex-col h-[calc(100vh-64px)]">
      <PageHeader
        eyebrow="Entity Map"
        title="Knowledge Graph"
        subtitle="Entities and relationships extracted locally from your documents — no LLM involved."
        actions={
          stats && (
            <div className="flex items-center gap-4 text-[12.5px] text-ink-500">
              <span><strong className="text-ink-900">{stats.total_entities}</strong> entities</span>
              <span><strong className="text-ink-900">{stats.total_relationships}</strong> relationships</span>
              <span><strong className="text-ink-900">{stats.documents_with_graph_data}</strong> documents</span>
            </div>
          )
        }
      />

      <div className="px-8 pb-3">
        <Card className="p-3 flex items-center gap-2.5 flex-wrap">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-ink-400" />
            <input
              data-testid="graph-search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search entities..."
              className="w-full h-9 pl-9 pr-3 rounded-lg ring-1 ring-ink-100 focus:ring-2 focus:ring-brand-400 outline-none text-[13px] bg-ink-50/40"
            />
          </div>

          <Select value={entityType} onChange={setEntityType} testId="filter-entity-type"
            options={[{ value: "", label: "All entity types" },
              ...Object.keys(TYPE_COLORS).map((t) => ({ value: t, label: t }))]} />

          <Select value={relationshipType} onChange={setRelationshipType} testId="filter-relationship-type"
            options={[{ value: "", label: "All relationships" },
              ...RELATIONSHIP_TYPES.map((t) => ({ value: t, label: t.replace(/_/g, " ") }))]} />

          <Select value={documentId} onChange={setDocumentId} testId="filter-document"
            options={[{ value: "", label: "All documents" },
              ...Object.entries(documents).map(([id, name]) => ({ value: id, label: name }))]} />

          <label className="flex items-center gap-2 text-[12px] text-ink-500 pl-1">
            <SlidersHorizontal className="h-3.5 w-3.5" />
            Min confidence
            <input
              type="range" min="0" max="1" step="0.05" value={minConfidence}
              data-testid="filter-confidence"
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              className="w-24 accent-brand-600"
            />
            <span className="font-mono text-ink-800 w-8">{minConfidence.toFixed(2)}</span>
          </label>
        </Card>
      </div>

      <div className="flex-1 min-h-0 px-8 pb-8 flex gap-0">
        <div className="flex-1 min-w-0 rounded-l-2xl overflow-hidden ring-1 ring-ink-100 bg-white relative">
          {loading && (
            <div className="absolute inset-0 z-10 grid place-items-center bg-white/70">
              <div className="flex items-center gap-2 text-[13px] text-ink-500">
                <Loader2 className="h-4 w-4 animate-spin text-brand-500" /> Loading graph…
              </div>
            </div>
          )}

          {error ? (
            <div className="h-full grid place-items-center p-8">
              <div className="text-center max-w-sm">
                <AlertTriangle className="h-8 w-8 text-rose-500 mx-auto" />
                <div className="mt-3 text-[14px] font-semibold text-ink-900">
                  Could not load the graph
                </div>
                <p className="mt-1 text-[13px] text-ink-500">{error}</p>
              </div>
            </div>
          ) : !loading && graph.nodes.length === 0 ? (
            <div className="h-full grid place-items-center p-8">
              <EmptyState
                icon={Network}
                title={hasFilters ? "No entities match these filters" : "No graph data yet"}
                description={
                  hasFilters
                    ? "Try widening the filters, or lowering the minimum confidence."
                    : "Upload a document with named people, teams or organisations. Entities and relationships are extracted during ingestion."
                }
              />
            </div>
          ) : (
            <ReactFlowProvider>
              <GraphCanvas
                graph={graph}
                selectedId={selectedId}
                highlighted={highlighted}
                onSelect={setSelectedId}
                onExpand={expand}
              />
            </ReactFlowProvider>
          )}

          {graph.truncated && (
            <div className="absolute bottom-3 left-3 z-10 rounded-lg bg-amber-50 ring-1 ring-amber-200 px-3 py-1.5 text-[11.5px] text-amber-800">
              Showing a partial view — filter or expand a node to explore further.
            </div>
          )}
        </div>

        {selectedId && (detail || detailLoading) && (
          <div className="rounded-r-2xl overflow-hidden ring-1 ring-l-0 ring-ink-100 flex">
            <EntityPanel
              detail={detail}
              loading={detailLoading}
              documents={documents}
              onClose={() => setSelectedId(null)}
              onExpand={expand}
              onSelect={setSelectedId}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function Select({ value, onChange, options, testId }) {
  return (
    <select
      value={value}
      data-testid={testId}
      onChange={(e) => onChange(e.target.value)}
      className="h-9 px-2.5 rounded-lg ring-1 ring-ink-100 bg-white text-[12.5px] text-ink-800 outline-none focus:ring-2 focus:ring-brand-400 max-w-[190px]"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>{option.label}</option>
      ))}
    </select>
  );
}
