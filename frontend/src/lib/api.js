// Thin fetch wrapper around the backend REST API.
//
// The base URL points at the FastAPI service's versioned prefix. Override it
// per-environment with VITE_API_BASE_URL (see .env.example); the default targets
// a locally running `uvicorn app.main:app` on port 8001.

const BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8001/api/v1"
).replace(/\/$/, "");

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, { method = "GET", body, headers, signal } = {}) {
  const isForm = body instanceof FormData;
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    signal,
    headers: {
      ...(isForm ? {} : body ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });

  let payload = null;
  const text = await res.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!res.ok) {
    const detail =
      (payload && (payload.detail || payload.error)) ||
      (typeof payload === "string" ? payload : null) ||
      `Request failed with status ${res.status}`;
    throw new ApiError(detail, res.status);
  }

  return payload;
}

/**
 * Run the full query pipeline and return a grounded answer.
 * @see POST /knowledge/ask
 */
export function askKnowledge(
  { query, topK = 20, rerankTopK = 5, mode = "hybrid", filters = null, rewrite = false },
  signal,
) {
  return request("/knowledge/ask", {
    method: "POST",
    signal,
    body: {
      query,
      top_k: topK,
      rerank_top_k: rerankTopK,
      mode,
      filters,
      rewrite,
    },
  });
}

/**
 * Retrieve relevant chunks without generating an answer.
 * @see POST /knowledge/retrieve
 */
export function retrieveChunks(
  { query, topK = 5, mode = "hybrid", filters = null },
  signal,
) {
  return request("/knowledge/retrieve", {
    method: "POST",
    signal,
    body: { query, top_k: topK, mode, filters },
  });
}

/**
 * Upload, process and index a document in one step.
 * @see POST /documents/upload-and-index
 */
export function uploadAndIndex(file, signal) {
  const form = new FormData();
  form.append("file", file);
  return request("/documents/upload-and-index", {
    method: "POST",
    body: form,
    signal,
  });
}

/**
 * List every document currently in the index.
 * @see GET /documents
 */
export function listDocuments(signal) {
  return request("/documents", { signal });
}

/**
 * Delete documents and every chunk belonging to them.
 * @see POST /documents/delete
 */
export function deleteDocuments(documentIds, signal) {
  return request("/documents/delete", {
    method: "POST",
    signal,
    body: { document_ids: documentIds },
  });
}

// ── Knowledge graph ──────────────────────────────────────────────

function queryString(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") search.set(key, value);
  });
  const query = search.toString();
  return query ? `?${query}` : "";
}

/**
 * Read a bounded slice of the knowledge graph.
 * @see GET /knowledge-graph
 */
export function fetchGraph(filters = {}, signal) {
  return request(`/knowledge-graph${queryString(filters)}`, { signal });
}

/** Graph totals and per-type breakdowns. @see GET /knowledge-graph/stats */
export function fetchGraphStats(signal) {
  return request("/knowledge-graph/stats", { signal });
}

/** Find entities by name. @see GET /knowledge-graph/search */
export function searchEntities(query, signal) {
  return request(`/knowledge-graph/search${queryString({ q: query })}`, { signal });
}

/** One entity, its relationships and its sources. */
export function fetchEntity(entityId, signal) {
  return request(`/knowledge-graph/entities/${encodeURIComponent(entityId)}`, { signal });
}

/** Expand the neighbourhood around an entity. */
export function fetchNeighbors(entityId, depth = 1, signal) {
  return request(
    `/knowledge-graph/entities/${encodeURIComponent(entityId)}/neighbors${queryString({ depth })}`,
    { signal },
  );
}

/** Liveness probe. @see GET /system/health */
export function checkHealth(signal) {
  return request("/system/health", { signal });
}

export { ApiError, BASE_URL };
