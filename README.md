# Enterprise Knowledge Intelligence Platform

Upload your documents, ask questions in plain English, and get answers that cite
the exact section they came from — plus a navigable graph of the people, teams
and projects your corpus talks about.

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Tests](https://img.shields.io/badge/tests-279%20passing-2E6A50)
![License](https://img.shields.io/badge/license-ISC-blue)

---

## Why this exists

Most "chat with your documents" demos hand the model a blob of text and hope.
This one is built around two commitments.

**Answers are grounded, never recalled.** The system retrieves first, packs a
token-budgeted context from real chunks, and only then asks a language model to
phrase what those sources already say. Every claim carries a citation that
resolves to a specific chunk and section heading. If the LLM is unreachable or
unconfigured, an extractive generator answers from the same passages instead —
degraded prose, identical citations, no hallucination surface either way.

**Every heavy component is swappable.** The vector store, embedder, reranker,
answer generator, document repository and file-format extractors are each a
Protocol in `app/ports/`. Swapping ChromaDB for Qdrant, or Gemini for a local
Llama, is an edit to one file: `app/api/dependencies.py`.

The knowledge graph adds a third: it runs entirely offline on CPU with **no LLM,
no API key and no network**, using spaCy dependency parsing plus deterministic
rules.

---

## Quick start

**Backend**

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm    # optional; the graph is empty without it
python -m uvicorn app.main:app --reload --port 8001
```

**Frontend** (second terminal)

```bash
cd frontend
npm install
npm run dev
```

Or run both from the repository root with `npm install && npm run dev`.

| Service | URL |
| --- | --- |
| API | http://localhost:8001/api/v1 |
| API docs (Swagger) | http://localhost:8001/docs |
| Web client | http://localhost:3000 |

**No configuration is required** — every setting has a working default and the
service boots unconfigured. Copy `backend/.env.example` to `backend/.env` to
override anything, including a `GEMINI_API_KEY` for LLM-generated answers.
Without a key, answer generation and query rewriting fall back to their offline
counterparts and the platform still works end to end.

### Try it in 30 seconds

The repo ships with [`sample-docs/`](sample-docs/) — six documents in five
formats describing one fictional company, deliberately cross-referencing each
other so retrieval and the graph have something real to work with.

```bash
for f in sample-docs/*; do
  [ "$(basename "$f")" = "README.md" ] && continue
  curl -s -X POST http://localhost:8001/api/v1/documents/upload-and-index -F "file=@$f"
done

curl -X POST http://localhost:8001/api/v1/knowledge/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Who owns Project Atlas and which team delivers it?", "mode": "hybrid"}'
```

> **Marcus Webb owns Project Atlas, and it is delivered by the Platform team [1].**

That answer is assembled from two different files in two different formats.

---

## How it works

### Ingestion — 11 stages

`app/ingestion/pipeline.py` sequences these and contains no logic of its own:

```
Document
  -> Text Extraction         per-format, registry-dispatched
  -> Cleaning                whitespace, control chars, artefacts
  -> Metadata                filename, type, pages, timestamps
  -> Structural Chunking     splits on markdown headings
  -> Entity Extraction       spaCy NER + head-noun typing
  -> Relationship Extraction dependency-parse patterns, scored
  -> Knowledge Graph         persisted to SQLite
  -> Recursive Chunking      oversized sections split with overlap
  -> Chunk Enrichment        section title + entities attached
  -> Embeddings              local sentence-transformer
  -> Indexing                ChromaDB + BM25 index invalidation
```

Every extractor normalises to markdown-flavoured text, which is what lets a
DOCX or a web page be sectioned the same way a PDF is.

### Query — 7 stages

`app/query/pipeline.py`. The response exposes the output of **every** stage,
which is what makes an answer auditable rather than merely plausible:

```
User Query
  -> Query Understanding   intent, language, entities, complexity score
  -> Query Rewriting       optional; LLM with heuristic fallback
  -> Hybrid Retrieval      dense + BM25, fused by Reciprocal Rank Fusion
  -> Reranking             against the user's own words, not the rewrite
  -> Context Building      token budget + stable citation numbering
  -> Answer Routing        direct | local_llm | online_llm, with a reason
  -> Answer Generation     registry maps route to generator
```

A single response returns the answer, its citations, the ranked evidence with
**per-strategy scores** (semantic, keyword, fusion, rerank), the routing
decision and why it was made, and suggested follow-up questions.

---

## Architecture

Layered so dependencies only ever point inwards:

```
api/            HTTP: routers, schemas, error mapping
  |
application/    Use cases: IngestionService, KnowledgeService, GraphService
  |
ingestion/ query/ retrieval/ indexing/ knowledge/    Pipeline stages
  |
ports/          Protocols the stages depend on          <- the seam
  |
embeddings/ vectorstores/ repositories/ llm/           Adapters
  |
domain/         Framework-free models
core/           Config, logging, exceptions
```

`api/dependencies.py` is the **composition root** — the only file that knows
both a Protocol and its implementation. Each provider is `lru_cache`d, so the
embedding model loads once per process.

### Extension points

Implement the Protocol, register the adapter, change nothing else.

| To add | Implement | Register in |
| --- | --- | --- |
| A file format (XLSX, EPUB) | `TextExtractor` | `default_extractors()` |
| A vector store (Qdrant, pgvector) | `VectorStore` | `get_vector_store()` |
| An embedding model | `Embedder` | `get_embedder()` |
| A cross-encoder reranker | `Reranker` | `get_query_pipeline()` |
| An LLM answer generator | `AnswerGenerator` | `get_answer_generators()` |
| A query rewriter | `QueryRewriter` | `get_query_rewriter()` |
| Durable document storage | `DocumentRepository` | `get_document_repository()` |
| Graph persistence | `GraphRepository` | `get_graph_repository()` |

---

## API

All 17 routes sit under `/api/v1`, documented live at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/documents` | List every indexed document |
| `GET` | `/documents/formats` | File extensions ingestion accepts |
| `POST` | `/documents/upload` | Process a document, return a `document_id` |
| `POST` | `/documents/{id}/index` | Embed and index a processed document |
| `POST` | `/documents/upload-and-index` | Both steps in one call |
| `POST` | `/documents/ingest-url` | Fetch a web page and ingest it |
| `GET` | `/documents/{id}` | Metadata for a processed document |
| `POST` | `/documents/delete` | Delete several documents by id |
| `DELETE` | `/documents/{id}` | Delete one document and its chunks |
| `POST` | `/knowledge/retrieve` | Hybrid search; ranked chunks, no generation |
| `POST` | `/knowledge/ask` | Full pipeline; answer with citations |
| `GET` | `/knowledge-graph` | Filtered slice of the entity graph |
| `GET` | `/knowledge-graph/stats` | Totals and per-type breakdowns |
| `GET` | `/knowledge-graph/search` | Find entities by name |
| `GET` | `/knowledge-graph/entities/{id}` | Entity, relationships and evidence |
| `GET` | `/knowledge-graph/entities/{id}/neighbors` | Expand a neighbourhood |
| `GET` | `/system/health` | Liveness probe |

Retrieval accepts a `mode` of `semantic`, `keyword` or `hybrid`, plus exact
metadata `filters` such as `{"document_id": "..."}`.

---

## Supported sources

| Source | Extensions | Notes |
| --- | --- | --- |
| PDF | `.pdf` | Per-page text via PyMuPDF; scanned pages need OCR first |
| Word | `.docx` | Heading styles become sections; tables flattened to labelled rows |
| HTML | `.html` `.htm` `.xhtml` | Scripts, styles and navigation stripped |
| Markdown | `.md` `.markdown` `.mdx` | Headings kept, emphasis removed |
| Plain text | `.txt` `.text` `.log` `.rst` | Read verbatim |
| Tabular | `.csv` `.tsv` | Rows become `Column: value` so values keep their labels |
| Notion export | `.zip` | Markdown + CSV archive; page-id suffixes stripped |
| Confluence export | `.zip` | HTML archive |
| Web page | *(URL)* | `POST /documents/ingest-url`, fetched server-side |

Archives run each member back through the same registry, so any format
supported standalone is supported inside an export. Members with no extractor
(images, attachments, stylesheets) are skipped, not fatal.

### Security defaults

URL ingestion **refuses private, loopback and link-local addresses**, because
the URL comes from a client and the server would otherwise fetch `localhost`
and cloud metadata endpoints on their behalf. Override with
`WEB_FETCH_ALLOW_PRIVATE_HOSTS=true` for local development only. Archive
ingestion caps member count and total uncompressed bytes against zip bombs.

---

## Knowledge graph

Entities and relationships are extracted during ingestion, scored
deterministically, persisted to SQLite and explored interactively at `/graph`.

**No language model is involved anywhere in this feature.** It runs offline on
CPU, needs no API key, and costs nothing per document.

The design bias is **precision over recall**, encoded in the defaults:

- A relationship scoring below **0.70** is discarded. Twenty accurate edges beat
  two hundred speculative ones.
- Two names merge only on an exact canonical match, or fuzzy similarity of
  **92** or above *with the same type*. A wrong merge fuses two real people and
  corrupts every edge attached to them; a missed merge just leaves two visible
  nodes.
- Typing consults the **head noun** before the NER label, because a general
  model has no enterprise vocabulary and tags `CFO`, `Project Atlas` and
  `the Engineering team` all as `ORG`.
- `DATE` and `MONEY` are extracted but never linked — they connect to
  everything and would drown the graph.

See [docs/knowledge-graph.md](docs/knowledge-graph.md) for the extraction rules,
schema, confidence model and limitations.

---

## Configuration

Every tunable lives in one `Settings` class, so no pipeline stage hard-codes a
magic number. Unknown keys in `.env` are ignored rather than fatal, so a stale
environment never blocks a boot.

| Group | Governs | Default |
| --- | --- | --- |
| Chunking | Size and overlap | `600 / 75` |
| Embeddings | Model | `BAAI/bge-small-en-v1.5` |
| Retrieval | Top-k, rerank top-k, RRF constant | `20 / 5 / 60` |
| Context | Token budget for the prompt | `2500` |
| Routing | Confidence and complexity thresholds | `0.9 / 0.7` |
| Knowledge graph | Edge threshold, merge threshold, node cap | `0.70 / 92 / 150` |
| Web fetch | Timeout, size cap, private-host policy | `30s / 10 MB / denied` |
| Archives | Member cap, uncompressed byte cap | `2000 / 256 MB` |
| LLM | Key, model, timeout, temperature | *unset — falls back offline* |

---

## Tests

```bash
cd backend
python -m pytest                  # 279 tests
python -m pytest -m integration   # additionally requires a running Ollama
```

Tests substitute in-memory adapters for the embedder and vector store, so the
suite exercises the **real** pipelines — same code paths, same sequencing —
with no model download and no network call. That is the payoff of writing every
stage against a Protocol, and it is what keeps the suite fast enough to run on
every change.

---

## Known limits

Stated plainly, because they shape what this is good for.

- **Processed documents are held in memory.** `InMemoryDocumentRepository`
  bridges the upload to index flow; after a restart, `/documents/{id}` and
  `/documents/{id}/index` return 404 for documents uploaded before it. Indexed
  vectors survive in ChromaDB, and `GET /documents` reads the index rather than
  the repository, so the document list survives too.
- **No authentication.** No route is guarded. This is not multi-tenant and is
  not ready to face the public internet as-is.
- **Ingestion is synchronous.** A large PDF holds an HTTP connection open for
  its whole pipeline run.
- **Scanned PDFs yield nothing** — there is no OCR stage.
- **English only.** The spaCy model, tokenizer and heuristics all assume it.
  Other languages embed and retrieve, but produce no useful graph.
- **The web client covers four pages.** Ask, Documents, Knowledge Graph and Home
  are wired to the API. The remaining routes in the sidebar are placeholders
  that fix the information architecture.
- **LLM answers leave the machine.** Everything else — embedding, retrieval,
  graph extraction — is local. Unsetting the API key removes even that.

---

## License

ISC
