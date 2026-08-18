# Knowledge Graph

Entities and relationships extracted from ingested documents, persisted to
SQLite, served over REST, and explored interactively in the web client.

> **No LLM is involved anywhere in this feature.** Extraction is spaCy
> dependency parsing plus deterministic rules. It runs offline on CPU, needs no
> API key, and costs nothing per document. The Gemini integration used by *Ask
> Knowledge* is entirely separate and is never called here.

---

## Install

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # ~12 MB, required for the graph
```

Without the spaCy model the platform still runs: documents ingest, chunk, embed
and become searchable, and the graph is simply empty. `NlpPipeline.is_available`
is False, a warning is logged once, and every extraction returns nothing.

---

## How entities are extracted

`app/knowledge/entities.py`, `app/knowledge/nlp.py`

1. **Sentence segmentation** — the document is parsed once by spaCy. Headings
   have no terminal punctuation and would otherwise merge into the following
   sentence (producing subjects like *"Acme Handbook Leadership Priya Sharma"*),
   so short unpunctuated lines get a full stop first. Character offsets are
   mapped back afterwards, so evidence still points at the right chunk.
2. **Typing** — the *head noun* is consulted before the NER label, because a
   general model has no enterprise vocabulary and tags `CFO`, `Project Atlas`
   and `the Engineering team` all as `ORG`:

   | Source | Types produced |
   | --- | --- |
   | Head noun (`ROLE_HEADS`, `DEPARTMENT_HEADS`, …) | `ROLE`, `DEPARTMENT`, `PROJECT`, `TECHNOLOGY` |
   | spaCy NER label | `PERSON`, `ORG`, `LOCATION`, `DATE`, `PRODUCT`, `EVENT`, `MONEY` |
   | Proper-noun fallback | `OTHER` |

3. **Rejection** — a span that is none of the above is **not** an entity. The
   old extractor turned every capitalised phrase into a node; the POS tagger
   distinguishes a real name (`PROPN`) from a merely capitalised common noun.
4. **Normalisation** (`normalization.py`) — a canonical key is built by
   lower-casing and stripping possessives, punctuation, determiners,
   honorifics and (for organisations only) legal suffixes, so
   `Microsoft Corporation` / `Microsoft Corp.` / `Microsoft` collapse to one
   node. A person's surname is never trimmed this way.
5. **Merging** — `EntityResolver` merges only on an exact canonical match, or
   a RapidFuzz similarity ≥ `KNOWLEDGE_GRAPH_ENTITY_MERGE_THRESHOLD` (default
   **92**) *with the same type*. Acronyms expand only when unambiguous. The bias
   is against merging: a wrong merge fuses two real people and corrupts every
   edge attached to them, while a missed merge just leaves two visible nodes.

`DATE` and `MONEY` are extracted but never linked — they connect to everything
and drown the graph.

---

## How relationships are extracted

`app/knowledge/relationships.py`

The previous implementation split a *line* on one of six hardcoded phrases and
kept whatever fell either side. This one walks the dependency parse:

```
sentence → predicate tokens → (subject, object) arguments
         → noun-phrase expansion → taxonomy → negation → confidence
```

Handled explicitly:

| Construction | Example | Result |
| --- | --- | --- |
| Active | `Acme acquired XYZ.` | `Acme —ACQUIRED→ XYZ` |
| Passive with inverse | `Finance is managed by Priya.` | `Finance —MANAGED_BY→ Priya` |
| Passive, self-inverse | `Helios is used by Engineering.` | `Engineering —USES→ Helios` (arguments swap) |
| Copular | `The payroll team is part of Finance.` | `payroll team —PART_OF→ Finance` |
| Relative clause | `Priya reports to the Head of Finance, who reports to the CFO.` | **both** edges |
| Conjunction | `X is part of Y and Y is managed by Z.` | **both** edges |
| Negation | `John does not report to Mary.` | recorded with `polarity = negative`, never asserted |

Argument spans are expanded from the head token so `Head` becomes
`Head of Finance`, bounded at 8 tokens so a runaway subtree cannot swallow a
clause.

### Relationship taxonomy

`app/knowledge/taxonomy.py` maps a lemmatised predicate (+ preposition) onto:

```
WORKS_IN  WORKS_FOR  REPORTS_TO  MANAGES  MANAGED_BY  OWNS  OWNED_BY
PART_OF  BELONGS_TO  RESPONSIBLE_FOR  LOCATED_IN  USES  DEVELOPS
PROVIDES  ACQUIRED  FOUNDED  CREATED  LEADS  MEMBER_OF  RELATED_TO
```

A predicate that matches nothing is **not discarded**: it becomes `RELATED_TO`
and keeps its original wording in `predicate`, e.g.
`type=RELATED_TO, predicate="collaborate with"`.

### Negation

`app/knowledge/negation.py` runs two independent checks:

* **Clause negation** — a `neg` dependency on the predicate, a negative
  quantifier subject (`nobody`, `no one`, `neither`), or a negated auxiliary
  (`cannot`).
* **Vacuous arguments** — pronouns and quantifiers (`anyone else`, `nothing`,
  `it`) are never entities.

Together these kill the exact false positive the old extractor produced:

```
"Valve is a flat organization where nobody reports to anyone else."
  old → Valve —REPORTS_TO→ anyone else      ← asserts the opposite
  new → (nothing)
```

### Confidence

Deterministic, from real parse signals — no random or hand-waved values:

| Signal | Effect |
| --- | --- |
| Base | `0.45` |
| Predicate matched the taxonomy (not `RELATED_TO`) | `+0.25` |
| Each argument typed (not `OTHER`) | `+0.10` each |
| Subject and object share the same head verb | `+0.05` |
| Arguments within 6 tokens | `+0.05`; beyond 15 | `−0.10` |
| Subject resolved through a relative pronoun | `−0.05` |
| Passive clause of a direction-sensitive type | `−0.05` |

Clamped to `[0, 1]`. Anything below
`KNOWLEDGE_GRAPH_RELATIONSHIP_THRESHOLD` (default **0.70**) is discarded.
Raising it trades recall for precision.

---

## Persistence

`app/repositories/sqlite_graph.py`

**Why SQLite.** The project had no relational store — Chroma holds vectors and
answers similarity queries, which cannot express neighbourhood traversal, type
filters or edge aggregation. SQLite adds no service and no dependency (it is in
the standard library) and lives as one file beside `chroma_db/`.

**Identity is deterministic.** `entity_id = sha1(type|canonical_name)` and
`relationship_id = sha1(source|type|target|polarity)`. Re-ingesting a document
therefore *updates* rows rather than duplicating them, and the same claim found
in three documents is **one edge with three pieces of evidence**.

### Schema

```sql
entities(id PK, canonical_name, display_name, type, description,
         mentions_count, confidence, created_at, updated_at)
  INDEX (canonical_name), (type)

entity_sources(entity_id FK, document_id, chunk_id)        PK (all three)
  INDEX (document_id)

relationships(id PK, source_id FK, target_id FK, type, predicate,
              confidence, polarity, created_at, updated_at)
  INDEX (source_id), (target_id), (type)

relationship_evidence(relationship_id FK, document_id, chunk_id,
                      sentence, confidence)                PK (rel, doc, sentence)
  INDEX (document_id)
```

Deleting a document removes its evidence and sources, then prunes edges with no
remaining evidence and entities with neither a source nor an edge.

---

## API

All routes under `/api/v1/knowledge-graph`. **Every read is bounded** — there is
deliberately no "return the whole graph" endpoint.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/knowledge-graph` | A filtered slice: `document_id`, `entity_type`, `relationship_type`, `search`, `min_confidence`, `limit` (≤500) |
| `GET` | `/knowledge-graph/stats` | Totals and per-type breakdowns |
| `GET` | `/knowledge-graph/search?q=` | Entities by name |
| `GET` | `/knowledge-graph/entities/{id}` | Entity, relationships, evidence, neighbours, sources |
| `GET` | `/knowledge-graph/entities/{id}/neighbors?depth=` | Breadth-first expansion (depth ≤ 3) |

Responses set `truncated: true` when a limit cut the result. Negative-polarity
edges are excluded from graph reads and stats, and appear in entity detail
clearly marked.

---

## Frontend

`frontend/src/pages/KnowledgeGraph.jsx`

* **[React Flow](https://reactflow.dev)** (`@xyflow/react`) — pan, zoom, fit,
  minimap and custom nodes out of the box.
* **Layout** — `lib/graphLayout.js`, a compact spring/repulsion simulation
  seeded from the node index rather than `Math.random`, so a graph lays out the
  same way every render. Pulling in dagre or elk for one screen was not worth
  the weight.
* Node colour encodes entity type; node size encodes degree within the slice.
* Clicking a node opens a side panel with its relationships, the **sentence each
  was extracted from**, and the source document name.
* Double-click or **Expand connections** merges that entity's neighbourhood in.
* Search dims everything outside the match's neighbourhood.
* Filters: entity type, relationship type, document, minimum confidence.

---

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `KNOWLEDGE_GRAPH_ENABLED` | `true` | Master switch |
| `KNOWLEDGE_GRAPH_RELATIONSHIP_THRESHOLD` | `0.70` | Minimum confidence to keep an edge |
| `KNOWLEDGE_GRAPH_ENTITY_MERGE_THRESHOLD` | `92.0` | RapidFuzz score (0–100) required to merge two names |
| `KNOWLEDGE_GRAPH_MAX_SENTENCE_CHARS` | `1000` | Longer "sentences" are parsing artefacts of tables |
| `KNOWLEDGE_GRAPH_DEFAULT_LIMIT` | `150` | Nodes returned by an unfiltered read |
| `GRAPH_DB_PATH` | `./knowledge_graph.db` | SQLite file |
| `SPACY_MODEL` | `en_core_web_sm` | Swap for `en_core_web_md`/`lg` for better recall |

---

## Limitations

* **Recall is bounded by `en_core_web_sm`.** Uncommon personal names are tagged
  `NOUN` rather than `PROPN` and are skipped rather than guessed —
  `Ravi manages the Engineering team` yields nothing, while
  `John manages the Engineering team` works. `en_core_web_md` or `lg` improves
  this at the cost of download size.
* **English only.** The parser, the taxonomy and the negation markers are all
  English.
* **Sentence-scoped.** A relationship spanning two sentences, or one that needs
  coreference resolution (`She reports to him`), is not extracted — pronouns are
  explicitly rejected as arguments.
* **No coreference.** `Acme` and `the company` remain separate nodes.
* **Extraction is not inference.** The graph records what documents *say*. It
  does not resolve contradictions between documents; both claims are stored with
  their own evidence.
* **`RELATED_TO` is a catch-all.** A meaningful share of edges land there with
  the original predicate preserved; they are real connections but weakly typed.
* **Layout is heuristic.** Very dense graphs (>200 visible nodes) will still
  overlap; filter or expand from a node instead.
