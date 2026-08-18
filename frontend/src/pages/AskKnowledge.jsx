import { useState } from "react";
import { motion } from "framer-motion";
import {
  Send,
  SlidersHorizontal,
  ChevronDown,
  ThumbsUp,
  ThumbsDown,
  Copy,
  MoreHorizontal,
  RefreshCw,
  Sparkles,
  Bookmark,
  Share2,
  ShieldCheck,
  FileText,
  ScrollText,
  Layers,
  ArrowRight,
  Brain,
  Search,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { askKnowledge } from "@/lib/api";
import { Card, FileTypeBadge, EmptyState } from "@/components/ui/Primitives.jsx";
import { classNames } from "@/lib/utils";

/** Project one backend RetrieveResult into the shape the source cards expect. */
function toSource(r) {
  const md = r.metadata || {};
  const name =
    md.section_title ||
    (md.chunk_id ? `Chunk ${md.chunk_id}` : "") ||
    md.document_id ||
    r.id;
  return {
    id: r.id,
    type: (md.file_type || "TXT").toUpperCase(),
    name,
    dept: md.entities ? md.entities : `via ${r.source} search`,
    page: md.section_id && md.section_id > 0 ? md.section_id : r.rank,
    score: r.score,
    text: r.text,
    rank: r.rank,
    retrievedBy: r.source,
    scores: r.scores || {},
    documentId: md.document_id,
  };
}

export default function AskKnowledge() {
  const [tab, setTab] = useState("answer");
  const [question, setQuestion] = useState("");
  const [followup, setFollowup] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const {
    topK, setTopK,
    reranking, toggleReranking,
    queryRewriting, toggleQueryRewriting,
    hybridSearch, toggleHybrid,
    semanticSearch, toggleSemantic,
  } = useAppStore();

  // Derive the retrieval mode from the search toggles. Hybrid wins when on;
  // otherwise fall back to semantic, then keyword.
  const mode = hybridSearch ? "hybrid" : semanticSearch ? "semantic" : "keyword";

  async function runQuery(q) {
    const text = (q ?? question).trim();
    if (!text || loading) return;
    setQuestion(text);
    setTab("answer");
    setLoading(true);
    setError(null);
    try {
      const data = await askKnowledge({
        query: text,
        topK,
        rerankTopK: reranking ? Math.min(topK, 5) : topK,
        mode,
        rewrite: queryRewriting,
      });
      setResult(data);
    } catch (e) {
      setError(e.message || "Something went wrong. Is the backend running?");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  // Sources come only from a live response; before that there is nothing to show.
  const sources = result ? result.results.map(toSource) : [];

  const keyPoints = result?.key_points ?? [];
  const relatedQuestions = result?.related_questions ?? [];

  const TABS = [
    { id: "answer", label: "Answer", icon: Sparkles },
    { id: "sources", label: "Sources", icon: ScrollText, count: sources.length },
    { id: "key", label: "Key Points", icon: Layers, count: keyPoints.length },
    { id: "related", label: "Related Questions", icon: Brain, count: relatedQuestions.length },
  ];

  return (
    <div className="px-8 py-6 grid grid-cols-[minmax(0,1fr)_320px] gap-6">
      {/* CENTER */}
      <div className="min-w-0 space-y-6">
        {/* Page heading */}
        <div>
          <div className="flex items-center gap-2 text-[11px] font-semibold tracking-[0.14em] uppercase text-brand-600 mb-2">
            <ShieldCheck className="h-3.5 w-3.5" />
            Grounded Answer
          </div>
          <h1 className="text-[32px] leading-[1.1] font-semibold tracking-tight text-ink-900">
            Ask Knowledge
          </h1>
          <p className="mt-2 text-[14px] text-ink-400 max-w-2xl">
            Ask a question about the documents you have indexed. Answers are
            generated only from retrieved passages, with citations.
          </p>
        </div>

        {/* Query panel */}
        <Card className="overflow-hidden">
          <div className="p-5">
            <div className="flex gap-4">
              <div className="h-9 w-9 shrink-0 rounded-full bg-gradient-to-br from-brand-400 to-brand-700 grid place-items-center text-white ring-2 ring-brand-100">
                <Search className="h-[16px] w-[16px]" strokeWidth={2} />
              </div>
              <div className="flex-1 min-w-0">
                <textarea
                  data-testid="question-input"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      runQuery();
                    }
                  }}
                  rows={2}
                  placeholder="Ask a question about your indexed documents..."
                  className="w-full resize-none text-[15.5px] leading-[1.55] text-ink-900 placeholder:text-ink-400 outline-none bg-transparent"
                />
                <div className="mt-3 flex items-center gap-2 flex-wrap">
                  <div className="ml-auto flex items-center gap-3 text-[12.5px] text-ink-500">
                    <Toggle label="Rewrite Query" on={queryRewriting} onChange={toggleQueryRewriting} testId="toggle-rewrite" />
                  </div>
                </div>
              </div>
              <button
                data-testid="submit-query"
                onClick={() => runQuery()}
                disabled={loading}
                className="self-start h-11 w-11 rounded-xl bg-gradient-to-b from-brand-500 to-brand-600 text-white grid place-items-center hover:from-brand-400 hover:to-brand-600 elev-2 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <Loader2 className="h-[18px] w-[18px] animate-spin" strokeWidth={2} />
                ) : (
                  <Send className="h-[18px] w-[18px]" strokeWidth={2} />
                )}
              </button>
            </div>
          </div>
        </Card>

        {/* Tabs */}
        <div>
          <div className="border-b border-ink-100 flex items-center gap-6 px-1">
            {TABS.map((t) => {
              const Icon = t.icon;
              const active = tab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  data-testid={`tab-${t.id}`}
                  className={classNames(
                    "relative h-11 inline-flex items-center gap-2 text-[13.5px] font-medium transition-colors",
                    active ? "text-brand-600" : "text-ink-500 hover:text-ink-800"
                  )}
                >
                  <Icon className="h-[15px] w-[15px]" strokeWidth={1.8} />
                  {t.label}
                  {t.count > 0 && (
                    <span className={classNames(
                      "ml-1 text-[11px] px-1.5 py-px rounded-full",
                      active ? "bg-brand-50 text-brand-600" : "bg-ink-100 text-ink-500"
                    )}>{t.count}</span>
                  )}
                  {active && (
                    <motion.span
                      layoutId="tab-underline"
                      className="absolute left-0 right-0 -bottom-px h-[2px] bg-brand-500"
                    />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Error banner */}
        {error && <ErrorBanner message={error} onRetry={() => runQuery()} />}

        {/* Answer content */}
        {tab === "answer" && (
          <AnswerCard
            result={result}
            loading={loading}
            onShowSources={() => setTab("sources")}
          />
        )}
        {tab === "sources" && <SourcesGrid sources={sources} />}
        {tab === "key" && <KeyPointsList points={keyPoints} generator={result?.generator} />}
        {tab === "related" && (
          <RelatedQuestionsList
            questions={relatedQuestions}
            generator={result?.generator}
            onAsk={runQuery}
          />
        )}

        {/* Shown with the answer, not only behind their tabs, because they are
            derived from the same grounded context in the same call. */}
        {tab === "answer" && keyPoints.length > 0 && (
          <InlineSection icon={Layers} title="Key Points">
            <ol className="space-y-2.5">
              {keyPoints.map((point, i) => (
                <li key={i} className="flex gap-3">
                  <span className="h-5 w-5 mt-0.5 shrink-0 grid place-items-center rounded-md bg-brand-50 text-brand-700 text-[11px] font-semibold">
                    {i + 1}
                  </span>
                  <span className="text-[13.5px] text-ink-800 leading-[1.6]">{point}</span>
                </li>
              ))}
            </ol>
          </InlineSection>
        )}

        {tab === "answer" && relatedQuestions.length > 0 && (
          <InlineSection icon={Brain} title="Related Questions">
            <div className="flex flex-wrap gap-2.5">
              {relatedQuestions.map((question, i) => (
                <button
                  key={i}
                  data-testid={`related-${i}`}
                  onClick={() => runQuery(question)}
                  className="px-3.5 h-9 rounded-full bg-white ring-1 ring-ink-100 text-[13px] text-brand-700 hover:bg-brand-50 hover:ring-brand-200 transition-all"
                >
                  {question}
                </button>
              ))}
            </div>
          </InlineSection>
        )}

        {/* Top Sources horizontal scroll */}
        {tab === "answer" && sources.length > 0 && (
          <TopSources sources={sources} count={sources.length} />
        )}

        {/* Follow-up */}
        {result && (
          <Card>
            <div className="p-4">
              <input
                data-testid="followup-input"
                value={followup}
                onChange={(e) => setFollowup(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && followup.trim()) {
                    runQuery(followup);
                    setFollowup("");
                  }
                }}
                placeholder="Ask a follow-up question..."
                className="w-full text-[14px] text-ink-900 placeholder:text-ink-400 outline-none bg-transparent py-1"
              />
              <div className="mt-3 flex items-center gap-2 flex-wrap">
                <button
                  data-testid="followup-send"
                  onClick={() => {
                    if (followup.trim()) {
                      runQuery(followup);
                      setFollowup("");
                    }
                  }}
                  className="ml-auto h-9 w-9 grid place-items-center rounded-lg bg-brand-500 hover:bg-brand-600 text-white"
                >
                  <Send className="h-[15px] w-[15px]" />
                </button>
              </div>
            </div>
          </Card>
        )}

        <p className="text-[11.5px] text-ink-400 px-1">
          Answers are generated by an LLM from retrieved passages. Verify important information against the cited sources.
        </p>
      </div>

      {/* RIGHT SIDEBAR */}
      <aside className="space-y-4 sticky top-[80px] self-start">
        {/* Every control here maps to a real parameter on POST /knowledge/ask. */}
        <Card>
          <div className="p-4">
            <div className="flex items-center gap-2 text-[12.5px] font-semibold text-ink-700 mb-3">
              <SlidersHorizontal className="h-[14px] w-[14px]" /> Search & Retrieval
            </div>
            <FieldRow label="Top K Results">
              <Select
                value={topK}
                onChange={(v) => setTopK(Number(v))}
                options={[3, 5, 8, 10, 15, 20].map((n) => ({ value: n, label: String(n) }))}
                testId="topk-select"
              />
            </FieldRow>
            <Divider />
            <ToggleRow label="Re-ranking" on={reranking} onChange={toggleReranking} testId="toggle-rerank" />
            <ToggleRow label="Query Rewriting" on={queryRewriting} onChange={toggleQueryRewriting} testId="toggle-qr" />
            <ToggleRow label="Hybrid Search" on={hybridSearch} onChange={toggleHybrid} testId="toggle-hybrid" />
            <ToggleRow label="Semantic Search" on={semanticSearch} onChange={toggleSemantic} testId="toggle-semantic" />
          </div>
        </Card>
      </aside>
    </div>
  );
}

function AnswerCard({ result, loading, onShowSources }) {
  if (loading) {
    return (
      <Card>
        <div className="p-6 space-y-3">
          <div className="flex items-center gap-2 text-[13.5px] text-ink-500">
            <Loader2 className="h-4 w-4 animate-spin text-brand-500" />
            Searching your knowledge base and generating an answer…
          </div>
          <div className="mt-2 space-y-2.5">
            <div className="h-3.5 rounded bg-ink-100 animate-pulse w-[92%]" />
            <div className="h-3.5 rounded bg-ink-100 animate-pulse w-[85%]" />
            <div className="h-3.5 rounded bg-ink-100 animate-pulse w-[74%]" />
          </div>
        </div>
      </Card>
    );
  }

  if (!result?.answer) {
    return (
      <EmptyState
        icon={Sparkles}
        title="No answer yet"
        description="Ask a question above to query your indexed documents. Nothing is shown until the backend returns a grounded answer."
      />
    );
  }

  const confidence = result.route?.confidence;

  return (
    <Card>
      <div className="p-6">
        <div className="flex gap-4">
          <div className="h-7 w-7 shrink-0 rounded-md bg-brand-50 grid place-items-center text-brand-600">
            <Sparkles className="h-[14px] w-[14px]" strokeWidth={2} />
          </div>
          <div className="flex-1 min-w-0 text-[14.5px] text-ink-800 leading-[1.7]">
            <div className="whitespace-pre-wrap">{result.answer}</div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-ink-100 flex items-center gap-2 flex-wrap">
          {confidence != null && (
            <Meta
              icon={ShieldCheck}
              label={`Confidence: ${confidence.toFixed(2)}`}
              success={confidence >= 0.7}
            />
          )}
          <Meta icon={Brain} label={`Model: ${result.generator}`} />
          <Meta icon={Layers} label={`Route: ${result.route?.route}`} />
          <Meta
            icon={ScrollText}
            label={`Retrieved: ${result.results.length} chunks`}
            onClick={onShowSources}
            testId="show-retrieved-chunks"
          />
          <div className="ml-auto flex items-center gap-1">
            <IconBtn icon={ThumbsUp} testId="rate-up" />
            <IconBtn icon={ThumbsDown} testId="rate-down" />
            <IconBtn icon={Copy} testId="copy" />
            <IconBtn icon={Bookmark} testId="bookmark" />
            <IconBtn icon={Share2} testId="share" />
            <IconBtn icon={MoreHorizontal} testId="more" />
          </div>
        </div>
      </div>
    </Card>
  );
}

function ErrorBanner({ message, onRetry }) {
  return (
    <div
      data-testid="ask-error"
      className="rounded-xl bg-rose-50 ring-1 ring-rose-200 p-4 flex items-start gap-3"
    >
      <AlertTriangle className="h-[18px] w-[18px] text-rose-500 shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <div className="text-[13.5px] font-semibold text-rose-800">Couldn't get an answer</div>
        <div className="text-[12.5px] text-rose-600 mt-0.5 break-words">{message}</div>
      </div>
      <button
        onClick={onRetry}
        className="h-8 px-3 inline-flex items-center gap-1.5 rounded-lg bg-white ring-1 ring-rose-200 text-[12.5px] text-rose-700 hover:bg-rose-100"
      >
        <RefreshCw className="h-3.5 w-3.5" /> Retry
      </button>
    </div>
  );
}

function TopSources({ sources, count }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div className="text-[13.5px] font-semibold text-ink-800 flex items-center gap-2">
          <ScrollText className="h-[14px] w-[14px] text-brand-600" />
          Top Sources <span className="text-ink-400 font-normal">({count ?? sources.length})</span>
        </div>
      </div>
      <div className="flex gap-3 overflow-x-auto scroll-stylish pb-3 -mx-1 px-1">
        {sources.map((s) => (
          <div
            key={s.id}
            data-testid={`source-${s.id}`}
            className="shrink-0 w-[244px] bg-white rounded-xl ring-1 ring-ink-100 hover:ring-brand-200 hover:elev-2 transition-all p-3.5"
          >
            <div className="flex items-start gap-3">
              <FileTypeBadge type={s.type} />
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-semibold text-ink-900 leading-tight line-clamp-2">{s.name}</div>
                <div className="mt-1 text-[11.5px] text-ink-400 truncate">{s.dept}</div>
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between text-[11.5px] text-ink-500">
              <span>Page {s.page}</span>
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200 font-mono">
                {s.score.toFixed(2)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * The retrieved chunks, in rank order.
 *
 * Each row expands to the exact passage the generator was given, so an answer
 * can be checked against its evidence. The bracketed number matches the `[n]`
 * citation markers in the answer text.
 */
function SourcesGrid({ sources }) {
  const [expanded, setExpanded] = useState(() => new Set());

  if (!sources.length) {
    return (
      <EmptyState
        icon={FileText}
        title="No sources yet"
        description="Retrieved passages and their relevance scores appear here once you ask a question."
      />
    );
  }

  const toggle = (id) =>
    setExpanded((current) => {
      const next = new Set(current);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  return (
    <Card>
      <div className="divide-y divide-ink-100">
        {sources.map((s, index) => {
          const open = expanded.has(s.id);
          return (
            <div key={s.id} data-testid={`chunk-${s.id}`}>
              <button
                type="button"
                onClick={() => toggle(s.id)}
                aria-expanded={open}
                className="w-full p-4 flex items-center gap-4 text-left hover:bg-ink-50/60"
              >
                <span className="text-[11.5px] font-mono text-ink-400 w-6 shrink-0">
                  [{index + 1}]
                </span>
                <FileTypeBadge type={s.type} />
                <div className="flex-1 min-w-0">
                  <div className="text-[13.5px] font-semibold text-ink-900 truncate">{s.name}</div>
                  <div className="text-[12px] text-ink-400 truncate">
                    {s.dept} · Page {s.page} · via {s.retrievedBy}
                  </div>
                </div>
                <span className="px-1.5 py-0.5 rounded-md bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200 font-mono text-[11.5px] shrink-0">
                  {s.score.toFixed(2)}
                </span>
                <ChevronDown
                  className={classNames(
                    "h-4 w-4 text-ink-400 shrink-0 transition-transform",
                    open && "rotate-180"
                  )}
                />
              </button>

              {open && (
                <div className="px-4 pb-4 pl-[74px] space-y-3">
                  <p className="text-[13px] leading-[1.65] text-ink-700 whitespace-pre-wrap bg-ink-50/70 rounded-lg p-3.5 ring-1 ring-ink-100">
                    {s.text}
                  </p>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {Object.entries(s.scores).map(([kind, value]) => (
                      <span
                        key={kind}
                        className="inline-flex items-center gap-1 h-6 px-2 rounded-md bg-white ring-1 ring-ink-100 text-[11px] text-ink-500"
                      >
                        {kind}
                        <span className="font-mono text-ink-800">{value.toFixed(3)}</span>
                      </span>
                    ))}
                    {s.documentId && (
                      <span className="text-[11px] text-ink-400 font-mono ml-1 truncate">
                        {s.documentId}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function InlineSection({ icon: Icon, title, children }) {
  return (
    <Card>
      <div className="p-5">
        <div className="text-[13.5px] font-semibold text-ink-800 flex items-center gap-2 mb-3.5">
          <Icon className="h-[14px] w-[14px] text-brand-600" />
          {title}
        </div>
        {children}
      </div>
    </Card>
  );
}

/**
 * Empty state for the derived tabs.
 *
 * The distinction matters: the extractive fallback cannot produce these at
 * all, which is a different situation from an LLM finding nothing to say.
 */
function NoExtras({ icon, what, generator }) {
  const extractive = generator && generator !== "gemini";
  return (
    <EmptyState
      icon={icon}
      title={`No ${what} for this answer`}
      description={
        extractive
          ? `This answer came from the '${generator}' generator, which quotes retrieved text and cannot derive ${what}. They appear when the LLM answers.`
          : `Ask a question and the model will derive ${what} from the same retrieved passages it answered from.`
      }
    />
  );
}

function KeyPointsList({ points, generator }) {
  if (!points.length) return <NoExtras icon={Layers} what="key points" generator={generator} />;

  return (
    <Card>
      <ul className="p-2">
        {points.map((point, i) => (
          <li key={i} className="flex gap-3 p-3 rounded-lg hover:bg-ink-50/60">
            <span className="h-6 w-6 shrink-0 grid place-items-center rounded-md bg-brand-50 text-brand-700 text-[12px] font-semibold">
              {i + 1}
            </span>
            <span className="text-[14px] text-ink-800 leading-relaxed">{point}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function RelatedQuestionsList({ questions, generator, onAsk }) {
  if (!questions.length)
    return <NoExtras icon={Brain} what="related questions" generator={generator} />;

  return (
    <Card>
      <ul className="divide-y divide-ink-100">
        {questions.map((question, i) => (
          <li
            key={i}
            onClick={() => onAsk?.(question)}
            data-testid={`related-tab-${i}`}
            className="p-4 flex items-center justify-between hover:bg-ink-50/60 group cursor-pointer"
          >
            <span className="text-[14px] text-ink-800">{question}</span>
            <ArrowRight className="h-4 w-4 text-ink-300 group-hover:text-brand-600 group-hover:translate-x-0.5 transition-all" />
          </li>
        ))}
      </ul>
    </Card>
  );
}

/* — UI bits — */
function Toggle({ label, on, onChange, testId }) {
  return (
    <button
      onClick={onChange}
      data-testid={testId}
      className="inline-flex items-center gap-2 group"
    >
      {label && <span className="text-ink-500">{label}</span>}
      <span className={classNames(
        "relative inline-flex h-[18px] w-[32px] rounded-full transition-colors",
        on ? "bg-brand-500" : "bg-ink-200"
      )}>
        <span className={classNames(
          "absolute top-0.5 h-[14px] w-[14px] rounded-full bg-white elev-1 transition-transform",
          on ? "translate-x-[15px]" : "translate-x-0.5"
        )} />
      </span>
    </button>
  );
}

function ToggleRow({ label, on, onChange, testId }) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-[13px] text-ink-700">{label}</span>
      <Toggle on={on} onChange={onChange} testId={testId} />
    </div>
  );
}

function FieldRow({ label, children }) {
  return (
    <div className="py-2">
      <div className="text-[11.5px] font-medium text-ink-500 mb-1.5">{label}</div>
      {children}
    </div>
  );
}

function Select({ value, onChange, options, testId }) {
  return (
    <div className="relative">
      <select
        data-testid={testId}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        className="w-full h-9 pl-3 pr-8 rounded-lg ring-1 ring-ink-100 bg-white text-[13px] text-ink-800 outline-none focus:ring-2 focus:ring-brand-400 appearance-none"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      <ChevronDown className="h-4 w-4 text-ink-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
    </div>
  );
}

function Divider() {
  return <div className="h-px bg-ink-100 my-2" />;
}

function Meta({ icon: Icon, label, success, onClick, testId }) {
  const className = classNames(
    "inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md text-[11.5px] ring-1",
    success ? "bg-emerald-50 text-emerald-700 ring-emerald-200" : "bg-ink-50 text-ink-600 ring-ink-100",
    onClick && "hover:bg-ink-100 hover:text-ink-900 hover:ring-ink-200 cursor-pointer transition-colors"
  );

  if (!onClick) {
    return (
      <span className={className}>
        <Icon className="h-3 w-3" />
        {label}
      </span>
    );
  }

  return (
    <button type="button" onClick={onClick} data-testid={testId} className={className}>
      <Icon className="h-3 w-3" />
      {label}
      <ArrowRight className="h-3 w-3 opacity-60" />
    </button>
  );
}

function IconBtn({ icon: Icon, testId }) {
  return (
    <button
      data-testid={testId}
      className="h-8 w-8 grid place-items-center rounded-md hover:bg-ink-100 text-ink-500 hover:text-ink-800"
    >
      <Icon className="h-[15px] w-[15px]" />
    </button>
  );
}
