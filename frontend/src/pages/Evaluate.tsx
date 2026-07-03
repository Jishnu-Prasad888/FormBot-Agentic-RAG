import React, { useState, useRef, useCallback } from "react";
import {
  FlaskConical,
  Plus,
  Trash2,
  Zap,
  AlertTriangle,
  Upload,
  Download,
  ChevronDown,
  ChevronRight,
  FileText,
  X,
  Check,
} from "lucide-react";
import { ragEvaluate } from "../api/client";
import Spinner from "../components/Spinner";
import ScoreBar from "../components/ScoreBar";
import type { EvaluationResponse } from "../types";
import { formatLatency } from "../utils/format";

interface Props {
  onToast: (type: any, msg: string) => void;
}

interface QA {
  question: string;
  expected_answer: string;
}

/** Per-question evaluation result returned by the backend */
interface QuestionResult {
  question: string;
  expected_answer: string;
  generated_answer: string;
  retrieved_context: string;
  // LLM-as-judge
  accuracy_llm?: number;
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  // Accuracy methods
  exact_match?: number;
  semantic_similarity?: number;
  f1?: number;
  accuracy_combined?: number;
  // Retrieval metrics
  recall_10?: number;
  recall_20?: number;
  recall_50?: number;
  mrr?: number;
  ndcg_10?: number;
  gold_answer_found?: boolean;
  // Rationales
  accuracy_rationale: string;
  faithfulness_rationale: string;
  answer_relevancy_rationale: string;
  context_precision_rationale: string;
  context_recall_rationale: string;
  // Meta
  latency_ms: number;
  error?: string;
}

interface EvalSummary {
  accuracy_llm: number;
  accuracy_combined: number;
  faithfulness: number;
  context_precision: number;
  context_recall: number;
  answer_relevancy: number;
  recall_10: number;
  recall_20: number;
  recall_50: number;
  mrr: number;
  ndcg_10: number;
  latency_avg_ms: number;
  failed_questions: { question: string; error: string }[];
  per_question: QuestionResult[];
}

interface RunMode {
  mode: "all" | "individual";
}

const SAMPLE_QAS: QA[] = [
  {
    question: "What is the main topic of the indexed documents?",
    expected_answer:
      "The documents cover government schemes and eligibility criteria.",
  },
  {
    question: "Who is eligible for the SCSS?",
    expected_answer:
      "Senior citizens aged 60 and above are eligible for the SCSS.",
  },
];

const METRIC_COLORS: Record<string, string> = {
  accuracy_llm: "#00d4ff",
  accuracy_combined: "#00d4ff",
  faithfulness: "#00ff9f",
  context_precision: "#ffe600",
  context_recall: "#ff4d6d",
  answer_relevancy: "#a78bfa",
  exact_match: "#00ff9f",
  semantic_similarity: "#a78bfa",
  f1: "#ffe600",
  recall_10: "#00d4ff",
  recall_20: "#ff8c00",
  recall_50: "#ff4d6d",
  mrr: "#a78bfa",
  ndcg_10: "#00ff9f",
};

const METRIC_LABELS: Record<string, string> = {
  accuracy_llm: "Accuracy (LLM)",
  accuracy_combined: "Accuracy (Combined)",
  faithfulness: "Faithfulness",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
  answer_relevancy: "Answer Relevancy",
  exact_match: "Exact Match",
  semantic_similarity: "Semantic Similarity",
  f1: "F1 Score",
  recall_10: "Recall@10",
  recall_20: "Recall@20",
  recall_50: "Recall@50",
  mrr: "MRR",
  ndcg_10: "nDCG@10",
};

const LLM_JUDGE_METRICS = ["accuracy_llm", "faithfulness", "context_precision", "context_recall", "answer_relevancy"] as const;
const ACCURACY_METRICS = ["exact_match", "semantic_similarity", "f1", "accuracy_combined"] as const;
const RETRIEVAL_METRICS = ["recall_10", "recall_20", "recall_50", "mrr", "ndcg_10"] as const;

const METRIC_KEYS = Object.keys(METRIC_LABELS) as (keyof typeof METRIC_LABELS)[];

// ─── CSV helpers ──────────────────────────────────────────────────────────────

function parseCSV(text: string): QA[] {
  const lines = text.trim().split("\n").filter(Boolean);
  const qas: QA[] = [];
  // Skip header row (Question no,Eval Question,Eval Answer)
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length < 3) continue;
    // Col 0 = question no (ignored), col 1 = question, col 2+ = answer (rejoin commas)
    const question = parts[1]?.trim().replace(/^"|"$/g, "") || "";
    const answer = parts.slice(2).join(",").trim().replace(/^"|"$/g, "") || "";
    if (question && answer) qas.push({ question, expected_answer: answer });
  }
  return qas;
}

function exportToCSV(rows: QuestionResult[]) {
  const escape = (v: any) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csvRows = rows.map((r, i) =>
    [
      i + 1,
      escape(r.question),
      escape(r.expected_answer),
      escape(r.generated_answer),
      escape(r.retrieved_context),
      // LLM-as-judge
      r.accuracy_llm ?? 0,
      r.faithfulness,
      r.context_precision,
      r.context_recall,
      r.answer_relevancy,
      // Accuracy methods
      r.exact_match ?? 0,
      r.semantic_similarity ?? 0,
      r.f1 ?? 0,
      r.accuracy_combined ?? 0,
      // Retrieval metrics
      r.recall_10 ?? 0,
      r.recall_20 ?? 0,
      r.recall_50 ?? 0,
      r.mrr ?? 0,
      r.ndcg_10 ?? 0,
      r.gold_answer_found ? "Yes" : "No",
      escape(r.accuracy_rationale),
      escape(r.faithfulness_rationale),
      escape(r.context_precision_rationale),
      escape(r.context_recall_rationale),
      escape(r.answer_relevancy_rationale),
      r.latency_ms,
    ].join(","),
  );
  const headers = [
    "Question No",
    "Question",
    "Expected Answer",
    "Generated Answer",
    "Retrieved Context",
    "Accuracy (LLM)",
    "Faithfulness",
    "Context Precision",
    "Context Recall",
    "Answer Relevancy",
    "Exact Match",
    "Semantic Similarity",
    "F1 Score",
    "Accuracy (Combined)",
    "Recall@10",
    "Recall@20",
    "Recall@50",
    "MRR",
    "nDCG@10",
    "Gold Answer Found",
    "Accuracy Rationale",
    "Faithfulness Rationale",
    "Context Precision Rationale",
    "Context Recall Rationale",
    "Answer Relevancy Rationale",
    "Latency (ms)",
  ];
  const blob = new Blob([[headers.join(","), ...csvRows].join("\n")], {
    type: "text/csv",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `rag_eval_${Date.now()}.csv`;
  a.click();
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function GradeChip({ score }: { score: number }) {
  const s = isNaN(score) ? 0 : score;
  const { label, color } =
    s >= 0.8
      ? { label: "EXCELLENT", color: "#00ff9f" }
      : s >= 0.6
        ? { label: "GOOD", color: "#ffe600" }
        : s >= 0.4
          ? { label: "FAIR", color: "#ff8c00" }
          : { label: "POOR", color: "#ff4d6d" };
  return (
    <span
      className="text-xs font-bold px-2 py-0.5"
      style={{
        fontFamily: "Space Mono, monospace",
        color,
        border: `1px solid ${color}`,
        background: color + "18",
        fontSize: "0.6rem",
      }}
    >
      {label}
    </span>
  );
}

function QuestionRow({
  row,
  idx,
  onRunIndividual,
  isRunning,
}: {
  row: QuestionResult;
  idx: number;
  onRunIndividual: (idx: number) => void;
  isRunning: boolean;
}) {
  const [open, setOpen] = useState(false);
  
  // Calculate average of LLM-as-judge metrics
  const llmMetrics = [
    row.accuracy_llm ?? 0,
    row.faithfulness,
    row.context_precision,
    row.context_recall,
    row.answer_relevancy,
  ];
  const avg = llmMetrics.reduce((a, b) => a + b, 0) / llmMetrics.length;

  // Gold answer found indicator
  const goldFound = row.gold_answer_found ?? false;

  return (
    <div
      className="border border-[#1e2d54] mb-2"
      style={{ background: "#080e20" }}
    >
      {/* Summary row */}
      <button
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-[#0f1629] transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? (
          <ChevronDown size={11} color="#4a5a8e" />
        ) : (
          <ChevronRight size={11} color="#4a5a8e" />
        )}
        <span
          className="text-xs font-bold w-6 text-[#4a5a8e]"
          style={{ fontFamily: "Space Mono, monospace" }}
        >
          #{idx + 1}
        </span>
        <span
          className="flex-1 text-xs truncate text-[#8a9abb]"
          style={{ fontFamily: "IBM Plex Mono, monospace" }}
        >
          {row.question}
        </span>
        {!row.error && <Check size={12} color="#00ff9f" />}
        <button
          className="text-xs px-2 py-1 border border-[#a78bfa] text-[#a78bfa] hover:bg-[#a78bfa] hover:text-[#0a0e1b] transition-colors"
          onClick={(e) => {
            e.stopPropagation();
            onRunIndividual(idx);
          }}
          disabled={isRunning}
          style={{
            opacity: isRunning ? 0.5 : 1,
            cursor: isRunning ? "not-allowed" : "pointer",
          }}
        >
          {isRunning ? "Running..." : "Run"}
        </button>
        {row.error ? (
          <span
            className="text-xs text-[#ff4d6d]"
            style={{ fontFamily: "Space Mono, monospace", fontSize: "0.6rem" }}
          >
            ERROR
          </span>
        ) : (
          <>
            <span
              className="text-xs font-bold w-12 text-right"
              style={{
                fontFamily: "Space Mono, monospace",
                color: avg >= 0.6 ? "#00ff9f" : "#ff4d6d",
              }}
            >
              {(avg * 100).toFixed(0)}%
            </span>
            <GradeChip score={avg} />
          </>
        )}
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="px-4 pb-4 space-y-4 border-t border-[#1e2d54]">
          {row.error ? (
            <div
              className="pt-3 text-xs text-[#ff4d6d]"
              style={{ fontFamily: "IBM Plex Mono, monospace" }}
            >
              Error: {row.error}
            </div>
          ) : (
            <>
              {/* LLM-as-Judge Metrics */}
              <div className="space-y-2 pt-3">
                <div
                  className="text-xs font-bold uppercase tracking-widest"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#a78bfa",
                    fontSize: "0.6rem",
                  }}
                >
                  LLM-as-Judge Metrics
                </div>
                {LLM_JUDGE_METRICS.map((key) => (
                  <div key={key} className="flex items-start gap-3 pl-2">
                    <div className="w-32 flex-shrink-0">
                      <div
                        className="text-xs font-bold mb-0.5"
                        style={{
                          fontFamily: "Space Mono, monospace",
                          color: METRIC_COLORS[key],
                          fontSize: "0.6rem",
                        }}
                      >
                        {METRIC_LABELS[key]}
                      </div>
                      <ScoreBar
                        score={(row as any)[key] || 0}
                        color={METRIC_COLORS[key]}
                      />
                    </div>
                    <div
                      className="text-xs text-white pt-0.5"
                      style={{
                        fontFamily: "IBM Plex Mono, monospace",
                        fontSize: "0.65rem",
                        lineHeight: "1.5",
                      }}
                    >
                      {(row as any)[`${key}_rationale`] || "—"}
                    </div>
                  </div>
                ))}
              </div>

              {/* Accuracy Methods */}
              <div className="space-y-2 pt-3">
                <div
                  className="text-xs font-bold uppercase tracking-widest"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#00ff9f",
                    fontSize: "0.6rem",
                  }}
                >
                  Accuracy Evaluation
                </div>
                {ACCURACY_METRICS.map((key) => (
                  <div key={key} className="flex items-center gap-3 pl-2">
                    <div className="w-32 flex-shrink-0">
                      <div
                        className="text-xs font-bold mb-0.5"
                        style={{
                          fontFamily: "Space Mono, monospace",
                          color: METRIC_COLORS[key],
                          fontSize: "0.6rem",
                        }}
                      >
                        {METRIC_LABELS[key]}
                      </div>
                      <ScoreBar
                        score={(row as any)[key] || 0}
                        color={METRIC_COLORS[key]}
                      />
                    </div>
                    <span
                      className="text-xs text-[#cbd5e1]"
                      style={{ fontFamily: "Space Mono, monospace" }}
                    >
                      {((row as any)[key] * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>

              {/* Retrieval Metrics */}
              <div className="space-y-2 pt-3">
                <div
                  className="text-xs font-bold uppercase tracking-widest flex items-center gap-2"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#ff4d6d",
                    fontSize: "0.6rem",
                  }}
                >
                  Retrieval Metrics
                  {goldFound && (
                    <span
                      style={{
                        color: "#00ff9f",
                        background: "#00ff9f22",
                        padding: "2px 6px",
                        borderRadius: "3px",
                        fontSize: "0.55rem",
                      }}
                    >
                      ✓ GOLD ANSWER FOUND
                    </span>
                  )}
                </div>
                {RETRIEVAL_METRICS.map((key) => (
                  <div key={key} className="flex items-center gap-3 pl-2">
                    <div className="w-32 flex-shrink-0">
                      <div
                        className="text-xs font-bold mb-0.5"
                        style={{
                          fontFamily: "Space Mono, monospace",
                          color: METRIC_COLORS[key],
                          fontSize: "0.6rem",
                        }}
                      >
                        {METRIC_LABELS[key]}
                      </div>
                      <ScoreBar
                        score={(row as any)[key] || 0}
                        color={METRIC_COLORS[key]}
                      />
                    </div>
                    <span
                      className="text-xs text-[#cbd5e1]"
                      style={{ fontFamily: "Space Mono, monospace" }}
                    >
                      {((row as any)[key] * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>

              {/* Answers */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3">
                <div>
                  <div
                    className="text-xs font-bold uppercase tracking-widest mb-1"
                    style={{
                      fontFamily: "Space Mono, monospace",
                      color: "#4a5a8e",
                      fontSize: "0.58rem",
                    }}
                  >
                    Expected Answer
                  </div>
                  <div
                    className="text-xs p-2 bg-[#0a1228] border border-[#1e2d54]"
                    style={{
                      fontFamily: "IBM Plex Mono, monospace",
                      color: "#6b82b0",
                      lineHeight: "1.6",
                    }}
                  >
                    {row.expected_answer}
                  </div>
                </div>
                <div>
                  <div
                    className="text-xs font-bold uppercase tracking-widest mb-1"
                    style={{
                      fontFamily: "Space Mono, monospace",
                      color: "#4a5a8e",
                      fontSize: "0.58rem",
                    }}
                  >
                    LLM Generated Answer
                  </div>
                  <div
                    className="text-xs p-2 bg-[#0a1228] border border-[#1e2d54]"
                    style={{
                      fontFamily: "IBM Plex Mono, monospace",
                      color: "#cbd5e1",
                      lineHeight: "1.6",
                    }}
                  >
                    {row.generated_answer}
                  </div>
                </div>
              </div>

              {/* Retrieved context */}
              <div>
                <div
                  className="text-xs font-bold uppercase tracking-widest mb-1"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#4a5a8e",
                    fontSize: "0.58rem",
                  }}
                >
                  Retrieved Context
                </div>
                <div
                  className="text-xs p-2 bg-[#0a1228] border border-[#1e2d54] max-h-28 overflow-y-auto"
                  style={{
                    fontFamily: "IBM Plex Mono, monospace",
                    color: "#4a5a8e",
                    lineHeight: "1.5",
                  }}
                >
                  {row.retrieved_context || "(none)"}
                </div>
              </div>

              <div
                className="text-xs text-[#2a3a6e]"
                style={{
                  fontFamily: "IBM Plex Mono, monospace",
                  fontSize: "0.6rem",
                }}
              >
                latency: {row.latency_ms.toFixed(0)}ms
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function Evaluate({ onToast }: Props) {
  const [qas, setQAs] = useState<QA[]>([{ question: "", expected_answer: "" }]);
  const [datasetName, setDatasetName] = useState("eval_run_1");
  const [result, setResult] = useState<EvalSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<{
    done: number;
    total: number;
  } | null>(null);
  const [runMode, setRunMode] = useState<"all" | "individual" | null>(null);
  const [runningIndividual, setRunningIndividual] = useState<Set<number>>(
    new Set(),
  );
  const fileRef = useRef<HTMLInputElement>(null);
  const [csvFileName, setCsvFileName] = useState<string | null>(null);

  const isQuestionEvaluated = (question: string): boolean => {
    if (!result) return false;
    return result.per_question.some((r) => r.question === question);
  };

  const handleRemoveLastResult = () => {
    if (!result || result.per_question.length === 0) return;

    const updated = result.per_question.slice(0, -1);
    if (updated.length === 0) {
      setResult(null);
      onToast("info", "Last result removed — all cleared");
      return;
    }

    const succeeded = updated.filter((r) => !r.error);
    const avg = (k: keyof QuestionResult) =>
      succeeded.length
        ? succeeded.reduce((s, r) => s + ((r[k] as number) || 0), 0) /
          succeeded.length
        : 0;

    setResult({
      accuracy_llm: avg("accuracy_llm"),
      accuracy_combined: avg("accuracy_combined"),
      faithfulness: avg("faithfulness"),
      context_precision: avg("context_precision"),
      context_recall: avg("context_recall"),
      answer_relevancy: avg("answer_relevancy"),
      recall_10: avg("recall_10"),
      recall_20: avg("recall_20"),
      recall_50: avg("recall_50"),
      mrr: avg("mrr"),
      ndcg_10: avg("ndcg_10"),
      latency_avg_ms:
        succeeded.length > 0
          ? updated.reduce((s, r) => s + r.latency_ms, 0) / succeeded.length
          : 0,
      failed_questions: updated
        .filter((r) => r.error)
        .map((r) => ({ question: r.question, error: r.error! })),
      per_question: updated,
    });
    onToast("info", "Last result removed");
  };

  const addQA = () =>
    setQAs((q) => [...q, { question: "", expected_answer: "" }]);
  const removeQA = (i: number) =>
    setQAs((q) => q.filter((_, idx) => idx !== i));
  const updateQA = (i: number, field: keyof QA, val: string) =>
    setQAs((q) =>
      q.map((item, idx) => (idx === i ? { ...item, [field]: val } : item)),
    );
  const loadSamples = () => {
    setQAs(SAMPLE_QAS);
    setCsvFileName(null);
  };

  // CSV upload handler
  const handleCSVUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        const text = ev.target?.result as string;
        const parsed = parseCSV(text);
        if (parsed.length === 0) {
          onToast(
            "warning",
            "No valid rows found. Format: Question no,Eval Question,Eval Answer",
          );
          return;
        }
        setQAs(parsed);
        setCsvFileName(file.name);
        onToast("success", `Loaded ${parsed.length} Q&A pairs from CSV`);
      };
      reader.readAsText(file);
      e.target.value = "";
    },
    [onToast],
  );

  const handleEval = async () => {
    const valid = qas.filter(
      (q) => q.question.trim() && q.expected_answer.trim(),
    );
    if (valid.length === 0) {
      onToast("warning", "Add at least one Q&A pair");
      return;
    }

    // Filter out already evaluated questions
    const toEvaluate = valid.filter((q) => !isQuestionEvaluated(q.question));
    if (toEvaluate.length === 0) {
      onToast("info", "All questions already evaluated");
      return;
    }

    setRunMode("all");
    setLoading(true);
    setProgress({ done: 0, total: toEvaluate.length });

    try {
      const perQuestion: QuestionResult[] = result?.per_question || [];
      let totalLatency = 0;
      const isLLMError = (msg: string) =>
        /ollama|localhost:11434|api\.openai\.com|openai/i.test(msg);
      let paused = false;
      let pauseReason = "";

      for (let i = 0; i < toEvaluate.length; i++) {
        if (paused) break;
        setProgress({ done: i, total: toEvaluate.length });
        try {
          const res = await ragEvaluate({
            questions: [toEvaluate[i]],
            dataset_name: `${datasetName}_q${i + 1}`,
          });
          const apiResult = res.per_question?.[0];
          const qr: QuestionResult = apiResult ?? {
            question: toEvaluate[i].question,
            expected_answer: toEvaluate[i].expected_answer,
            generated_answer: "",
            retrieved_context: "",
            accuracy_llm: res.accuracy_llm ?? res.accuracy ?? 0,
            accuracy_combined: res.accuracy_combined ?? 0,
            faithfulness: res.faithfulness ?? 0,
            answer_relevancy: res.answer_relevancy ?? 0,
            context_precision: res.context_precision ?? 0,
            context_recall: res.context_recall ?? 0,
            exact_match: res.exact_match ?? 0,
            semantic_similarity: res.semantic_similarity ?? 0,
            f1: res.f1 ?? 0,
            recall_10: res.recall_10 ?? 0,
            recall_20: res.recall_20 ?? 0,
            recall_50: res.recall_50 ?? 0,
            mrr: res.mrr ?? 0,
            ndcg_10: res.ndcg_10 ?? 0,
            gold_answer_found: res.gold_answer_found ?? false,
            accuracy_rationale: "",
            faithfulness_rationale: "",
            answer_relevancy_rationale: "",
            context_precision_rationale: "",
            context_recall_rationale: "",
            latency_ms: res.latency_avg_ms ?? 0,
          };
          if (apiResult?.error && isLLMError(apiResult.error)) {
            paused = true;
            pauseReason = apiResult.error;
          }
          perQuestion.push(qr);
          totalLatency += qr.latency_ms;
        } catch (e: any) {
          perQuestion.push({
            question: toEvaluate[i].question,
            expected_answer: toEvaluate[i].expected_answer,
            generated_answer: "",
            retrieved_context: "",
            accuracy_llm: 0,
            accuracy_combined: 0,
            faithfulness: 0,
            answer_relevancy: 0,
            context_precision: 0,
            context_recall: 0,
            exact_match: 0,
            semantic_similarity: 0,
            f1: 0,
            recall_10: 0,
            recall_20: 0,
            recall_50: 0,
            mrr: 0,
            ndcg_10: 0,
            gold_answer_found: false,
            accuracy_rationale: "",
            faithfulness_rationale: "",
            answer_relevancy_rationale: "",
            context_precision_rationale: "",
            context_recall_rationale: "",
            latency_ms: 0,
            error: e?.response?.data?.error || String(e),
          });
          const errMsg = e?.response?.data?.error || String(e);
          if (isLLMError(errMsg)) {
            paused = true;
            pauseReason = errMsg;
          }
        }
      }

      setProgress({ done: paused ? perQuestion.length : toEvaluate.length, total: toEvaluate.length });

      const succeeded = perQuestion.filter((r) => !r.error);
      const avg = (k: keyof QuestionResult) =>
        succeeded.length
          ? succeeded.reduce((s, r) => s + ((r[k] as number) || 0), 0) /
            succeeded.length
          : 0;

      const summary: EvalSummary = {
        accuracy_llm: avg("accuracy_llm"),
        accuracy_combined: avg("accuracy_combined"),
        faithfulness: avg("faithfulness"),
        context_precision: avg("context_precision"),
        context_recall: avg("context_recall"),
        answer_relevancy: avg("answer_relevancy"),
        recall_10: avg("recall_10"),
        recall_20: avg("recall_20"),
        recall_50: avg("recall_50"),
        mrr: avg("mrr"),
        ndcg_10: avg("ndcg_10"),
        latency_avg_ms: succeeded.length ? totalLatency / succeeded.length : 0,
        failed_questions: perQuestion
          .filter((r) => r.error)
          .map((r) => ({ question: r.question, error: r.error! })),
        per_question: perQuestion,
      };

      setResult(summary);
      if (paused) {
        onToast(
          "warning",
          `Evaluation paused due to LLM error: ${pauseReason}`,
        );
      } else {
        onToast(
          "success",
          `Evaluation complete — ${succeeded.length}/${valid.length} succeeded`,
        );
      }
    } catch (e: any) {
      onToast("error", e?.response?.data?.error || "Evaluation failed");
    } finally {
      setLoading(false);
      setProgress(null);
    }
  };

  const handleRunIndividual = async (index: number) => {
    const qa = qas[index];
    if (!qa.question.trim() || !qa.expected_answer.trim()) {
      onToast("warning", "Question and expected answer required");
      return;
    }

    setRunMode("individual");
    setRunningIndividual((s) => new Set(s).add(index));

    try {
      const res = await ragEvaluate({
        questions: [qa],
        dataset_name: `${datasetName}_individual_q${index + 1}`,
      });

      const qr: QuestionResult = res.per_question?.[0] ?? {
        question: qa.question,
        expected_answer: qa.expected_answer,
        generated_answer: "",
        retrieved_context: "",
        accuracy_llm: res.accuracy_llm ?? res.accuracy ?? 0,
        accuracy_combined: res.accuracy_combined ?? 0,
        faithfulness: res.faithfulness ?? 0,
        answer_relevancy: res.answer_relevancy ?? 0,
        context_precision: res.context_precision ?? 0,
        context_recall: res.context_recall ?? 0,
        exact_match: res.exact_match ?? 0,
        semantic_similarity: res.semantic_similarity ?? 0,
        f1: res.f1 ?? 0,
        recall_10: res.recall_10 ?? 0,
        recall_20: res.recall_20 ?? 0,
        recall_50: res.recall_50 ?? 0,
        mrr: res.mrr ?? 0,
        ndcg_10: res.ndcg_10 ?? 0,
        gold_answer_found: res.gold_answer_found ?? false,
        accuracy_rationale: "",
        faithfulness_rationale: "",
        answer_relevancy_rationale: "",
        context_precision_rationale: "",
        context_recall_rationale: "",
        latency_ms: res.latency_avg_ms ?? 0,
      };

      // Add or update result
      setResult((prev) => {
        if (!prev) {
          // First individual result
          const summary: EvalSummary = {
            accuracy_llm: qr.accuracy_llm ?? 0,
            accuracy_combined: qr.accuracy_combined ?? 0,
            faithfulness: qr.faithfulness,
            context_precision: qr.context_precision,
            context_recall: qr.context_recall,
            answer_relevancy: qr.answer_relevancy,
            recall_10: qr.recall_10 ?? 0,
            recall_20: qr.recall_20 ?? 0,
            recall_50: qr.recall_50 ?? 0,
            mrr: qr.mrr ?? 0,
            ndcg_10: qr.ndcg_10 ?? 0,
            latency_avg_ms: qr.latency_ms,
            failed_questions: qr.error
              ? [{ question: qr.question, error: qr.error }]
              : [],
            per_question: [qr],
          };
          return summary;
        }

        // Update existing result
        const existing = prev.per_question.findIndex(
          (r) => r.question === qa.question,
        );

        let updated = [...prev.per_question];
        if (existing >= 0) {
          updated[existing] = qr;
        } else {
          updated.push(qr);
        }

        const succeeded = updated.filter((r) => !r.error);
        const avg = (k: keyof QuestionResult) =>
          succeeded.length
            ? succeeded.reduce((s, r) => s + ((r[k] as number) || 0), 0) /
              succeeded.length
            : 0;

        return {
          accuracy_llm: avg("accuracy_llm"),
          accuracy_combined: avg("accuracy_combined"),
          faithfulness: avg("faithfulness"),
          context_precision: avg("context_precision"),
          context_recall: avg("context_recall"),
          answer_relevancy: avg("answer_relevancy"),
          recall_10: avg("recall_10"),
          recall_20: avg("recall_20"),
          recall_50: avg("recall_50"),
          mrr: avg("mrr"),
          ndcg_10: avg("ndcg_10"),
          latency_avg_ms:
            succeeded.length > 0
              ? updated.reduce((s, r) => s + r.latency_ms, 0) / succeeded.length
              : 0,
          failed_questions: updated
            .filter((r) => r.error)
            .map((r) => ({ question: r.question, error: r.error! })),
          per_question: updated,
        };
      });

      onToast("success", `Q&A #${index + 1} evaluated`);
    } catch (e: any) {
      onToast("error", e?.response?.data?.error || "Evaluation failed");
    } finally {
      setRunningIndividual((s) => {
        const ns = new Set(s);
        ns.delete(index);
        return ns;
      });
    }
  };

  const overallAvg = result
    ? (result.accuracy_llm +
        result.faithfulness +
        result.context_precision +
        result.context_recall +
        result.answer_relevancy) /
      5
    : 0;

  const gradeInfo =
    overallAvg >= 0.8
      ? { label: "EXCELLENT", color: "#00ff9f" }
      : overallAvg >= 0.6
        ? { label: "GOOD", color: "#ffe600" }
        : overallAvg >= 0.4
          ? { label: "FAIR", color: "#ff8c00" }
          : { label: "POOR", color: "#ff4d6d" };

  return (
    <div className="p-4 md:p-6 max-w-6xl mx-auto space-y-5">
      {/* Header */}
      <div>
        <h1
          className="text-xl font-bold"
          style={{ fontFamily: "Space Mono, monospace", color: "#e2e8f0" }}
        >
          Evaluate<span style={{ color: "#a78bfa" }}>//</span>RAG
        </h1>
        <p
          className="text-xs text-[#4a5a8e] mt-0.5"
          style={{ fontFamily: "IBM Plex Mono, monospace" }}
        >
          LLM-as-Judge · Accuracy Methods · Retrieval Metrics
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {/* ── Left: Input ─────────────────────────────────────────────────── */}
        <div className="space-y-4">
          {/* Dataset config */}
          <div className="brutal-card p-4 space-y-3">
            <div
              className="text-xs font-bold uppercase tracking-widest"
              style={{ fontFamily: "Space Mono, monospace", color: "#a78bfa" }}
            >
              Dataset Configuration
            </div>
            <div>
              <label
                className="text-xs text-[#4a5a8e] block mb-1"
                style={{
                  fontFamily: "Space Mono, monospace",
                  fontSize: "0.6rem",
                }}
              >
                DATASET_NAME
              </label>
              <input
                className="input-brutal w-full py-2 px-3 text-xs"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                style={{ fontFamily: "IBM Plex Mono, monospace" }}
              />
            </div>

            {/* Action buttons */}
            <div className="flex flex-wrap gap-2">
              <button
                className="btn-brutal px-3 py-1.5 text-xs flex items-center gap-1"
                onClick={addQA}
              >
                <Plus size={10} /> Add Q&A
              </button>
              <button
                className="btn-brutal px-3 py-1.5 text-xs btn-brutal-yellow"
                onClick={loadSamples}
              >
                Load Samples
              </button>
              <button
                className="btn-brutal px-3 py-1.5 text-xs flex items-center gap-1"
                style={{ borderColor: "#00ff9f", color: "#00ff9f" }}
                onClick={() => fileRef.current?.click()}
              >
                <Upload size={10} /> Upload CSV
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={handleCSVUpload}
              />
            </div>

            {/* CSV hint */}
            <div
              className="text-xs text-[#2a3a6e]"
              style={{
                fontFamily: "IBM Plex Mono, monospace",
                fontSize: "0.6rem",
                lineHeight: "1.6",
              }}
            >
              CSV format:{" "}
              <span className="text-[#4a5a8e]">
                Question no,Eval Question,Eval Answer
              </span>
              {csvFileName && (
                <span className="ml-2 flex items-center gap-1 inline-flex">
                  <FileText size={9} color="#00ff9f" />
                  <span className="text-[#00ff9f]">{csvFileName}</span>
                  <button
                    onClick={() => {
                      setCsvFileName(null);
                      setQAs([{ question: "", expected_answer: "" }]);
                    }}
                  >
                    <X size={9} color="#ff4d6d" />
                  </button>
                </span>
              )}
            </div>
          </div>

          {/* Q&A pairs */}
          <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
            {qas.map((qa, i) => (
              <div
                key={i}
                className="brutal-card p-4 space-y-2"
                style={{ boxShadow: "3px 3px 0 #a78bfa44" }}
              >
                <div className="flex items-center justify-between">
                  <span
                    className="text-xs font-bold uppercase tracking-widest"
                    style={{
                      fontFamily: "Space Mono, monospace",
                      color: "#a78bfa",
                      fontSize: "0.6rem",
                    }}
                  >
                    Q&A Pair #{i + 1}
                  </span>
                  <div className="flex items-center gap-2">
                    {isQuestionEvaluated(qa.question) && (
                      <Check size={12} color="#00ff9f" />
                    )}
                    <div className="flex items-center gap-1">
                      {qa.question.trim() && qa.expected_answer.trim() && (
                        <button
                          onClick={() => handleRunIndividual(i)}
                          disabled={runningIndividual.has(i)}
                          className="text-xs px-2 py-1 border border-[#00ff9f] text-[#00ff9f] hover:bg-[#00ff9f] hover:text-[#0a0e1b] transition-colors"
                          style={{
                            opacity: runningIndividual.has(i) ? 0.5 : 1,
                            cursor: runningIndividual.has(i)
                              ? "not-allowed"
                              : "pointer",
                          }}
                        >
                          {runningIndividual.has(i) ? "Running..." : "Run"}
                        </button>
                      )}
                      {qas.length > 1 && (
                        <button
                          onClick={() => removeQA(i)}
                          className="text-[#ff4d6d] hover:opacity-70"
                        >
                          <Trash2 size={11} />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                <div>
                  <label
                    className="text-xs text-[#4a5a8e] block mb-1"
                    style={{
                      fontFamily: "Space Mono, monospace",
                      fontSize: "0.6rem",
                    }}
                  >
                    QUESTION
                  </label>
                  <textarea
                    className="input-brutal w-full py-2 px-3 text-xs resize-none"
                    rows={2}
                    placeholder="What question should the RAG system answer?"
                    value={qa.question}
                    onChange={(e) => updateQA(i, "question", e.target.value)}
                    style={{ fontFamily: "IBM Plex Sans, sans-serif" }}
                  />
                </div>
                <div>
                  <label
                    className="text-xs text-[#4a5a8e] block mb-1"
                    style={{
                      fontFamily: "Space Mono, monospace",
                      fontSize: "0.6rem",
                    }}
                  >
                    EXPECTED ANSWER
                  </label>
                  <textarea
                    className="input-brutal w-full py-2 px-3 text-xs resize-none"
                    rows={2}
                    placeholder="What is the correct / expected answer?"
                    value={qa.expected_answer}
                    onChange={(e) =>
                      updateQA(i, "expected_answer", e.target.value)
                    }
                    style={{ fontFamily: "IBM Plex Sans, sans-serif" }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Progress bar */}
          {loading && progress && (
            <div className="brutal-card p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span
                  className="text-xs font-bold"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#a78bfa",
                    fontSize: "0.65rem",
                  }}
                >
                  EVALUATING...
                </span>
                <span
                  className="text-xs"
                  style={{
                    fontFamily: "IBM Plex Mono, monospace",
                    color: "#6b82b0",
                    fontSize: "0.65rem",
                  }}
                >
                  {progress.done} / {progress.total}
                </span>
              </div>
              <div
                className="w-full bg-[#0f1629] border border-[#1e2d54]"
                style={{ height: "6px" }}
              >
                <div
                  className="h-full transition-all duration-500"
                  style={{
                    width: `${(progress.done / progress.total) * 100}%`,
                    background: "linear-gradient(90deg, #a78bfa, #00d4ff)",
                  }}
                />
              </div>
              <div className="flex gap-1 flex-wrap">
                {Array.from({ length: progress.total }).map((_, i) => (
                  <div
                    key={i}
                    className="w-2 h-2 rounded-full transition-colors duration-300"
                    style={{
                      background:
                        i < progress.done
                          ? "#00ff9f"
                          : i === progress.done
                            ? "#a78bfa"
                            : "#1e2d54",
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-2">
            <button
              className="btn-brutal flex-1 py-3 text-sm flex items-center justify-center gap-2"
              style={{
                borderColor: "#a78bfa",
                color: "#a78bfa",
                boxShadow: "4px 4px 0 #a78bfa",
              }}
              onClick={handleEval}
              disabled={loading}
            >
              {loading && runMode === "all" ? (
                <Spinner size={16} color="#a78bfa" />
              ) : (
                <FlaskConical size={16} />
              )}
              {loading && runMode === "all" ? "RUNNING ALL..." : "RUN ALL"}
            </button>
          </div>

          {/* Progress bar */}
          {loading && progress && (
            <div className="brutal-card p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span
                  className="text-xs font-bold"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#a78bfa",
                    fontSize: "0.65rem",
                  }}
                >
                  EVALUATING...
                </span>
                <span
                  className="text-xs"
                  style={{
                    fontFamily: "IBM Plex Mono, monospace",
                    color: "#6b82b0",
                    fontSize: "0.65rem",
                  }}
                >
                  {progress.done} / {progress.total}
                </span>
              </div>
              <div
                className="w-full bg-[#0f1629] border border-[#1e2d54]"
                style={{ height: "6px" }}
              >
                <div
                  className="h-full transition-all duration-500"
                  style={{
                    width: `${(progress.done / progress.total) * 100}%`,
                    background: "linear-gradient(90deg, #a78bfa, #00d4ff)",
                  }}
                />
              </div>
              <div className="flex gap-1 flex-wrap">
                {Array.from({ length: progress.total }).map((_, i) => (
                  <div
                    key={i}
                    className="w-2 h-2 rounded-full transition-colors duration-300"
                    style={{
                      background:
                        i < progress.done
                          ? "#00ff9f"
                          : i === progress.done
                            ? "#a78bfa"
                            : "#1e2d54",
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Right: Results ───────────────────────────────────────────────── */}
        <div className="space-y-4">
          {loading && !result && (
            <div className="brutal-card p-10 flex flex-col items-center gap-4">
              <Spinner size={32} color="#a78bfa" label="LLM_JUDGE_SCORING..." />
              <p
                className="text-xs text-center text-[#4a5a8e]"
                style={{ fontFamily: "IBM Plex Mono, monospace" }}
              >
                LLM is scoring each metric — this may take a moment
              </p>
            </div>
          )}

          {result && (
            <>
              {/* Overall grade */}
              <div
                className="brutal-card p-4 text-center"
                style={{ boxShadow: `6px 6px 0 ${gradeInfo.color}` }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div
                    className="text-xs text-[#4a5a8e]"
                    style={{
                      fontFamily: "Space Mono, monospace",
                      fontSize: "0.6rem",
                    }}
                  >
                    OVERALL GRADE · LLM-AS-JUDGE
                  </div>
                  {result.per_question.length > 0 && (
                    <button
                      onClick={handleRemoveLastResult}
                      className="text-xs px-2 py-0.5 border border-[#ff4d6d] text-[#ff4d6d] hover:bg-[#ff4d6d] hover:text-[#0a0e1b] transition-colors"
                      style={{ fontSize: "0.65rem" }}
                    >
                      Undo Last
                    </button>
                  )}
                </div>
                <div
                  className="text-4xl font-bold"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: gradeInfo.color,
                  }}
                >
                  {(overallAvg * 100).toFixed(1)}%
                </div>
                <div
                  className="text-xs font-bold mt-1"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: gradeInfo.color,
                  }}
                >
                  {gradeInfo.label}
                </div>
                <div
                  className="text-xs text-[#4a5a8e] mt-1"
                  style={{
                    fontFamily: "IBM Plex Mono, monospace",
                    fontSize: "0.65rem",
                  }}
                >
                  avg latency: {result.latency_avg_ms.toFixed(0)}ms ·{" "}
                  {result.per_question.length - result.failed_questions.length}/
                  {result.per_question.length} succeeded
                </div>
                {/* Export */}
                <button
                  className="btn-brutal mt-3 px-4 py-1.5 text-xs flex items-center gap-1.5 mx-auto"
                  style={{ borderColor: "#00ff9f", color: "#00ff9f" }}
                  onClick={() => exportToCSV(result.per_question)}
                >
                  <Download size={11} /> Export CSV
                </button>
              </div>

              {/* Metric bars */}
              <div className="brutal-card p-4 space-y-4">
                <div
                  className="text-xs font-bold uppercase tracking-widest"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#6b82b0",
                  }}
                >
                  LLM-as-Judge Metrics
                </div>
                {(LLM_JUDGE_METRICS as readonly string[]).map((key) => (
                  <ScoreBar
                    key={key}
                    score={(result as any)[key] || 0}
                    label={METRIC_LABELS[key as keyof typeof METRIC_LABELS]}
                    color={METRIC_COLORS[key]}
                  />
                ))}
              </div>

              <div className="brutal-card p-4 space-y-4">
                <div
                  className="text-xs font-bold uppercase tracking-widest"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#6b82b0",
                  }}
                >
                  Accuracy Evaluation Methods
                </div>
                {(ACCURACY_METRICS as readonly string[]).map((key) => (
                  <ScoreBar
                    key={key}
                    score={(result as any)[key] || 0}
                    label={METRIC_LABELS[key as keyof typeof METRIC_LABELS]}
                    color={METRIC_COLORS[key]}
                  />
                ))}
              </div>

              <div className="brutal-card p-4 space-y-4">
                <div
                  className="text-xs font-bold uppercase tracking-widest"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#6b82b0",
                  }}
                >
                  Retrieval Metrics
                </div>
                {(RETRIEVAL_METRICS as readonly string[]).map((key) => (
                  <ScoreBar
                    key={key}
                    score={(result as any)[key] || 0}
                    label={METRIC_LABELS[key as keyof typeof METRIC_LABELS]}
                    color={METRIC_COLORS[key]}
                  />
                ))}
              </div>

              {/* Per-question results */}
              <div className="brutal-card p-4">
                <div
                  className="text-xs font-bold uppercase tracking-widest mb-3 flex items-center justify-between"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    color: "#6b82b0",
                  }}
                >
                  <span>
                    Per-Question Results [{result.per_question.length}]
                  </span>
                  <span
                    className="text-[#2a3a6e]"
                    style={{ fontSize: "0.6rem" }}
                  >
                    click to expand · or run individually
                  </span>
                </div>
                <div>
                  {[...result.per_question].reverse().map((row, i) => {
                    const originalIdx = result.per_question.length - 1 - i;
                    return (
                      <QuestionRow
                        key={originalIdx}
                        row={row}
                        idx={originalIdx}
                        onRunIndividual={() => handleRunIndividual(originalIdx)}
                        isRunning={runningIndividual.has(originalIdx)}
                      />
                    );
                  })}
                </div>
              </div>

              {/* Failed questions */}
              {result.failed_questions.length > 0 && (
                <div
                  className="brutal-card p-4"
                  style={{ boxShadow: "4px 4px 0 #ff4d6d" }}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle size={12} color="#ff4d6d" />
                    <span
                      className="text-xs font-bold uppercase tracking-widest"
                      style={{
                        fontFamily: "Space Mono, monospace",
                        color: "#ff4d6d",
                      }}
                    >
                      Failed [{result.failed_questions.length}]
                    </span>
                  </div>
                  {result.failed_questions.map((f, i) => (
                    <div
                      key={i}
                      className="p-2 mb-2"
                      style={{
                        background: "#ff4d6d0d",
                        border: "1px solid #ff4d6d33",
                      }}
                    >
                      <div
                        className="text-xs text-[#e2e8f0]"
                        style={{
                          fontFamily: "IBM Plex Mono, monospace",
                          fontSize: "0.7rem",
                        }}
                      >
                        {f.question}
                      </div>
                      <div
                        className="text-xs text-[#ff4d6d] mt-1"
                        style={{
                          fontFamily: "IBM Plex Mono, monospace",
                          fontSize: "0.65rem",
                        }}
                      >
                        Error: {f.error}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {!result && !loading && (
            <div className="brutal-card p-8 flex flex-col items-center gap-3 text-center">
              <FlaskConical size={28} color="#2a3a6e" />
              <span
                className="text-xs text-[#4a5a8e]"
                style={{ fontFamily: "Space Mono, monospace" }}
              >
                Configure Q&A pairs and run evaluation
              </span>
              <div
                className="text-xs text-[#2a3a6e] space-y-1"
                style={{ fontFamily: "IBM Plex Mono, monospace" }}
              >
                <div>Metrics scored by LLM — accuracy, faithfulness,</div>
                <div>precision, recall, and answer relevancy.</div>
                <div className="mt-2 text-[#1e2d54]">
                  Upload CSV · Export results · Expand per-question detail
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
