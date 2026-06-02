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
  accuracy: number;
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  accuracy_rationale: string;
  faithfulness_rationale: string;
  answer_relevancy_rationale: string;
  context_precision_rationale: string;
  context_recall_rationale: string;
  latency_ms: number;
  error?: string;
}

interface EvalSummary {
  accuracy: number;
  faithfulness: number;
  context_precision: number;
  context_recall: number;
  answer_relevancy: number;
  latency_avg_ms: number;
  failed_questions: { question: string; error: string }[];
  per_question: QuestionResult[];
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
  accuracy: "#00d4ff",
  faithfulness: "#00ff9f",
  context_precision: "#ffe600",
  context_recall: "#ff4d6d",
  answer_relevancy: "#a78bfa",
};

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  faithfulness: "Faithfulness",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
  answer_relevancy: "Answer Relevancy",
};

const METRIC_KEYS = Object.keys(
  METRIC_LABELS,
) as (keyof typeof METRIC_LABELS)[];

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
  const headers = [
    "Question No",
    "Question",
    "Expected Answer",
    "Generated Answer",
    "Retrieved Context",
    "Accuracy",
    "Faithfulness",
    "Context Precision",
    "Context Recall",
    "Answer Relevancy",
    "Accuracy Rationale",
    "Faithfulness Rationale",
    "Context Precision Rationale",
    "Context Recall Rationale",
    "Answer Relevancy Rationale",
    "Latency (ms)",
  ];
  const escape = (v: any) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csvRows = rows.map((r, i) =>
    [
      i + 1,
      escape(r.question),
      escape(r.expected_answer),
      escape(r.generated_answer),
      escape(r.retrieved_context),
      r.accuracy,
      r.faithfulness,
      r.context_precision,
      r.context_recall,
      r.answer_relevancy,
      escape(r.accuracy_rationale),
      escape(r.faithfulness_rationale),
      escape(r.context_precision_rationale),
      escape(r.context_recall_rationale),
      escape(r.answer_relevancy_rationale),
      r.latency_ms,
    ].join(","),
  );
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
  const { label, color } =
    score >= 0.8
      ? { label: "EXCELLENT", color: "#00ff9f" }
      : score >= 0.6
        ? { label: "GOOD", color: "#ffe600" }
        : score >= 0.4
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

function QuestionRow({ row, idx }: { row: QuestionResult; idx: number }) {
  const [open, setOpen] = useState(false);
  const avg =
    (row.accuracy +
      row.faithfulness +
      row.context_precision +
      row.context_recall +
      row.answer_relevancy) /
    5;

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

              {/* Metric scores + rationales */}
              <div className="space-y-2">
                {METRIC_KEYS.map((key) => (
                  <div key={key} className="flex items-start gap-3">
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
                      className="text-xs text-[#4a5a8e] pt-0.5"
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
  const fileRef = useRef<HTMLInputElement>(null);
  const [csvFileName, setCsvFileName] = useState<string | null>(null);

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

    setLoading(true);
    setResult(null);
    setProgress({ done: 0, total: valid.length });

    try {
      // The backend should ideally support SSE/websocket for progress; here we
      // simulate progress by calling the evaluate endpoint which accepts a
      // progress_callback concept via polling — or we rely on a batch endpoint
      // that returns per_question results. We pass the full batch and poll.
      //
      // Since the existing API is a single POST, we do sequential individual calls
      // so the UI can show per-question progress.
      const perQuestion: QuestionResult[] = [];
      let totalLatency = 0;

      for (let i = 0; i < valid.length; i++) {
        setProgress({ done: i, total: valid.length });
        try {
          const res = await ragEvaluate({
            questions: [valid[i]],
            dataset_name: `${datasetName}_q${i + 1}`,
          });
          // res is an EvaluationResponse with per_question array (index 0)
          const qr: QuestionResult = res.per_question?.[0] ?? {
            question: valid[i].question,
            expected_answer: valid[i].expected_answer,
            generated_answer: "",
            retrieved_context: "",
            accuracy: res.accuracy ?? 0,
            faithfulness: res.faithfulness ?? 0,
            answer_relevancy: res.answer_relevancy ?? 0,
            context_precision: res.context_precision ?? 0,
            context_recall: res.context_recall ?? 0,
            accuracy_rationale: "",
            faithfulness_rationale: "",
            answer_relevancy_rationale: "",
            context_precision_rationale: "",
            context_recall_rationale: "",
            latency_ms: res.latency_avg_ms ?? 0,
          };
          perQuestion.push(qr);
          totalLatency += qr.latency_ms;
        } catch (e: any) {
          perQuestion.push({
            question: valid[i].question,
            expected_answer: valid[i].expected_answer,
            generated_answer: "",
            retrieved_context: "",
            accuracy: 0,
            faithfulness: 0,
            answer_relevancy: 0,
            context_precision: 0,
            context_recall: 0,
            accuracy_rationale: "",
            faithfulness_rationale: "",
            answer_relevancy_rationale: "",
            context_precision_rationale: "",
            context_recall_rationale: "",
            latency_ms: 0,
            error: e?.response?.data?.error || String(e),
          });
        }
      }

      setProgress({ done: valid.length, total: valid.length });

      const succeeded = perQuestion.filter((r) => !r.error);
      const avg = (k: keyof QuestionResult) =>
        succeeded.length
          ? succeeded.reduce((s, r) => s + ((r[k] as number) || 0), 0) /
            succeeded.length
          : 0;

      const summary: EvalSummary = {
        accuracy: avg("accuracy"),
        faithfulness: avg("faithfulness"),
        context_precision: avg("context_precision"),
        context_recall: avg("context_recall"),
        answer_relevancy: avg("answer_relevancy"),
        latency_avg_ms: succeeded.length ? totalLatency / succeeded.length : 0,
        failed_questions: perQuestion
          .filter((r) => r.error)
          .map((r) => ({ question: r.question, error: r.error! })),
        per_question: perQuestion,
      };

      setResult(summary);
      onToast(
        "success",
        `Evaluation complete — ${succeeded.length}/${valid.length} succeeded`,
      );
    } catch (e: any) {
      onToast("error", e?.response?.data?.error || "Evaluation failed");
    } finally {
      setLoading(false);
    }
  };

  const overallAvg = result
    ? (result.accuracy +
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
          LLM-as-Judge: accuracy, faithfulness, precision & recall — no cosine
          heuristics
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
                  {qas.length > 1 && (
                    <button
                      onClick={() => removeQA(i)}
                      className="text-[#ff4d6d] hover:opacity-70"
                    >
                      <Trash2 size={11} />
                    </button>
                  )}
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

          <button
            className="btn-brutal w-full py-3 text-sm flex items-center justify-center gap-2"
            style={{
              borderColor: "#a78bfa",
              color: "#a78bfa",
              boxShadow: "4px 4px 0 #a78bfa",
            }}
            onClick={handleEval}
            disabled={loading}
          >
            {loading ? (
              <Spinner size={16} color="#a78bfa" />
            ) : (
              <FlaskConical size={16} />
            )}
            {loading ? "EVALUATING..." : "RUN EVALUATION"}
          </button>
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
                <div
                  className="text-xs text-[#4a5a8e] mb-1"
                  style={{
                    fontFamily: "Space Mono, monospace",
                    fontSize: "0.6rem",
                  }}
                >
                  OVERALL GRADE · LLM-AS-JUDGE
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
                  Metric Breakdown
                </div>
                {METRIC_KEYS.map((key) => (
                  <ScoreBar
                    key={key}
                    score={(result as any)[key] || 0}
                    label={METRIC_LABELS[key]}
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
                    click to expand
                  </span>
                </div>
                <div>
                  {result.per_question.map((row, i) => (
                    <QuestionRow key={i} row={row} idx={i} />
                  ))}
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
