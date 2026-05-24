import React, { useState } from 'react';
import { FlaskConical, Plus, Trash2, Zap, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { ragEvaluate } from '../api/client';
import Spinner from '../components/Spinner';
import ScoreBar from '../components/ScoreBar';
import type { EvaluationResponse } from '../types';
import { formatLatency } from '../utils/format';

interface Props { onToast: (type: any, msg: string) => void; }

interface QA { question: string; expected_answer: string; }

const SAMPLE_QAS: QA[] = [
  { question: 'What is the main topic of the indexed documents?', expected_answer: 'The documents cover government schemes and eligibility criteria.' },
  { question: 'Who is eligible for the scheme?', expected_answer: 'Small and marginal farmers with valid land records.' },
];

const METRIC_COLORS: Record<string, string> = {
  accuracy:          '#00d4ff',
  faithfulness:      '#00ff9f',
  context_precision: '#ffe600',
  context_recall:    '#ff4d6d',
  answer_relevancy:  '#a78bfa',
};

const METRIC_LABELS: Record<string, string> = {
  accuracy:          'Accuracy',
  faithfulness:      'Faithfulness',
  context_precision: 'Context Precision',
  context_recall:    'Context Recall',
  answer_relevancy:  'Answer Relevancy',
};

export default function Evaluate({ onToast }: Props) {
  const [qas, setQAs] = useState<QA[]>([{ question: '', expected_answer: '' }]);
  const [datasetName, setDatasetName] = useState('eval_run_1');
  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const addQA = () => setQAs(q => [...q, { question: '', expected_answer: '' }]);
  const removeQA = (i: number) => setQAs(q => q.filter((_, idx) => idx !== i));
  const updateQA = (i: number, field: keyof QA, val: string) =>
    setQAs(q => q.map((item, idx) => idx === i ? { ...item, [field]: val } : item));

  const loadSamples = () => setQAs(SAMPLE_QAS);

  const handleEval = async () => {
    const valid = qas.filter(q => q.question.trim() && q.expected_answer.trim());
    if (valid.length === 0) { onToast('warning', 'Add at least one Q&A pair'); return; }
    setLoading(true);
    setResult(null);
    try {
      const res = await ragEvaluate({ questions: valid, dataset_name: datasetName });
      setResult(res);
      onToast('success', `Evaluation complete: ${(res.accuracy * 100).toFixed(1)}% accuracy`);
    } catch (e: any) {
      onToast('error', e?.response?.data?.error || 'Evaluation failed');
    } finally { setLoading(false); }
  };

  const getGrade = (score: number) => {
    if (score >= 0.8) return { label: 'EXCELLENT', color: '#00ff9f' };
    if (score >= 0.6) return { label: 'GOOD', color: '#ffe600' };
    if (score >= 0.4) return { label: 'FAIR', color: '#ff8c00' };
    return { label: 'POOR', color: '#ff4d6d' };
  };

  return (
    <div className="p-4 md:p-6 max-w-6xl mx-auto space-y-5">
      <div>
        <h1 className="text-xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
          Evaluate<span style={{ color: '#a78bfa' }}>//</span>RAG
        </h1>
        <p className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
          Measure accuracy, faithfulness, precision and recall across your pipeline
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {/* Left: Input */}
        <div className="space-y-4">
          {/* Dataset name + controls */}
          <div className="brutal-card p-4 space-y-3">
            <div className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#a78bfa' }}>
              Dataset Configuration
            </div>
            <div>
              <label className="text-xs text-[#4a5a8e] block mb-1" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>
                DATASET_NAME
              </label>
              <input
                className="input-brutal w-full py-2 px-3 text-xs"
                value={datasetName}
                onChange={e => setDatasetName(e.target.value)}
                style={{ fontFamily: 'IBM Plex Mono, monospace' }}
              />
            </div>
            <div className="flex gap-2">
              <button className="btn-brutal px-3 py-1.5 text-xs flex items-center gap-1" onClick={addQA}>
                <Plus size={10} /> Add Q&A
              </button>
              <button
                className="btn-brutal px-3 py-1.5 text-xs btn-brutal-yellow"
                onClick={loadSamples}
              >
                Load Samples
              </button>
            </div>
          </div>

          {/* Q&A pairs */}
          <div className="space-y-3">
            {qas.map((qa, i) => (
              <div key={i} className="brutal-card p-4 space-y-2" style={{ boxShadow: '3px 3px 0 #a78bfa44' }}>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#a78bfa', fontSize: '0.6rem' }}>
                    Q&A Pair #{i + 1}
                  </span>
                  {qas.length > 1 && (
                    <button onClick={() => removeQA(i)} className="text-[#ff4d6d] hover:opacity-70">
                      <Trash2 size={11} />
                    </button>
                  )}
                </div>
                <div>
                  <label className="text-xs text-[#4a5a8e] block mb-1" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>QUESTION</label>
                  <textarea
                    className="input-brutal w-full py-2 px-3 text-xs resize-none"
                    rows={2}
                    placeholder="What question should the RAG system answer?"
                    value={qa.question}
                    onChange={e => updateQA(i, 'question', e.target.value)}
                    style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}
                  />
                </div>
                <div>
                  <label className="text-xs text-[#4a5a8e] block mb-1" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>EXPECTED ANSWER</label>
                  <textarea
                    className="input-brutal w-full py-2 px-3 text-xs resize-none"
                    rows={2}
                    placeholder="What is the correct / expected answer?"
                    value={qa.expected_answer}
                    onChange={e => updateQA(i, 'expected_answer', e.target.value)}
                    style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}
                  />
                </div>
              </div>
            ))}
          </div>

          <button
            className="btn-brutal w-full py-3 text-sm flex items-center justify-center gap-2"
            style={{ borderColor: '#a78bfa', color: '#a78bfa', boxShadow: '4px 4px 0 #a78bfa' }}
            onClick={handleEval}
            disabled={loading}
          >
            {loading ? <Spinner size={16} color="#a78bfa" /> : <FlaskConical size={16} />}
            {loading ? 'EVALUATING...' : 'RUN EVALUATION'}
          </button>
        </div>

        {/* Right: Results */}
        <div className="space-y-4">
          {loading && (
            <div className="brutal-card p-12 flex flex-col items-center gap-4">
              <Spinner size={32} color="#a78bfa" label="RUNNING_EVALUATION..." />
              <p className="text-xs text-center text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                Running retrieval and scoring each question...
              </p>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Overall grade */}
              {(() => {
                const avg = (result.accuracy + result.faithfulness + result.context_precision + result.context_recall + result.answer_relevancy) / 5;
                const grade = getGrade(avg);
                return (
                  <div className="brutal-card p-4 text-center" style={{ boxShadow: `6px 6px 0 ${grade.color}` }}>
                    <div className="text-xs text-[#4a5a8e] mb-1" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>
                      OVERALL GRADE
                    </div>
                    <div className="text-4xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: grade.color }}>
                      {(avg * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs font-bold mt-1" style={{ fontFamily: 'Space Mono, monospace', color: grade.color }}>
                      {grade.label}
                    </div>
                    <div className="text-xs text-[#4a5a8e] mt-1" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                      avg latency: {formatLatency(result.latency_avg_ms)}
                    </div>
                  </div>
                );
              })()}

              {/* Metric bars */}
              <div className="brutal-card p-4 space-y-4">
                <div className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
                  Metric Breakdown
                </div>
                {Object.entries(METRIC_LABELS).map(([key, label]) => (
                  <ScoreBar
                    key={key}
                    score={(result as any)[key] || 0}
                    label={label}
                    color={METRIC_COLORS[key]}
                  />
                ))}
              </div>

              {/* Metric explanations */}
              <div className="brutal-card p-4 space-y-2">
                <div className="text-xs font-bold uppercase tracking-widest mb-2" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
                  Score Interpretation
                </div>
                {[
                  { key: 'accuracy', note: 'Semantic similarity between generated and expected answers' },
                  { key: 'faithfulness', note: 'How grounded the answer is in retrieved context' },
                  { key: 'context_precision', note: 'Fraction of retrieved chunks relevant to the query' },
                  { key: 'context_recall', note: 'Whether the needed information was retrieved' },
                  { key: 'answer_relevancy', note: 'How directly the answer addresses the question' },
                ].map(({ key, note }) => (
                  <div key={key} className="flex items-start gap-2 py-1 border-b border-[#1a2444]">
                    <span className="w-2 h-2 mt-1 flex-shrink-0 rounded-full" style={{ background: METRIC_COLORS[key] }} />
                    <div>
                      <span className="text-xs font-bold" style={{ fontFamily: 'Space Mono, monospace', color: METRIC_COLORS[key], fontSize: '0.65rem' }}>
                        {METRIC_LABELS[key]}:{' '}
                      </span>
                      <span className="text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                        {note}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Failed questions */}
              {result.failed_questions.length > 0 && (
                <div className="brutal-card p-4" style={{ boxShadow: '4px 4px 0 #ff4d6d' }}>
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle size={12} color="#ff4d6d" />
                    <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#ff4d6d' }}>
                      Failed [{result.failed_questions.length}]
                    </span>
                  </div>
                  {result.failed_questions.map((f, i) => (
                    <div key={i} className="p-2 mb-2" style={{ background: '#ff4d6d0d', border: '1px solid #ff4d6d33' }}>
                      <div className="text-xs text-[#e2e8f0]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.7rem' }}>
                        {f.question}
                      </div>
                      <div className="text-xs text-[#ff4d6d] mt-1" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
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
              <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'Space Mono, monospace' }}>
                Configure Q&A pairs and run evaluation
              </span>
              <div className="text-xs text-[#2a3a6e] space-y-1" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                <div>Metrics computed via embedding similarity</div>
                <div>No ground-truth labels required beyond expected answers</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
