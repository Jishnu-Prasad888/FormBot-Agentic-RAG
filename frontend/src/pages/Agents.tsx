import React, { useState } from 'react';
import { Bot, Zap, Clock, Target, ChevronDown, ChevronUp, Filter } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { runAgent } from '../api/client';
import Spinner from '../components/Spinner';
import SourceChips from '../components/SourceChip';
import ScoreBar from '../components/ScoreBar';
import type { AgentType, AgentResponse } from '../types';
import { formatLatency, truncate } from '../utils/format';

interface Props { onToast: (type: any, msg: string) => void; }

const AGENTS: { id: AgentType; label: string; desc: string; color: string; capabilities: string[] }[] = [
  {
    id: 'coordinator',
    label: 'Coordinator',
    desc: 'Orchestrates all agents and synthesises a final answer',
    color: '#00d4ff',
    capabilities: ['Task decomposition', 'Agent selection', 'Response synthesis', 'Intent classification'],
  },
  {
    id: 'vector',
    label: 'Vector Retrieval',
    desc: 'Dense + sparse hybrid search over all collections',
    color: '#38bdf8',
    capabilities: ['Embedding search', 'BM25 keyword search', 'RRF fusion', 'Metadata filtering'],
  },
  {
    id: 'sqlite',
    label: 'SQLite / Table',
    desc: 'Structured queries over CSV and tabular data',
    color: '#00ff9f',
    capabilities: ['TableRAG execution', 'Schema search', 'Row-level retrieval', 'Aggregation queries'],
  },
  {
    id: 'router',
    label: 'Document Router',
    desc: 'Auto-routes queries to the best RAG pipeline by doc type',
    color: '#ffe600',
    capabilities: ['PDF routing', 'Markdown routing', 'CSV routing', 'Type detection'],
  },
  {
    id: 'web',
    label: 'Web Enrichment',
    desc: 'Fetches and indexes web content on demand',
    color: '#ff4d6d',
    capabilities: ['URL ingestion', 'HTML extraction', 'Real-time indexing', 'Citation generation'],
  },
  {
    id: 'evaluator',
    label: 'Retrieval Evaluator',
    desc: 'Scores retrieval quality: faithfulness, precision, recall',
    color: '#a78bfa',
    capabilities: ['Faithfulness scoring', 'Context precision', 'Context recall', 'Answer relevancy'],
  },
];

export default function Agents({ onToast }: Props) {
  const [selectedAgent, setSelectedAgent] = useState<AgentType>('coordinator');
  const [query, setQuery] = useState('');
  const [context, setContext] = useState('');
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<AgentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [showMeta, setShowMeta] = useState(false);

  const agent = AGENTS.find(a => a.id === selectedAgent)!;

  const handleRun = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      let ctx: any = {};
      if (context.trim()) { try { ctx = JSON.parse(context); } catch { onToast('warning', 'Invalid context JSON'); } }
      const res = await runAgent(selectedAgent, { query: query.trim(), context: ctx, top_k: topK });
      setResult(res);
    } catch (e: any) {
      onToast('error', e?.response?.data?.error || 'Agent failed');
    } finally { setLoading(false); }
  };

  return (
    <div className="p-4 md:p-6 max-w-7xl mx-auto space-y-5">
      <div>
        <h1 className="text-xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
          Agent<span style={{ color: '#00d4ff' }}>//</span>Framework
        </h1>
        <p className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
          Six specialised retrieval agents with plan/execute/evaluate lifecycle
        </p>
      </div>

      {/* Agent cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-2">
        {AGENTS.map(a => (
          <button
            key={a.id}
            onClick={() => setSelectedAgent(a.id)}
            className="p-3 text-left transition-all"
            style={{
              border: `2px solid ${selectedAgent === a.id ? a.color : '#2a3a6e'}`,
              background: selectedAgent === a.id ? a.color + '12' : '#0f1629',
              boxShadow: selectedAgent === a.id ? `3px 3px 0 ${a.color}` : 'none',
              transform: selectedAgent === a.id ? 'translate(-2px,-2px)' : 'none',
            }}
          >
            <Bot size={14} color={selectedAgent === a.id ? a.color : '#4a5a8e'} />
            <div className="mt-1.5 text-xs font-bold uppercase tracking-wide" style={{ fontFamily: 'Space Mono, monospace', color: selectedAgent === a.id ? a.color : '#6b82b0', fontSize: '0.6rem' }}>
              {a.label}
            </div>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* Agent info + controls */}
        <div className="space-y-4">
          {/* Info */}
          <div className="brutal-card p-4" style={{ boxShadow: `4px 4px 0 ${agent.color}` }}>
            <div className="flex items-center gap-2 mb-2">
              <Bot size={14} color={agent.color} />
              <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: agent.color }}>
                {agent.label}
              </span>
            </div>
            <p className="text-xs text-[#8a9abb] mb-3" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.7rem', lineHeight: '1.6' }}>
              {agent.desc}
            </p>
            <div className="space-y-1">
              {agent.capabilities.map(cap => (
                <div key={cap} className="flex items-center gap-2">
                  <span className="w-1 h-1 flex-shrink-0" style={{ background: agent.color }} />
                  <span className="text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                    {cap}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Config */}
          <div className="brutal-card p-4 space-y-3">
            <div className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
              Config
            </div>
            <div>
              <label className="text-xs text-[#4a5a8e] block mb-1" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>TOP_K</label>
              <input
                className="input-brutal w-full py-2 px-3 text-xs"
                type="number" min={1} max={50} value={topK}
                onChange={e => setTopK(Number(e.target.value))}
                style={{ fontFamily: 'Space Mono, monospace' }}
              />
            </div>
            <div>
              <label className="text-xs text-[#4a5a8e] block mb-1" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>
                CONTEXT (JSON)
              </label>
              <textarea
                className="input-brutal w-full py-2 px-3 text-xs resize-none"
                rows={3}
                placeholder='{"document_id":"uuid","strategy":"hybrid"}'
                value={context}
                onChange={e => setContext(e.target.value)}
                style={{ fontFamily: 'IBM Plex Mono, monospace' }}
              />
            </div>
          </div>
        </div>

        {/* Query + results */}
        <div className="xl:col-span-2 space-y-4">
          <div className="brutal-card p-4 space-y-3" style={{ boxShadow: `4px 4px 0 ${agent.color}` }}>
            <textarea
              className="input-brutal w-full py-3 px-3 text-sm resize-none"
              rows={3}
              placeholder={`Ask the ${agent.label} agent...`}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && e.ctrlKey) handleRun(); }}
              style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}
            />
            <div className="flex justify-between items-center">
              <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                Ctrl+Enter to run
              </span>
              <button
                className="btn-brutal px-4 py-2 text-xs flex items-center gap-2"
                onClick={handleRun}
                disabled={loading || !query.trim()}
                style={{ borderColor: agent.color, color: agent.color, boxShadow: `3px 3px 0 ${agent.color}` }}
              >
                {loading ? <Spinner size={12} color={agent.color} /> : <Zap size={12} />}
                Run Agent
              </button>
            </div>
          </div>

          {loading && (
            <div className="brutal-card p-8 flex justify-center">
              <Spinner size={28} color={agent.color} label={`${selectedAgent.toUpperCase()}_AGENT_RUNNING...`} />
            </div>
          )}

          {result && !loading && (
            <div className="space-y-3">
              {/* Meta */}
              <div className="flex flex-wrap items-center gap-3">
                <span className="tag-chip" style={{ borderColor: agent.color, color: agent.color }}>
                  {result.agent}
                </span>
                <div className="flex items-center gap-1.5">
                  <Clock size={10} color="#4a5a8e" />
                  <span className="text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                    {formatLatency(result.latency_ms)}
                  </span>
                </div>
                {result.confidence != null && (
                  <div className="flex items-center gap-1.5">
                    <Target size={10} color={agent.color} />
                    <span className="text-xs" style={{ fontFamily: 'IBM Plex Mono, monospace', color: agent.color, fontSize: '0.65rem' }}>
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>

              {/* Answer */}
              <div className="brutal-card p-4" style={{ boxShadow: `4px 4px 0 ${agent.color}` }}>
                <div className="text-xs font-bold uppercase tracking-widest mb-3" style={{ fontFamily: 'Space Mono, monospace', color: agent.color }}>
                  Agent Response
                </div>
                <div className="prose prose-invert prose-sm max-w-none" style={{ fontFamily: 'IBM Plex Sans, sans-serif', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.7' }}>
                  <ReactMarkdown>{result.answer}</ReactMarkdown>
                </div>
                {result.sources && result.sources.length > 0 && <SourceChips sources={result.sources as any} />}
              </div>

              {/* Metadata toggle */}
              {result.metadata && Object.keys(result.metadata).length > 0 && (
                <div className="brutal-card overflow-hidden">
                  <button
                    className="w-full flex items-center justify-between p-3"
                    style={{ background: '#1a2444' }}
                    onClick={() => setShowMeta(m => !m)}
                  >
                    <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
                      Agent Metadata
                    </span>
                    {showMeta ? <ChevronUp size={12} color="#6b82b0" /> : <ChevronDown size={12} color="#6b82b0" />}
                  </button>
                  {showMeta && (
                    <div className="p-3">
                      <pre className="text-xs text-[#6b82b0] overflow-auto" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem', maxHeight: 200 }}>
                        {JSON.stringify(result.metadata, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
