import React, { useState, useRef } from 'react';
import { Brain, Zap, Filter, StopCircle, Clock, Target } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { ragQuery, streamRagQuery, ragRetrieve } from '../api/client';
import Spinner from '../components/Spinner';
import ScoreBar from '../components/ScoreBar';
import SourceChips from '../components/SourceChip';
import type { RAGQueryResponse, RAGStrategy } from '../types';
import { formatLatency, strategyColor, truncate } from '../utils/format';

interface Props { onToast: (type: any, msg: string) => void; }

const STRATEGIES: { id: RAGStrategy; label: string; color: string }[] = [
  { id: 'hybrid',   label: 'Hybrid RAG',        color: '#00d4ff' },
  { id: 'vector',   label: 'Vector RAG',         color: '#38bdf8' },
  { id: 'bm25',     label: 'BM25',               color: '#ffe600' },
  { id: 'table',    label: 'TableRAG',           color: '#00ff9f' },
  { id: 'pdf',      label: 'PDF Hierarchical',   color: '#ff4d6d' },
  { id: 'markdown', label: 'Markdown Structure', color: '#a78bfa' },
];

export default function RAGQuery({ onToast }: Props) {
  const [query, setQuery] = useState('');
  const [strategy, setStrategy] = useState<RAGStrategy>('hybrid');
  const [topK, setTopK] = useState(5);
  const [filters, setFilters] = useState('');
  const [mode, setMode] = useState<'generate' | 'stream' | 'retrieve'>('generate');
  const [result, setResult] = useState<RAGQueryResponse | null>(null);
  const [retrievedChunks, setRetrievedChunks] = useState<any[]>([]);
  const [streamText, setStreamText] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const stopRef = useRef(false);

  const activeColor = strategyColor(strategy);

  const handleRun = async () => {
    if (!query.trim()) return;
    let parsedFilters: any;
    try { if (filters.trim()) parsedFilters = JSON.parse(filters); } catch { onToast('warning', 'Invalid filter JSON'); }
    const req = { query: query.trim(), strategy, top_k: topK, filters: parsedFilters };

    if (mode === 'retrieve') {
      setLoading(true);
      setRetrievedChunks([]);
      try {
        const chunks = await ragRetrieve(req);
        setRetrievedChunks(chunks);
      } catch (e: any) {
        onToast('error', e?.response?.data?.error || 'Retrieve failed');
      } finally { setLoading(false); }
      return;
    }

    if (mode === 'stream') {
      setStreamText('');
      setStreaming(true);
      stopRef.current = false;
      let buf = '';
      await streamRagQuery(
        req,
        (token) => { if (!stopRef.current) { buf += token; setStreamText(buf); } },
        () => setStreaming(false),
        (e) => { setStreaming(false); onToast('error', 'Stream error'); }
      );
      return;
    }

    setLoading(true);
    setResult(null);
    try {
      const res = await ragQuery(req);
      setResult(res);
    } catch (e: any) {
      onToast('error', e?.response?.data?.error || 'Query failed');
    } finally { setLoading(false); }
  };

  return (
    <div className="p-4 md:p-6 max-w-7xl mx-auto space-y-5">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
          RAG<span style={{ color: '#00d4ff' }}>//</span>Query
        </h1>
        <p className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
          Retrieve context and generate grounded answers
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* Left: controls */}
        <div className="xl:col-span-1 space-y-4">
          {/* Strategy */}
          <div className="brutal-card p-4 space-y-2">
            <div className="text-xs font-bold uppercase tracking-widest mb-3" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
              Strategy
            </div>
            {STRATEGIES.map(s => (
              <button
                key={s.id}
                onClick={() => setStrategy(s.id)}
                className="w-full text-left px-3 py-2 flex items-center justify-between transition-all"
                style={{
                  border: `2px solid ${strategy === s.id ? s.color : '#2a3a6e'}`,
                  background: strategy === s.id ? s.color + '12' : 'transparent',
                }}
              >
                <span className="text-xs font-bold" style={{ fontFamily: 'Space Mono, monospace', color: strategy === s.id ? s.color : '#6b82b0', fontSize: '0.65rem' }}>
                  {s.label}
                </span>
                {strategy === s.id && (
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.color }} />
                )}
              </button>
            ))}
          </div>

          {/* Mode */}
          <div className="brutal-card p-4 space-y-2">
            <div className="text-xs font-bold uppercase tracking-widest mb-3" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
              Mode
            </div>
            {(['generate', 'stream', 'retrieve'] as const).map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className="w-full text-left px-3 py-2 text-xs uppercase tracking-wider font-bold transition-all"
                style={{
                  fontFamily: 'Space Mono, monospace',
                  border: `2px solid ${mode === m ? '#00d4ff' : '#2a3a6e'}`,
                  background: mode === m ? '#00d4ff12' : 'transparent',
                  color: mode === m ? '#00d4ff' : '#6b82b0',
                  fontSize: '0.65rem',
                }}
              >
                {m === 'generate' ? '⚡ Generate' : m === 'stream' ? '🌊 Stream' : '🔍 Retrieve Only'}
              </button>
            ))}
          </div>

          {/* Params */}
          <div className="brutal-card p-4 space-y-3">
            <div className="text-xs font-bold uppercase tracking-widest mb-1" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
              Parameters
            </div>
            <div>
              <label className="text-xs text-[#4a5a8e] block mb-1" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>
                TOP_K
              </label>
              <input
                className="input-brutal w-full py-2 px-3 text-xs"
                type="number" min={1} max={50} value={topK}
                onChange={e => setTopK(Number(e.target.value))}
                style={{ fontFamily: 'Space Mono, monospace' }}
              />
            </div>
            <div>
              <label className="text-xs text-[#4a5a8e] block mb-1" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>
                FILTERS (JSON)
              </label>
              <textarea
                className="input-brutal w-full py-2 px-3 text-xs resize-none"
                rows={3}
                placeholder='{"language":"en"}'
                value={filters}
                onChange={e => setFilters(e.target.value)}
                style={{ fontFamily: 'IBM Plex Mono, monospace' }}
              />
            </div>
          </div>
        </div>

        {/* Right: query + results */}
        <div className="xl:col-span-2 space-y-4">
          {/* Query input */}
          <div className="brutal-card p-4 space-y-3" style={{ boxShadow: `4px 4px 0 ${activeColor}` }}>
            <div className="flex items-center gap-2">
              <Brain size={14} color={activeColor} />
              <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: activeColor }}>
                Query
              </span>
            </div>
            <textarea
              className="input-brutal w-full py-3 px-3 text-sm resize-none"
              rows={3}
              placeholder="What would you like to know from your documents?"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && e.ctrlKey) handleRun(); }}
              style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                Ctrl+Enter to run
              </span>
              <div className="flex items-center gap-2">
                {streaming && (
                  <button
                    className="flex items-center gap-1 text-xs text-[#ff4d6d]"
                    onClick={() => { stopRef.current = true; setStreaming(false); }}
                  >
                    <StopCircle size={12} /> Stop
                  </button>
                )}
                <button
                  className="btn-brutal px-4 py-2 text-xs flex items-center gap-2"
                  onClick={handleRun}
                  disabled={loading || streaming || !query.trim()}
                  style={{ borderColor: activeColor, color: activeColor, boxShadow: `3px 3px 0 ${activeColor}` }}
                >
                  {loading || streaming ? <Spinner size={12} color={activeColor} /> : <Zap size={12} />}
                  Run
                </button>
              </div>
            </div>
          </div>

          {/* Results */}
          {(loading || streaming) && !streamText && (
            <div className="brutal-card p-8 flex justify-center">
              <Spinner size={28} color={activeColor} label="PROCESSING..." />
            </div>
          )}

          {/* Streaming result */}
          {(streaming || streamText) && mode === 'stream' && (
            <div className="brutal-card p-4" style={{ boxShadow: `4px 4px 0 ${activeColor}` }}>
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: activeColor }}>
                  Streaming Response
                </span>
                {streaming && <Spinner size={10} color={activeColor} />}
              </div>
              <div className="prose prose-invert prose-sm max-w-none" style={{ fontFamily: 'IBM Plex Sans, sans-serif', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.7' }}>
                <ReactMarkdown>{streamText}</ReactMarkdown>
              </div>
              {streaming && <span className="cursor" />}
            </div>
          )}

          {/* Generated result */}
          {result && mode === 'generate' && (
            <div className="space-y-3">
              {/* Meta row */}
              <div className="flex flex-wrap gap-3">
                <div className="flex items-center gap-1.5 px-2 py-1" style={{ border: '1px solid #2a3a6e', background: '#0f1629' }}>
                  <Clock size={10} color="#4a5a8e" />
                  <span className="text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                    {formatLatency(result.latency_ms)}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 px-2 py-1" style={{ border: '1px solid #2a3a6e', background: '#0f1629' }}>
                  <Target size={10} color={activeColor} />
                  <span className="text-xs" style={{ fontFamily: 'IBM Plex Mono, monospace', color: activeColor, fontSize: '0.65rem' }}>
                    {(result.confidence * 100).toFixed(1)}% confidence
                  </span>
                </div>
                <span className="tag-chip" style={{ borderColor: activeColor, color: activeColor }}>
                  {result.strategy}
                </span>
              </div>

              {/* Answer */}
              <div className="brutal-card p-4" style={{ boxShadow: `4px 4px 0 ${activeColor}` }}>
                <div className="text-xs font-bold uppercase tracking-widest mb-3" style={{ fontFamily: 'Space Mono, monospace', color: activeColor }}>
                  Answer
                </div>
                <div className="prose prose-invert prose-sm max-w-none" style={{ fontFamily: 'IBM Plex Sans, sans-serif', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.7' }}>
                  <ReactMarkdown>{result.answer}</ReactMarkdown>
                </div>
                <SourceChips sources={result.sources} />
              </div>
            </div>
          )}

          {/* Retrieved chunks */}
          {retrievedChunks.length > 0 && mode === 'retrieve' && (
            <div className="space-y-3">
              <div className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#00ff9f' }}>
                Retrieved [{retrievedChunks.length}] Chunks
              </div>
              {retrievedChunks.map((c: any, i: number) => (
                <div key={i} className="brutal-card p-3" style={{ boxShadow: '3px 3px 0 #00ff9f44' }}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="badge border-[#00ff9f] text-[#00ff9f]">#{i + 1}</span>
                      <span className="text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                        {truncate(c.filename || c.chunk_id, 40)}
                      </span>
                    </div>
                    <ScoreBar score={c.score} showValue color="#00ff9f" />
                  </div>
                  <p className="text-xs leading-relaxed" style={{ fontFamily: 'IBM Plex Mono, monospace', color: '#8a9abb', fontSize: '0.75rem' }}>
                    {truncate(c.chunk_text, 300)}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
