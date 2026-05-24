import React, { useState } from 'react';
import { Search, Zap, Clock, ChevronDown, ChevronUp, Filter } from 'lucide-react';
import { vectorSearch, bm25Search, hybridSearch, metadataSearch, tableSearch } from '../api/client';
import Spinner from '../components/Spinner';
import ScoreBar from '../components/ScoreBar';
import type { SearchResponse, SearchStrategy } from '../types';
import { formatLatency, truncate, strategyColor } from '../utils/format';

interface Props { onToast: (type: any, msg: string) => void; }

const STRATEGIES: { id: SearchStrategy; label: string; desc: string; color: string }[] = [
  { id: 'hybrid',   label: 'Hybrid',   desc: 'Vector + BM25 + RRF', color: '#00d4ff' },
  { id: 'vector',   label: 'Vector',   desc: 'Dense embeddings',     color: '#38bdf8' },
  { id: 'bm25',     label: 'BM25',     desc: 'Keyword sparse',       color: '#ffe600' },
  { id: 'metadata', label: 'Metadata', desc: 'Filter-first search',  color: '#a78bfa' },
  { id: 'table',    label: 'Table',    desc: 'TableRAG CSV data',    color: '#00ff9f' },
];

export default function SearchPage({ onToast }: Props) {
  const [query, setQuery] = useState('');
  const [strategy, setStrategy] = useState<SearchStrategy>('hybrid');
  const [topK, setTopK] = useState(5);
  const [filters, setFilters] = useState('');
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [history, setHistory] = useState<{ query: string; strategy: string; results: number; latency: number }[]>([]);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      let parsedFilters: any = undefined;
      if (filters.trim()) {
        try { parsedFilters = JSON.parse(filters); } catch { onToast('warning', 'Invalid filter JSON — ignored'); }
      }
      const req = { query: query.trim(), top_k: topK, filters: parsedFilters };
      let res: SearchResponse;
      if (strategy === 'vector')   res = await vectorSearch(req);
      else if (strategy === 'bm25') res = await bm25Search(req);
      else if (strategy === 'metadata') res = await metadataSearch(req);
      else if (strategy === 'table') res = await tableSearch(req);
      else res = await hybridSearch(req);
      setResult(res);
      setHistory(h => [{ query: query.trim(), strategy, results: res.results.length, latency: res.latency_ms }, ...h.slice(0, 9)]);
    } catch (e: any) {
      onToast('error', e?.response?.data?.error || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const activeStrategy = STRATEGIES.find(s => s.id === strategy)!;

  return (
    <div className="p-4 md:p-6 max-w-7xl mx-auto space-y-5">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
          Search<span style={{ color: '#38bdf8' }}>//</span>Retrieval
        </h1>
        <p className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
          Query your indexed documents across multiple retrieval strategies
        </p>
      </div>

      {/* Strategy selector */}
      <div className="flex flex-wrap gap-2">
        {STRATEGIES.map(s => (
          <button
            key={s.id}
            onClick={() => setStrategy(s.id)}
            className="px-3 py-2 text-left transition-all"
            style={{
              border: `2px solid ${strategy === s.id ? s.color : '#2a3a6e'}`,
              background: strategy === s.id ? s.color + '15' : 'transparent',
              boxShadow: strategy === s.id ? `3px 3px 0 ${s.color}` : 'none',
              transform: strategy === s.id ? 'translate(-2px, -2px)' : 'none',
            }}
          >
            <div className="text-xs font-bold uppercase tracking-wider" style={{ fontFamily: 'Space Mono, monospace', color: strategy === s.id ? s.color : '#6b82b0', fontSize: '0.65rem' }}>
              {s.label}
            </div>
            <div className="text-xs text-[#4a5a8e] hidden sm:block" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.6rem' }}>
              {s.desc}
            </div>
          </button>
        ))}
      </div>

      {/* Search box */}
      <div className="brutal-card p-4 space-y-3">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#4a5a8e]" />
            <input
              className="input-brutal w-full pl-9 pr-4 py-3 text-sm"
              placeholder={`Search with ${activeStrategy.label} strategy...`}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
            />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-[#4a5a8e] hidden sm:inline" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.65rem' }}>K=</span>
            <input
              className="input-brutal w-14 py-3 px-2 text-xs text-center"
              type="number" min={1} max={50} value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              style={{ fontFamily: 'Space Mono, monospace' }}
            />
          </div>
          <button
            className="btn-brutal px-4 py-3 flex items-center gap-2 text-xs"
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            style={{ borderColor: activeStrategy.color, color: activeStrategy.color, boxShadow: `3px 3px 0 ${activeStrategy.color}` }}
          >
            {loading ? <Spinner size={12} color={activeStrategy.color} /> : <Zap size={12} />}
            <span className="hidden sm:inline">Search</span>
          </button>
        </div>

        {/* Filter row */}
        <div className="flex items-center gap-2">
          <Filter size={11} color="#4a5a8e" />
          <input
            className="input-brutal flex-1 py-2 px-3 text-xs"
            placeholder='Metadata filters JSON: {"language":"en","document_type":"pdf"}'
            value={filters}
            onChange={e => setFilters(e.target.value)}
            style={{ fontFamily: 'IBM Plex Mono, monospace' }}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Results */}
        <div className="lg:col-span-2 space-y-3">
          {loading && (
            <div className="brutal-card p-8 flex justify-center">
              <Spinner size={28} color={activeStrategy.color} label={`${strategy.toUpperCase()}_SEARCH...`} />
            </div>
          )}

          {result && !loading && (
            <>
              {/* Meta */}
              <div className="flex flex-wrap items-center gap-3 px-1">
                <span className="text-xs font-bold" style={{ fontFamily: 'Space Mono, monospace', color: activeStrategy.color }}>
                  {result.results.length} RESULTS
                </span>
                <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                  <Clock size={9} className="inline mr-1" />{formatLatency(result.latency_ms)}
                </span>
                <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                  CONF: {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>

              {result.results.length === 0 ? (
                <div className="brutal-card p-6 text-center">
                  <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'Space Mono, monospace' }}>NO_RESULTS_FOUND</span>
                </div>
              ) : (
                result.results.map((r, i) => (
                  <div key={i} className="brutal-card overflow-hidden" style={{ boxShadow: `3px 3px 0 ${activeStrategy.color}44` }}>
                    <div
                      className="flex items-center justify-between p-3 cursor-pointer"
                      style={{ background: '#1a2444', borderBottom: expandedIdx === i ? '2px solid #2a3a6e' : 'none' }}
                      onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="badge border-[#2a3a6e] text-[#4a5a8e] flex-shrink-0">#{i + 1}</span>
                        <span className="text-xs truncate" style={{ fontFamily: 'IBM Plex Mono, monospace', color: '#a0b4d0' }}>
                          {r.filename || r.chunk_id}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 flex-shrink-0 ml-2">
                        <div className="hidden sm:flex items-center gap-2 w-24">
                          <ScoreBar score={r.score} showValue={false} color={activeStrategy.color} />
                          <span className="text-xs font-bold w-10 text-right" style={{ fontFamily: 'Space Mono, monospace', color: activeStrategy.color, fontSize: '0.65rem' }}>
                            {(r.score * 100).toFixed(0)}%
                          </span>
                        </div>
                        {expandedIdx === i ? <ChevronUp size={12} color="#6b82b0" /> : <ChevronDown size={12} color="#6b82b0" />}
                      </div>
                    </div>
                    {expandedIdx === i && (
                      <div className="p-3 space-y-2">
                        <p className="text-xs leading-relaxed" style={{ fontFamily: 'IBM Plex Mono, monospace', color: '#8a9abb', fontSize: '0.75rem' }}>
                          {r.chunk_text}
                        </p>
                        {r.metadata && Object.keys(r.metadata).length > 0 && (
                          <div className="flex flex-wrap gap-1 pt-2 border-t border-[#1a2444]">
                            {Object.entries(r.metadata).filter(([k]) => ['section','page_number','document_type','language','chunk_type'].includes(k)).map(([k, v]) => (
                              <span key={k} className="text-xs px-2 py-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace', background: '#1a2444', border: '1px solid #2a3a6e', color: '#6b82b0', fontSize: '0.6rem' }}>
                                {k}={String(v)}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              )}
            </>
          )}
        </div>

        {/* Sidebar: history */}
        <div className="space-y-3">
          <div className="brutal-card p-3">
            <div className="text-xs font-bold uppercase tracking-widest mb-3" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
              Search History
            </div>
            {history.length === 0 ? (
              <span className="text-xs text-[#2a3a6e]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>No searches yet</span>
            ) : (
              <div className="space-y-2">
                {history.map((h, i) => (
                  <button
                    key={i}
                    className="w-full text-left p-2 border border-[#2a3a6e] hover:border-[#00d4ff] transition-colors"
                    style={{ background: '#0a0e1a' }}
                    onClick={() => { setQuery(h.query); setStrategy(h.strategy as SearchStrategy); }}
                  >
                    <div className="text-xs truncate" style={{ fontFamily: 'IBM Plex Mono, monospace', color: '#a0b4d0', fontSize: '0.7rem' }}>
                      {truncate(h.query, 35)}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="tag-chip" style={{ borderColor: strategyColor(h.strategy), color: strategyColor(h.strategy), fontSize: '0.55rem' }}>
                        {h.strategy}
                      </span>
                      <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.6rem' }}>
                        {h.results} results · {formatLatency(h.latency)}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
