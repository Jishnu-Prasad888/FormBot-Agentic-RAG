import React, { useEffect, useState } from 'react';
import { Database, RefreshCw, Trash2, Search, Globe, Plus, Hash } from 'lucide-react';
import { listCollections, ingestWeb } from '../api/client';
import Spinner from '../components/Spinner';

interface Props { onToast: (type: any, msg: string) => void; }

const COLLECTION_META: Record<string, { color: string; desc: string }> = {
  pdf_documents:      { color: '#ff4d6d', desc: 'Hierarchical PDF chunks' },
  markdown_documents: { color: '#00d4ff', desc: 'Header-aware Markdown sections' },
  table_documents:    { color: '#00ff9f', desc: 'CSV schema + row indexes' },
  text_documents:     { color: '#ffe600', desc: 'Plain text and JSON chunks' },
  audio_transcripts:  { color: '#a78bfa', desc: 'Audio transcript chunks' },
  web_documents:      { color: '#38bdf8', desc: 'Web-ingested HTML content' },
};

export default function Collections({ onToast }: Props) {
  const [collections, setCollections] = useState<string[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [webUrl, setWebUrl] = useState('');
  const [ingesting, setIngesting] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const res = await listCollections();
      setCollections(res.collections || []);
      setCounts(res.counts || {});
    } catch { onToast('error', 'Failed to load collections'); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const handleIngestWeb = async () => {
    if (!webUrl.trim()) return;
    setIngesting(true);
    try {
      const res = await ingestWeb(webUrl.trim());
      onToast('success', `Ingested: ${res.chunk_count} chunks from ${webUrl}`);
      setWebUrl('');
      load();
    } catch (e: any) {
      onToast('error', e?.response?.data?.error || 'Web ingest failed');
    } finally { setIngesting(false); }
  };

  const totalDocs = Object.values(counts).reduce((a, b) => a + b, 0);

  return (
    <div className="p-4 md:p-6 max-w-6xl mx-auto space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
            Collections<span style={{ color: '#00ff9f' }}>//</span>ChromaDB
          </h1>
          <p className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
            {collections.length} collections · {totalDocs.toLocaleString()} total chunks indexed
          </p>
        </div>
        <button
          className="btn-brutal px-3 py-2 text-xs flex items-center gap-2"
          onClick={load}
          disabled={loading}
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Refresh
        </button>
      </div>

      {/* Web ingest panel */}
      <div className="brutal-card p-4" style={{ boxShadow: '4px 4px 0 #38bdf8' }}>
        <div className="flex items-center gap-2 mb-3">
          <Globe size={13} color="#38bdf8" />
          <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#38bdf8' }}>
            Web Ingestion
          </span>
        </div>
        <div className="flex gap-2">
          <input
            className="input-brutal flex-1 py-2 px-3 text-xs"
            placeholder="https://example.com/documentation"
            value={webUrl}
            onChange={e => setWebUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleIngestWeb()}
            style={{ fontFamily: 'IBM Plex Mono, monospace' }}
          />
          <button
            className="btn-brutal px-4 py-2 text-xs flex items-center gap-2"
            style={{ borderColor: '#38bdf8', color: '#38bdf8', boxShadow: '3px 3px 0 #38bdf8' }}
            onClick={handleIngestWeb}
            disabled={ingesting || !webUrl.trim()}
          >
            {ingesting ? <Spinner size={12} color="#38bdf8" /> : <Globe size={12} />}
            Ingest
          </button>
        </div>
        <p className="text-xs text-[#4a5a8e] mt-2" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
          Fetches URL, extracts text, chunks, embeds and indexes into web_documents collection
        </p>
      </div>

      {/* Collection grid */}
      {loading ? (
        <div className="flex justify-center py-12"><Spinner size={28} label="LOADING_COLLECTIONS..." /></div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {collections.map(name => {
            const meta = COLLECTION_META[name] || { color: '#6b82b0', desc: 'Vector collection' };
            const count = counts[name] || 0;
            const pct = totalDocs > 0 ? count / totalDocs : 0;
            return (
              <div
                key={name}
                className="brutal-card p-4"
                style={{ boxShadow: `4px 4px 0 ${meta.color}44` }}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="p-2 border-2" style={{ borderColor: meta.color + '44', background: meta.color + '11' }}>
                    <Database size={14} color={meta.color} />
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: meta.color }}>
                      {count.toLocaleString()}
                    </div>
                    <div className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.6rem' }}>
                      chunks
                    </div>
                  </div>
                </div>
                <div className="text-xs font-bold mb-1" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0', fontSize: '0.7rem' }}>
                  {name}
                </div>
                <div className="text-xs text-[#6b82b0] mb-3" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                  {meta.desc}
                </div>
                {/* Fill bar */}
                <div className="score-bar">
                  <div className="score-fill" style={{ width: `${pct * 100}%`, background: `linear-gradient(90deg, ${meta.color}88, ${meta.color})` }} />
                </div>
                <div className="text-xs text-[#4a5a8e] mt-1" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.6rem' }}>
                  {(pct * 100).toFixed(1)}% of total index
                </div>
              </div>
            );
          })}

          {/* Empty state for missing collections */}
          {collections.length === 0 && (
            <div className="col-span-full brutal-card p-8 flex flex-col items-center gap-3">
              <Database size={28} color="#2a3a6e" />
              <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'Space Mono, monospace' }}>
                NO_COLLECTIONS_FOUND
              </span>
              <span className="text-xs text-[#2a3a6e]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                Upload documents to create collections
              </span>
            </div>
          )}
        </div>
      )}

      {/* Index summary table */}
      {collections.length > 0 && (
        <div className="brutal-card overflow-hidden">
          <div className="p-3 flex items-center gap-2" style={{ background: '#1a2444' }}>
            <Hash size={12} color="#6b82b0" />
            <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
              Index Summary
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Collection</th>
                  <th>Chunks</th>
                  <th>% of Total</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody>
                {collections.map(name => {
                  const meta = COLLECTION_META[name] || { color: '#6b82b0', desc: 'Vector collection' };
                  const count = counts[name] || 0;
                  const pct = totalDocs > 0 ? (count / totalDocs * 100).toFixed(1) : '0.0';
                  return (
                    <tr key={name}>
                      <td>
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 flex-shrink-0" style={{ background: meta.color }} />
                          <span style={{ color: '#e2e8f0', fontFamily: 'IBM Plex Mono, monospace' }}>{name}</span>
                        </div>
                      </td>
                      <td style={{ color: meta.color, fontFamily: 'Space Mono, monospace' }}>
                        {count.toLocaleString()}
                      </td>
                      <td style={{ color: '#6b82b0' }}>{pct}%</td>
                      <td>
                        <span className="text-xs" style={{ color: '#4a5a8e', fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                          {meta.desc}
                        </span>
                      </td>
                    </tr>
                  );
                })}
                <tr>
                  <td style={{ color: '#00d4ff', fontFamily: 'Space Mono, monospace', fontWeight: 'bold' }}>TOTAL</td>
                  <td style={{ color: '#00d4ff', fontFamily: 'Space Mono, monospace', fontWeight: 'bold' }}>
                    {totalDocs.toLocaleString()}
                  </td>
                  <td style={{ color: '#00d4ff' }}>100%</td>
                  <td></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
