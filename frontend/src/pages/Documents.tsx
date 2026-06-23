import React, { useEffect, useState, useRef } from 'react';
import {
  Upload, Trash2, RefreshCw, ChevronDown, ChevronUp,
  FileText, Eye, Layers, Filter, X, AlertCircle, CheckCircle
} from 'lucide-react';
import {
  listDocuments, uploadDocument, deleteDocument,
  reindexDocument, getDocumentChunks, getDocumentMetadata
} from '../api/client';
import Spinner from '../components/Spinner';
import type { Document, Chunk } from '../types';
import { formatDateTime, formatLatency, docTypeBg, truncate } from '../utils/format';

interface Props { onToast: (type: any, msg: string) => void; }

export default function Documents({ onToast }: Props) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Record<string, Chunk[]>>({});
  const [reindexing, setReindexing] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [filterType, setFilterType] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const load = async () => {
    setLoading(true);
    try {
      const res = await listDocuments(0, 100);
      setDocs(res.documents || []);
      setTotal(res.total || 0);
    } catch (e) {
      onToast('error', 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    let success = 0, fail = 0;
    for (const file of Array.from(files)) {
      try {
        // Optional: read sidecar JSON for KAG metadata if present (same name + .json)
        let metadata: any = undefined;
        try {
          const jsonName = file.name.replace(/\.[^.]+$/, '') + '.json';
          // browser cannot read sibling files without input; keeping placeholder for future wiring
          metadata = undefined;
        } catch {
          metadata = undefined;
        }
        await uploadDocument(file, metadata);
        success++;
      } catch {
        fail++;
      }
    }
    setUploading(false);
    if (success > 0) onToast('success', `${success} file(s) uploaded and indexed`);
    if (fail > 0) onToast('error', `${fail} file(s) failed`);
    load();
  };

  const handleDelete = async (id: string) => {
    setDeleting(id);
    try {
      await deleteDocument(id);
      onToast('success', 'Document deleted');
      setDocs(d => d.filter(x => x.id !== id));
    } catch {
      onToast('error', 'Delete failed');
    } finally {
      setDeleting(null);
    }
  };

  const handleReindex = async (id: string) => {
    setReindexing(id);
    try {
      const res = await reindexDocument(id);
      onToast('success', `Reindexed: ${res.chunk_count} chunks`);
      load();
    } catch {
      onToast('error', 'Reindex failed');
    } finally {
      setReindexing(null);
    }
  };

  const toggleExpand = async (id: string) => {
    if (expandedId === id) { setExpandedId(null); return; }
    setExpandedId(id);
    if (!chunks[id]) {
      try {
        const c = await getDocumentChunks(id);
        setChunks(prev => ({ ...prev, [id]: c }));
      } catch { onToast('error', 'Failed to load chunks'); }
    }
  };

  const filtered = filterType ? docs.filter(d => d.document_type === filterType) : docs;

  return (
    <div className="p-4 md:p-6 space-y-5 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
            Documents<span style={{ color: '#ff4d6d' }}>//</span>{total}
          </h1>
          <p className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
            Upload, manage and reindex your knowledge base
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Filter */}
          <div className="relative">
            <select
              className="input-brutal text-xs px-3 py-2 pr-7 appearance-none"
              value={filterType}
              onChange={e => setFilterType(e.target.value)}
              style={{ fontFamily: 'Space Mono, monospace' }}
            >
              <option value="">ALL TYPES</option>
              {['pdf','markdown','csv','text','json'].map(t => (
                <option key={t} value={t}>{t.toUpperCase()}</option>
              ))}
            </select>
            <Filter size={10} className="absolute right-2 top-1/2 -translate-y-1/2 text-[#4a5a8e] pointer-events-none" />
          </div>
          <button
            className="btn-brutal px-3 py-2 text-xs flex items-center gap-2"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? <Spinner size={12} /> : <Upload size={12} />}
            Upload
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.md,.txt,.csv,.json"
            className="hidden"
            onChange={e => handleUpload(e.target.files)}
          />
        </div>
      </div>

      {/* Drop zone */}
      <div
        className={`border-3 border-dashed transition-all p-6 flex flex-col items-center justify-center gap-2 cursor-pointer ${dragOver ? 'border-[#00d4ff] bg-[#00d4ff]/5' : 'border-[#2a3a6e]'}`}
        style={{ borderWidth: 2, borderStyle: 'dashed' }}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); handleUpload(e.dataTransfer.files); }}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload size={20} color={dragOver ? '#00d4ff' : '#2a3a6e'} />
        <span className="text-xs text-center" style={{ fontFamily: 'Space Mono, monospace', color: dragOver ? '#00d4ff' : '#4a5a8e' }}>
          {uploading ? 'PROCESSING...' : 'DROP FILES OR CLICK TO UPLOAD'}
        </span>
        <span className="text-xs text-[#2a3a6e]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
          PDF · MD · TXT · CSV · JSON
        </span>
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex justify-center py-12"><Spinner size={28} label="LOADING_DOCS..." /></div>
      ) : filtered.length === 0 ? (
        <div className="brutal-card p-8 flex flex-col items-center gap-3">
          <FileText size={24} color="#2a3a6e" />
          <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'Space Mono, monospace' }}>
            NO_DOCUMENTS_FOUND
          </span>
        </div>
      ) : (
        <div className="brutal-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 32 }}></th>
                  <th>Filename</th>
                  <th>Type</th>
                  <th className="hidden md:table-cell">Strategy</th>
                  <th>Chunks</th>
                  <th className="hidden lg:table-cell">Collection</th>
                  <th className="hidden lg:table-cell">Uploaded</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(doc => (
                  <React.Fragment key={doc.id}>
                    <tr>
                      <td>
                        <button
                          onClick={() => toggleExpand(doc.id)}
                          className="text-[#4a5a8e] hover:text-[#00d4ff] transition-colors"
                        >
                          {expandedId === doc.id ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </button>
                      </td>
                      <td style={{ maxWidth: 200 }}>
                        <div className="flex items-center gap-2">
                          <FileText size={11} color="#6b82b0" />
                          <span className="truncate text-xs" style={{ color: '#e2e8f0' }}>
                            {doc.filename}
                          </span>
                        </div>
                        <div className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono', fontSize: '0.6rem' }}>
                          {doc.id.slice(0, 8)}...
                        </div>
                      </td>
                      <td>
                        <span className={`tag-chip ${docTypeBg(doc.document_type)}`}>
                          {doc.document_type}
                        </span>
                      </td>
                      <td className="hidden md:table-cell" style={{ fontSize: '0.65rem', color: '#6b82b0' }}>
                        {doc.retrieval_strategy?.replace(/_/g,'_')}
                      </td>
                      <td>
                        <span className="font-bold" style={{ color: '#00ff9f', fontFamily: 'Space Mono, monospace' }}>
                          {doc.chunk_count}
                        </span>
                      </td>
                      <td className="hidden lg:table-cell" style={{ fontSize: '0.65rem', color: '#4a5a8e' }}>
                        {doc.collection_name}
                      </td>
                      <td className="hidden lg:table-cell" style={{ fontSize: '0.65rem', color: '#4a5a8e' }}>
                        {formatDateTime(doc.created_at)}
                      </td>
                      <td>
                        <div className="flex items-center gap-2">
                          <button
                            title="Reindex"
                            onClick={() => handleReindex(doc.id)}
                            disabled={reindexing === doc.id}
                            className="text-[#ffe600] hover:text-[#ffe600]/70 disabled:opacity-40"
                          >
                            {reindexing === doc.id ? <Spinner size={11} color="#ffe600" /> : <RefreshCw size={11} />}
                          </button>
                          <button
                            title="Delete"
                            onClick={() => handleDelete(doc.id)}
                            disabled={deleting === doc.id}
                            className="text-[#ff4d6d] hover:text-[#ff4d6d]/70 disabled:opacity-40"
                          >
                            {deleting === doc.id ? <Spinner size={11} color="#ff4d6d" /> : <Trash2 size={11} />}
                          </button>
                        </div>
                      </td>
                    </tr>
                    {expandedId === doc.id && (
                      <tr>
                        <td colSpan={8} className="p-0">
                          <div className="p-4" style={{ background: '#0a0e1a', borderTop: '2px solid #1a2444' }}>
                            <div className="flex items-center gap-2 mb-3">
                              <Layers size={11} color="#00d4ff" />
                              <span className="text-xs font-bold uppercase tracking-wider" style={{ fontFamily: 'Space Mono, monospace', color: '#00d4ff' }}>
                                Chunks ({chunks[doc.id]?.length ?? '...'})
                              </span>
                            </div>
                            {!chunks[doc.id] ? (
                              <Spinner size={16} label="Loading chunks..." />
                            ) : (
                              <div className="space-y-2 max-h-64 overflow-y-auto">
                                {chunks[doc.id].map(c => (
                                  <div
                                    key={c.id}
                                    className="p-2"
                                    style={{ background: '#0f1629', border: '1px solid #2a3a6e' }}
                                  >
                                    <div className="flex items-center gap-2 mb-1">
                                      <span className="badge border-[#2a3a6e] text-[#6b82b0]">#{c.chunk_index}</span>
                                      {c.chunk_metadata?.section && (
                                        <span className="text-xs text-[#00d4ff]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                                          §{c.chunk_metadata.section}
                                        </span>
                                      )}
                                    </div>
                                    <p className="text-xs text-[#8a9abb] leading-relaxed" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.7rem' }}>
                                      {truncate(c.chunk_text, 200)}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
