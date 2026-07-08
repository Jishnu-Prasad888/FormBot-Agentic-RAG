import React, { useEffect, useState } from 'react';
import {
  FileText, MessageSquare, Activity,
  ArrowRight, Zap, CheckCircle, XCircle,
  Clock
} from 'lucide-react';
import { listDocuments, listConversations, healthCheck, healthDb, healthOllama } from '../api/client';
import Spinner from '../components/Spinner';
import type { Page } from '../types';
import { formatDateTime } from '../utils/format';

interface Props {
  onNavigate: (p: Page) => void;
  onHealthUpdate: (h: Record<string, 'ok' | 'error' | 'unknown'>) => void;
}

export default function Dashboard({ onNavigate, onHealthUpdate }: Props) {
  const [stats, setStats] = useState({ docs: 0, convs: 0 });
  const [health, setHealth] = useState<Record<string, any>>({});
  const [recentDocs, setRecentDocs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [docsRes, convsRes] = await Promise.allSettled([
          listDocuments(0, 5),
          listConversations(),
        ]);

        if (docsRes.status === 'fulfilled') {
          setStats(s => ({ ...s, docs: docsRes.value.total || 0 }));
          setRecentDocs(docsRes.value.documents?.slice(0, 5) || []);
        }
        if (convsRes.status === 'fulfilled') {
          setStats(s => ({ ...s, convs: convsRes.value.total || 0 }));
        }

        // Health checks
        const [api, db, ollama] = await Promise.allSettled([
          healthCheck(), healthDb(), healthOllama(),
        ]);
        const h = {
          api:    api.status === 'fulfilled' && api.value.status === 'ok' ? 'ok' : 'error',
          db:     db.status === 'fulfilled' && db.value.status === 'ok' ? 'ok' : 'error',
          ollama: ollama.status === 'fulfilled' && ollama.value.status === 'ok' ? 'ok' : 'error',
        } as Record<string, 'ok' | 'error'>;
        setHealth({ ...h,
          ollamaModels: ollama.status === 'fulfilled' ? ollama.value.models : [],
        });
        onHealthUpdate(h);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner size={32} label="LOADING_SYSTEM..." />
      </div>
    );
  }

  const STAT_CARDS = [
    { label: 'Documents', value: stats.docs, icon: FileText, color: '#ff4d6d', page: 'documents' as Page, desc: 'Indexed files' },
    { label: 'Conversations', value: stats.convs, icon: MessageSquare, color: '#00d4ff', page: 'evaluate' as Page, desc: 'Chat sessions' },
  ];

  const HEALTH_CHECKS = [
    { key: 'api',    label: 'FastAPI', detail: 'REST API' },
    { key: 'db',     label: 'JSON Store',  detail: 'File-based storage' },
    { key: 'ollama', label: 'Ollama',  detail: `${health.ollamaModels?.length || 0} models` },
  ];

  const QUICK_ACTIONS = [
    { label: 'Upload Document', desc: 'Index new file', color: '#ff4d6d', page: 'documents' as Page },
    { label: 'Evaluate', desc: 'Score your RAG', color: '#a78bfa', page: 'evaluate' as Page },
  ];

  return (
    <div className="p-4 md:p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <div className="w-8 h-0.5" style={{ background: '#00d4ff' }} />
          <span className="text-xs uppercase tracking-widest text-[#4a5a8e]" style={{ fontFamily: 'Space Mono, monospace' }}>
            System Overview
          </span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
          Simple<span style={{ color: '#00d4ff' }}>//</span>RAG
        </h1>
        <p className="text-sm text-[#6b82b0] mt-1" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
          Simple Retrieval-Augmented Generation
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {STAT_CARDS.map(card => {
          const Icon = card.icon;
          return (
            <button
              key={card.label}
              className="brutal-card p-4 text-left"
              onClick={() => onNavigate(card.page)}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="p-2 border-2" style={{ borderColor: card.color + '44', background: card.color + '11' }}>
                  <Icon size={16} color={card.color} />
                </div>
                <ArrowRight size={12} color="#4a5a8e" />
              </div>
              <div className="text-2xl md:text-3xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: card.color }}>
                {card.value.toString().padStart(2, '0')}
              </div>
              <div className="text-xs font-bold uppercase tracking-wider mt-1" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
                {card.label}
              </div>
              <div className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                {card.desc}
              </div>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Health checks */}
        <div className="brutal-card p-4">
          <div className="flex items-center gap-2 mb-4">
            <Activity size={14} color="#00ff9f" />
            <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#00ff9f' }}>
              System Health
            </span>
          </div>
          <div className="space-y-2">
            {HEALTH_CHECKS.map(({ key, label, detail }) => {
              const ok = health[key] === 'ok';
              return (
                <div key={key} className="flex items-center justify-between py-2 border-b" style={{ borderColor: '#1a2444' }}>
                  <div className="flex items-center gap-3">
                    <span className={`status-dot ${ok ? 'status-ok status-pulse' : 'status-error'}`} />
                    <div>
                      <div className="text-xs font-bold" style={{ fontFamily: 'Space Mono, monospace', color: ok ? '#e2e8f0' : '#ff4d6d' }}>
                        {label}
                      </div>
                      <div className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                        {detail}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {ok ? <CheckCircle size={12} color="#00ff9f" /> : <XCircle size={12} color="#ff4d6d" />}
                    <span className="text-xs" style={{ fontFamily: 'IBM Plex Mono, monospace', color: ok ? '#00ff9f' : '#ff4d6d', fontSize: '0.65rem' }}>
                      {ok ? 'ONLINE' : 'OFFLINE'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Quick actions */}
        <div className="brutal-card p-4">
          <div className="flex items-center gap-2 mb-4">
            <Zap size={14} color="#ffe600" />
            <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#ffe600' }}>
              Quick Actions
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {QUICK_ACTIONS.map(action => (
              <button
                key={action.label}
                className="p-3 text-left transition-all"
                style={{
                  background: action.color + '0d',
                  border: `2px solid ${action.color}33`,
                  fontFamily: 'Space Mono, monospace',
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = action.color;
                  (e.currentTarget as HTMLElement).style.background = action.color + '1a';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = action.color + '33';
                  (e.currentTarget as HTMLElement).style.background = action.color + '0d';
                }}
                onClick={() => onNavigate(action.page)}
              >
                <div className="text-xs font-bold uppercase tracking-wide" style={{ color: action.color, fontSize: '0.65rem' }}>
                  {action.label}
                </div>
                <div className="text-xs mt-0.5 text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>
                  {action.desc}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Recent docs */}
      {recentDocs.length > 0 && (
        <div className="brutal-card p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Clock size={14} color="#00d4ff" />
              <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#00d4ff' }}>
                Recent Documents
              </span>
            </div>
            <button
              className="text-xs text-[#4a5a8e] hover:text-[#00d4ff] flex items-center gap-1"
              style={{ fontFamily: 'Space Mono, monospace' }}
              onClick={() => onNavigate('documents')}
            >
              View all <ArrowRight size={10} />
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Type</th>
                  <th>Model</th>
                  <th>Chunks</th>
                  <th>Uploaded</th>
                </tr>
              </thead>
              <tbody>
                {recentDocs.map(doc => (
                  <tr key={doc.id}>
                    <td style={{ color: '#e2e8f0', maxWidth: 200 }}>
                      <span className="truncate block">{doc.filename}</span>
                    </td>
                    <td>
                      <span className="tag-chip" style={{
                        borderColor: doc.document_type === 'pdf' ? '#ff4d6d' :
                          doc.document_type === 'csv' ? '#00ff9f' :
                          doc.document_type === 'markdown' ? '#00d4ff' : '#ffe600',
                        color: doc.document_type === 'pdf' ? '#ff4d6d' :
                          doc.document_type === 'csv' ? '#00ff9f' :
                          doc.document_type === 'markdown' ? '#00d4ff' : '#ffe600',
                      }}>
                        {doc.document_type}
                      </span>
                    </td>
                    <td style={{ color: '#6b82b0', fontSize: '0.7rem' }}>
                      {doc.embedding_model || 'vector'}
                    </td>
                    <td style={{ color: '#00ff9f' }}>{doc.chunk_count}</td>
                    <td style={{ color: '#4a5a8e', fontSize: '0.7rem' }}>{formatDateTime(doc.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
