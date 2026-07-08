import React, { useState } from 'react';
import {
  LayoutDashboard, FileText, FlaskConical, Menu, X, Zap,
  Activity, ChevronRight
} from 'lucide-react';
import type { Page } from '../types';

interface Props {
  currentPage: Page;
  onNavigate: (p: Page) => void;
  children: React.ReactNode;
  healthStatus: Record<string, 'ok' | 'error' | 'unknown'>;
}

const NAV_ITEMS: { id: Page; label: string; icon: React.ElementType; accent?: string }[] = [
  { id: 'dashboard',   label: 'Dashboard',   icon: LayoutDashboard },
  { id: 'documents',   label: 'Documents',   icon: FileText },
  { id: 'evaluate',    label: 'Evaluate',    icon: FlaskConical },
];

export default function Layout({ currentPage, onNavigate, children, healthStatus }: Props) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const allOk = Object.values(healthStatus).every(s => s === 'ok');
  const anyError = Object.values(healthStatus).some(s => s === 'error');

  return (
    <div className="scanline flex h-screen overflow-hidden" style={{ background: '#0a0e1a' }}>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/70 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed md:static z-30 flex flex-col h-full transition-transform duration-200
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
        style={{
          width: 220,
          background: '#0f1629',
          borderRight: '3px solid #2a3a6e',
          flexShrink: 0,
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-2 px-4 py-4 border-b-2" style={{ borderColor: '#2a3a6e' }}>
          <div className="flex items-center justify-center w-8 h-8 border-2 border-[#00d4ff]" style={{ background: '#1a2444' }}>
            <Zap size={14} color="#00d4ff" />
          </div>
          <div>
            <div className="text-xs font-bold uppercase tracking-widest text-[#00d4ff]" style={{ fontFamily: 'Space Mono, monospace' }}>
              Simple RAG
            </div>
            <div className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.6rem' }}>
              v2.0.0
            </div>
          </div>
          <button
            className="ml-auto md:hidden text-[#6b82b0] hover:text-[#00d4ff]"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={16} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-2 overflow-y-auto">
          {NAV_ITEMS.map(item => {
            const Icon = item.icon;
            const active = currentPage === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item w-full text-left ${active ? 'active' : ''}`}
                onClick={() => { onNavigate(item.id); setSidebarOpen(false); }}
              >
                <Icon size={14} />
                <span>{item.label}</span>
                {active && <ChevronRight size={10} className="ml-auto" />}
              </button>
            );
          })}
        </nav>

        {/* Status footer */}
        <div className="p-3 border-t-2" style={{ borderColor: '#2a3a6e' }}>
          <div className="text-xs uppercase tracking-widest mb-2" style={{ fontFamily: 'Space Mono, monospace', color: '#4a5a8e' }}>
            System
          </div>
          {[
            { label: 'API', key: 'api' },
            { label: 'DB', key: 'db' },
            { label: 'Ollama', key: 'ollama' },
          ].map(({ label, key }) => {
            const s = healthStatus[key] || 'unknown';
            return (
              <div key={key} className="flex items-center justify-between py-1">
                <span className="text-xs" style={{ fontFamily: 'IBM Plex Mono, monospace', color: '#6b82b0', fontSize: '0.7rem' }}>
                  {label}
                </span>
                <span
                  className={`status-dot ${s === 'ok' ? 'status-ok status-pulse' : s === 'error' ? 'status-error' : 'status-warn'}`}
                />
              </div>
            );
          })}
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header
          className="flex items-center gap-3 px-4 h-12 border-b-2 flex-shrink-0"
          style={{ background: '#0f1629', borderColor: '#2a3a6e' }}
        >
          <button
            className="md:hidden text-[#6b82b0] hover:text-[#00d4ff] p-1"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={18} />
          </button>

          {/* Breadcrumb */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'Space Mono, monospace' }}>
              Simple RAG
            </span>
            <span className="text-[#2a3a6e] text-xs">/</span>
            <span
              className="text-xs uppercase tracking-widest text-[#00d4ff]"
              style={{ fontFamily: 'Space Mono, monospace' }}
            >
              {currentPage}
            </span>
          </div>

          <div className="ml-auto flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <Activity size={11} color={allOk ? '#00ff9f' : anyError ? '#ff4d6d' : '#ffe600'} />
              <span
                className="text-xs hidden sm:inline"
                style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.7rem', color: allOk ? '#00ff9f' : anyError ? '#ff4d6d' : '#ffe600' }}
              >
                {allOk ? 'ALL_SYSTEMS_OK' : anyError ? 'DEGRADED' : 'CHECKING...'}
              </span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto" style={{ background: '#0a0e1a' }}>
          {children}
        </main>
      </div>
    </div>
  );
}
