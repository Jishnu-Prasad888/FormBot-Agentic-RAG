import React, { useEffect } from 'react';
import { CheckCircle, XCircle, Info, AlertTriangle, X } from 'lucide-react';
import type { Toast } from '../types';

interface Props {
  toasts: Toast[];
  onRemove: (id: string) => void;
}

const ICONS = {
  success: CheckCircle,
  error: XCircle,
  info: Info,
  warning: AlertTriangle,
};

const COLORS = {
  success: { border: '#00ff9f', color: '#00ff9f', bg: 'rgba(0,255,159,0.1)' },
  error:   { border: '#ff4d6d', color: '#ff4d6d', bg: 'rgba(255,77,109,0.1)' },
  info:    { border: '#00d4ff', color: '#00d4ff', bg: 'rgba(0,212,255,0.1)' },
  warning: { border: '#ffe600', color: '#ffe600', bg: 'rgba(255,230,0,0.1)' },
};

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: () => void }) {
  useEffect(() => {
    const t = setTimeout(onRemove, 4000);
    return () => clearTimeout(t);
  }, [onRemove]);

  const Icon = ICONS[toast.type];
  const c = COLORS[toast.type];

  return (
    <div
      className="flex items-start gap-3 p-3 min-w-[280px] max-w-[380px] animate-fade-up"
      style={{
        background: c.bg,
        border: `2px solid ${c.border}`,
        boxShadow: `3px 3px 0 ${c.border}`,
        fontFamily: 'IBM Plex Mono, monospace',
      }}
    >
      <Icon size={14} color={c.color} className="flex-shrink-0 mt-0.5" />
      <span className="text-xs flex-1" style={{ color: '#e2e8f0' }}>{toast.message}</span>
      <button onClick={onRemove} className="flex-shrink-0 text-[#4a5a8e] hover:text-[#e2e8f0]">
        <X size={12} />
      </button>
    </div>
  );
}

export default function ToastContainer({ toasts, onRemove }: Props) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map(t => (
        <ToastItem key={t.id} toast={t} onRemove={() => onRemove(t.id)} />
      ))}
    </div>
  );
}
