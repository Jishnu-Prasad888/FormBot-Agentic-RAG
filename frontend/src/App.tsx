import React, { useState, useCallback, useEffect } from 'react';
import Layout from './components/Layout';
import ToastContainer from './components/Toast';
import Dashboard from './pages/Dashboard';
import Documents from './pages/Documents';
import Evaluate from './pages/Evaluate';
import type { Page, Toast } from './types';
import { healthCheck, healthDb, healthChroma, healthOllama } from './api/client';

export default function App() {
  const [page, setPage] = useState<Page>('dashboard');
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [healthStatus, setHealthStatus] = useState<Record<string, 'ok' | 'error' | 'unknown'>>({
    api: 'unknown', db: 'unknown', chroma: 'unknown', ollama: 'unknown',
  });

  const addToast = useCallback((type: Toast['type'], message: string) => {
    const id = Math.random().toString(36).slice(2);
    setToasts(t => [...t, { id, type, message }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts(t => t.filter(x => x.id !== id));
  }, []);

  const updateHealth = useCallback((h: Record<string, 'ok' | 'error' | 'unknown'>) => {
    setHealthStatus(h);
  }, []);

  // Poll health every 30s
  useEffect(() => {
    const check = async () => {
      const [api, db, chroma, ollama] = await Promise.allSettled([
        healthCheck(), healthDb(), healthChroma(), healthOllama(),
      ]);
      setHealthStatus({
        api:    api.status    === 'fulfilled' && api.value.status    === 'ok' ? 'ok' : 'error',
        db:     db.status     === 'fulfilled' && db.value.status     === 'ok' ? 'ok' : 'error',
        chroma: chroma.status === 'fulfilled' && chroma.value.status === 'ok' ? 'ok' : 'error',
        ollama: ollama.status === 'fulfilled' && ollama.value.status === 'ok' ? 'ok' : 'error',
      });
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  const commonProps = { onToast: addToast };

  const renderPage = () => {
    switch (page) {
      case 'dashboard':   return <Dashboard onNavigate={setPage} onHealthUpdate={updateHealth} />;
      case 'documents':   return <Documents {...commonProps} />;
      case 'evaluate':    return <Evaluate {...commonProps} />;
      default:            return <Dashboard onNavigate={setPage} onHealthUpdate={updateHealth} />;
    }
  };

  return (
    <>
      <Layout currentPage={page} onNavigate={setPage} healthStatus={healthStatus}>
        {renderPage()}
      </Layout>
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </>
  );
}
