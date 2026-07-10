import axios from 'axios';

export const BASE_URL = process.env.REACT_APP_API_URL || 'http://10.64.26.80:9000';

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

// ─── Documents ────────────────────────────────────────────────────────────────

export const uploadDocument = async (file: File, metadata?: Record<string, any>) => {
  const form = new FormData();
  form.append('file', file);
  if (metadata) form.append('metadata', JSON.stringify(metadata));
  const { data } = await api.post('/api/documents/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const uploadDocuments = async (
  files: File[],
  metadata?: Record<string, any>,
  onProgress?: (percent: number) => void
) => {
  const form = new FormData();
  files.forEach(f => form.append('files', f));
  if (metadata) form.append('metadata', JSON.stringify(metadata));
  const { data } = await api.post('/api/documents/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: evt => {
      if (onProgress && evt.total) {
        const percent = Math.round((evt.loaded * 100) / evt.total);
        onProgress(percent);
      }
    },
  });
  return data;
};

export const listDocuments = async (skip = 0, limit = 50) => {
  const { data } = await api.get('/api/documents', { params: { skip, limit } });
  return data;
};

export const getDocument = async (id: string) => {
  const { data } = await api.get(`/api/documents/${id}`);
  return data;
};

export const deleteDocument = async (id: string) => {
  const { data } = await api.delete(`/api/documents/${id}`);
  return data;
};

export const reindexDocument = async (id: string) => {
  const { data } = await api.post(`/api/documents/${id}/reindex`);
  return data;
};

export const getDocumentChunks = async (id: string) => {
  const { data } = await api.get(`/api/documents/${id}/chunks`);
  return data;
};

export const getDocumentMetadata = async (id: string) => {
  const { data } = await api.get(`/api/documents/${id}/metadata`);
  return data;
};

// ─── Search ───────────────────────────────────────────────────────────────────

export const vectorSearch = async (req: any) => {
  const { data } = await api.post('/api/search/vector', req);
  return data;
};

export const bm25Search = async (req: any) => {
  const { data } = await api.post('/api/search/bm25', req);
  return data;
};

export const hybridSearch = async (req: any) => {
  const { data } = await api.post('/api/search/hybrid', req);
  return data;
};

export const metadataSearch = async (req: any) => {
  const { data } = await api.post('/api/search/metadata', req);
  return data;
};

export const tableSearch = async (req: any) => {
  const { data } = await api.post('/api/search/table', req);
  return data;
};

// ─── Chat ─────────────────────────────────────────────────────────────────────

export const sendChat = async (message: string, conversation_id?: string, top_k = 5) => {
  const { data } = await api.post('/api/chat', { message, conversation_id, top_k });
  return data;
};

export const listConversations = async () => {
  const { data } = await api.get('/api/chat/conversations');
  return data;
};

export const getConversation = async (id: string) => {
  const { data } = await api.get(`/api/chat/conversations/${id}`);
  return data;
};

export const deleteConversation = async (id: string) => {
  const { data } = await api.delete(`/api/chat/conversations/${id}`);
  return data;
};

// ─── RAG ─────────────────────────────────────────────────────────────────────

export const ragQuery = async (req: any) => {
  const { data } = await api.post('/api/rag/query', req);
  return data;
};

export const ragRetrieve = async (req: any) => {
  const { data } = await api.post('/api/rag/retrieve', req);
  return data;
};

export const ragEvaluate = async (req: any) => {
  const { data } = await api.post('/api/rag/evaluate', req);
  return data;
};

export const ocrQuestionsFromImages = async (opts: { files?: File[]; use_sample?: boolean }) => {
  const form = new FormData();
  if (opts.use_sample) form.append('use_sample', 'true');
  opts.files?.forEach(file => form.append('files', file));
  const { data } = await api.post('/api/rag/evaluate/images', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

// ─── Agents ──────────────────────────────────────────────────────────────────

export const runAgent = async (type: string, req: any) => {
  const { data } = await api.post(`/api/agents/${type}`, req);
  return data;
};

// ─── Health ──────────────────────────────────────────────────────────────────

export const healthCheck = async () => {
  const { data } = await api.get('/health');
  return data;
};
export const healthDb = async () => {
  const { data } = await api.get('/health/db');
  return data;
};
export const healthChroma = async () => {
  const { data } = await api.get('/health/chroma');
  return data;
};
export const healthOllama = async () => {
  const { data } = await api.get('/health/ollama');
  return data;
};
export const healthNeo4j = async () => {
  const { data } = await api.get('/health/neo4j');
  return data;
};
export const healthQdrant = async () => {
  const { data } = await api.get('/health/qdrant');
  return data;
};
export const healthElasticsearch = async () => {
  const { data } = await api.get('/api/elasticsearch/status');
  return data;
};

// ─── ChromaDB ────────────────────────────────────────────────────────────────

export const listCollections = async () => {
  const { data } = await api.get('/api/chroma/collections');
  return data;
};

// ─── Web ─────────────────────────────────────────────────────────────────────

export const ingestWeb = async (url: string, metadata?: any) => {
  const { data } = await api.post('/api/web/ingest', { url, metadata });
  return data;
};

// ─── Streaming helpers ────────────────────────────────────────────────────────

export const streamRagQuery = async (
  req: any,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (e: any) => void
) => {
  try {
    const response = await fetch(`${BASE_URL}/api/rag/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      onToken(decoder.decode(value));
    }
    onDone();
  } catch (e) {
    onError(e);
  }
};

export const streamChat = async (
  message: string,
  conversation_id: string | undefined,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (e: any) => void
) => {
  try {
    const response = await fetch(`${BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, conversation_id, top_k: 5 }),
    });
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      onToken(decoder.decode(value));
    }
    onDone();
  } catch (e) {
    onError(e);
  }
};

// ─── Live Assist ─────────────────────────────────────────────────────────────

export const pushLiveFrame = async (blob: Blob, session_id?: string) => {
  const form = new FormData();
  form.append('file', new File([blob], 'frame.jpg', { type: blob.type || 'image/jpeg' }));
  if (session_id) form.append('session_id', session_id);
  const { data } = await api.post('/api/live/frame/push', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const liveTranscribe = async (blob: Blob, language?: string) => {
  const form = new FormData();
  form.append('audio', new File([blob], 'audio.webm', { type: blob.type || 'audio/webm' }));
  if (language) form.append('language', language);
  const { data } = await api.post('/api/live/transcribe', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const liveTts = async (text: string, voice?: string) => {
  const resp = await fetch(`${BASE_URL}/api/live/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice }),
  });
  const blob = await resp.blob();
  return URL.createObjectURL(blob);
};

export const clearLiveFrame = async (session_id: string) => {
  const { data } = await api.post('/api/live/frame/clear', { session_id });
  return data;
};

export const streamLiveAsk = async (
  body: {
    question: string;
    conversation_id?: string;
    session_id?: string;
    target_language?: string;
    top_k?: number;
    use_form_context?: boolean;
    manual_context?: string;
  },
  handlers: {
    onToken: (token: string) => void;
    onDone: () => void;
    onError: (e: any) => void;
    onConversation?: (id: string) => void;
  },
) => {
  try {
    const resp = await fetch(`${BASE_URL}/api/live/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const cid = resp.headers.get('x-conversation-id');
    if (cid && handlers.onConversation) handlers.onConversation(cid);
    const reader = resp.body!.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      handlers.onToken(decoder.decode(value));
    }
    handlers.onDone();
  } catch (e) {
    handlers.onError(e);
  }
};
