// ─── Document Types ────────────────────────────────────────────────────────────

export interface Document {
  id: string;
  filename: string;
  filepath: string;
  document_type: 'pdf' | 'markdown' | 'text' | 'csv' | 'json';
  retrieval_strategy: string;
  language: string;
  chunk_count: number;
  embedding_model: string;
  collection_name: string;
  metadata_json: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Chunk {
  id: string;
  document_id: string;
  chunk_index: number;
  chunk_text: string;
  chunk_metadata: Record<string, any>;
  created_at: string;
}

// ─── Search Types ──────────────────────────────────────────────────────────────

export type SearchStrategy = 'vector' | 'bm25' | 'hybrid' | 'metadata' | 'table';

export interface SearchRequest {
  query: string;
  top_k?: number;
  filters?: Record<string, any>;
  collection_name?: string;
}

export interface SearchResult {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_text: string;
  score: number;
  metadata: Record<string, any>;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  confidence: number;
  sources: string[];
  latency_ms: number;
  strategy: string;
}

// ─── Chat Types ────────────────────────────────────────────────────────────────

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: Source[];
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

export interface Source {
  filename: string;
  chunk_id: string;
  score: number;
}

// ─── RAG Types ─────────────────────────────────────────────────────────────────

export type RAGStrategy = 'vector' | 'bm25' | 'hybrid' | 'table' | 'pdf' | 'markdown';

export interface RAGQueryRequest {
  query: string;
  strategy?: RAGStrategy;
  top_k?: number;
  filters?: Record<string, any>;
}

export interface RAGQueryResponse {
  query: string;
  answer: string;
  sources: Source[];
  strategy: string;
  latency_ms: number;
  confidence: number;
}

export interface EvalQuestion {
  question: string;
  expected_answer: string;
}

export interface EvaluationRequest {
  questions: EvalQuestion[];
  dataset_name?: string;
}

export interface EvaluationResponse {
  accuracy: number;
  faithfulness: number;
  context_precision: number;
  context_recall: number;
  answer_relevancy: number;
  latency_avg_ms: number;
  failed_questions: Array<{ question: string; error: string }>;
}

// ─── Agent Types ───────────────────────────────────────────────────────────────

export type AgentType = 'coordinator' | 'vector' | 'sqlite' | 'router' | 'web' | 'evaluator';

export interface AgentRequest {
  query: string;
  context?: Record<string, any>;
  top_k?: number;
  filters?: Record<string, any>;
}

export interface AgentResponse {
  agent: string;
  query: string;
  answer: string;
  sources: Source[];
  reasoning?: string;
  latency_ms: number;
  metadata?: Record<string, any>;
  confidence?: number;
}

// ─── Health Types ──────────────────────────────────────────────────────────────

export interface HealthStatus {
  status: 'ok' | 'error';
  database?: string;
  chromadb?: string;
  ollama?: string;
  collections?: string[];
  models?: string[];
  detail?: string;
}

// ─── ChromaDB Types ────────────────────────────────────────────────────────────

export interface ChromaCollection {
  name: string;
  count: number;
}

// ─── UI State Types ────────────────────────────────────────────────────────────

export type Page =
  | 'dashboard'
  | 'documents'
  | 'evaluate'
  | 'kag';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
}
