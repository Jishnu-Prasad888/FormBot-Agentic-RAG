import React from 'react';
import { Database, Network, Share2, GitBranch, Shield, Rocket, Settings, Repeat, BarChart } from 'lucide-react';

const sections = [
  {
    title: 'Architecture Snapshot',
    items: [
      'PostgreSQL @7000 for metadata, versions, audit',
      'Neo4j + APOC for relationships and conditional logic',
      'Qdrant @6333/6334 for 3072-dim embeddings (text-embedding-3-large)',
      'LLM layer synthesizes vector + graph + metadata',
    ],
    accent: '#00d4ff',
    icon: <Network size={14} color="#00d4ff" />,
  },
  {
    title: 'Document Prep',
    items: [
      'Chunk by structure: fields/sections/steps (200–500 tokens, 50–100 overlap)',
      'Extract fields, sections, references, requirements, conditions',
      'Enrich metadata: version, date, type, tags, regulatory refs, confidence',
      'PDFs: LLM + synonym dictionary to add context-rich metadata',
    ],
    accent: '#00ff9f',
    icon: <Share2 size={14} color="#00ff9f" />,
  },
  {
    title: 'Ingestion',
    items: [
      'Postgres: documents + chunks with content_hash, version, chunk_type, entities',
      'Qdrant: payload with version, document_type, chunk_position, field_name, regulatory_reference',
      'Neo4j: Form/FormVersion/Field/Requirement/Regulation nodes + REQUIRES/REFERENCES/DEPENDS_ON/SUPERSEDES',
    ],
    accent: '#ffe600',
    icon: <Database size={14} color="#ffe600" />,
  },
  {
    title: 'Retrieval & RAG',
    items: [
      'Tier 1: Postgres keyword/ILIKE for exact names/forms',
      'Tier 2: Qdrant semantic with metadata filters (type, version, domain)',
      'Tier 3: Neo4j traversal (depth ≤3) for related forms/requirements/regulations',
      'Merge, rerank, ground with source attributions and uncertainty notes',
    ],
    accent: '#a78bfa',
    icon: <GitBranch size={14} color="#a78bfa" />,
  },
];

const roadmap = [
  { label: 'Weeks 1–2', detail: 'Bring up DBs, schemas, initial load, health checks', color: '#00d4ff', value: 0.3 },
  { label: 'Weeks 3–4', detail: 'Graph build: fields, requirements, regulations, versioning', color: '#00ff9f', value: 0.55 },
  { label: 'Week 5', detail: 'Vector tuning: chunk refinement, thresholds, payload indexes', color: '#ffe600', value: 0.7 },
  { label: 'Week 6', detail: 'Query pipeline & RAG grounding, ranking, verification', color: '#ff4d6d', value: 0.85 },
  { label: 'Week 7+', detail: 'Monitoring, audits, version rollbacks, scale-out', color: '#a78bfa', value: 1.0 },
];

export default function KagGuide() {
  return (
    <div className="p-4 md:p-6 space-y-6 max-w-6xl mx-auto">
      <header className="space-y-2">
        <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-[#4a5a8e]" style={{ fontFamily: 'Space Mono, monospace' }}>
          <div className="w-8 h-0.5 bg-[#00d4ff]" />
          KAG for Bank Documents
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl md:text-3xl font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#e2e8f0' }}>
            Knowledge-Augmented Generation Playbook
          </h1>
          <span className="text-xs px-2 py-1" style={{ border: '2px solid #00ff9f', color: '#00ff9f', fontFamily: 'Space Mono, monospace' }}>
            text-embedding-3-large
          </span>
          <span className="text-xs px-2 py-1" style={{ border: '2px solid #ffe600', color: '#ffe600', fontFamily: 'Space Mono, monospace' }}>
            Postgres @7000
          </span>
          <span className="text-xs px-2 py-1" style={{ border: '2px solid #00d4ff', color: '#00d4ff', fontFamily: 'Space Mono, monospace' }}>
            Neo4j + APOC @7687
          </span>
          <span className="text-xs px-2 py-1" style={{ border: '2px solid #ff4d6d', color: '#ff4d6d', fontFamily: 'Space Mono, monospace' }}>
            Qdrant @6333/6334
          </span>
        </div>
        <p className="text-sm text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
          Retrieval chain: keyword (Postgres) → semantic (Qdrant) → graph (Neo4j) → grounded generation with citations.
        </p>
      </header>

      {/* Architecture */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sections.map((section) => (
          <div key={section.title} className="brutal-card p-4 space-y-2">
            <div className="flex items-center gap-2">
              {section.icon}
              <span className="text-xs uppercase tracking-widest" style={{ color: section.accent, fontFamily: 'Space Mono, monospace' }}>
                {section.title}
              </span>
            </div>
            <ul className="space-y-1">
              {section.items.map((item) => (
                <li key={item} className="text-xs text-[#8a9abb]" style={{ fontFamily: 'IBM Plex Mono, monospace', lineHeight: 1.5 }}>
                  • {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Pipelines */}
      <div className="brutal-card p-4 space-y-4">
        <div className="flex items-center gap-2">
          <Rocket size={14} color="#00ff9f" />
          <span className="text-xs uppercase tracking-widest" style={{ color: '#00ff9f', fontFamily: 'Space Mono, monospace' }}>Pipelines</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
          <div>
            <div className="text-[#e2e8f0] font-semibold mb-1" style={{ fontFamily: 'Space Mono, monospace' }}>PostgreSQL</div>
            <div className="text-[#8a9abb]">Documents (id, name, content_hash, version, status), chunks (type, summary, entities, vector_id), structured forms and requirements.</div>
          </div>
          <div>
            <div className="text-[#e2e8f0] font-semibold mb-1" style={{ fontFamily: 'Space Mono, monospace' }}>Neo4j</div>
            <div className="text-[#8a9abb]">Form → Field → Requirement → Regulation; dependencies, REQUIRES_IF, SUPERSEDES, RELATED_TO; customer/account mappings.</div>
          </div>
          <div>
            <div className="text-[#e2e8f0] font-semibold mb-1" style={{ fontFamily: 'Space Mono, monospace' }}>Qdrant</div>
            <div className="text-[#8a9abb]">Collections: bank_forms_collection, regulations_collection, guidelines_collection, plus type-based. Payload filters on document_type/version/domain; HNSW M=32, ef=64, ef_construct=200.</div>
          </div>
        </div>
      </div>

      {/* Retrieval tiers */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          {
            title: 'Tier 1: Exact',
            color: '#00ff9f',
            lines: [
              'Postgres keyword / ILIKE',
              'Fast path for form names, ids, citations',
              'Return immediately if high confidence',
            ],
          },
          {
            title: 'Tier 2: Semantic',
            color: '#00d4ff',
            lines: [
              'Qdrant semantic search (top-k=10)',
              'Metadata filters: document_type, version, domain, chunk_type',
              'Hybrid with keyword boost + rerank',
            ],
          },
          {
            title: 'Tier 3: Graph',
            color: '#a78bfa',
            lines: [
              'Neo4j traversal depth ≤3',
              'Form ↔ Regulation ↔ Requirement ↔ Field',
              'Adds candidates and relationship paths for grounding',
            ],
          },
        ].map(card => (
          <div key={card.title} className="brutal-card p-4 space-y-2">
            <div className="text-xs uppercase tracking-widest" style={{ color: card.color, fontFamily: 'Space Mono, monospace' }}>{card.title}</div>
            <ul className="space-y-1">
              {card.lines.map(line => (
                <li key={line} className="text-xs text-[#8a9abb]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>• {line}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Roadmap */}
      <div className="brutal-card p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Settings size={14} color="#ffe600" />
          <span className="text-xs uppercase tracking-widest" style={{ color: '#ffe600', fontFamily: 'Space Mono, monospace' }}>Implementation Roadmap</span>
        </div>
        <div className="space-y-2">
          {roadmap.map(step => (
            <div key={step.label} className="space-y-1">
              <div className="flex items-center justify-between">
                <div className="text-xs" style={{ color: '#e2e8f0', fontFamily: 'Space Mono, monospace' }}>{step.label}</div>
                <div className="text-xs" style={{ color: '#6b82b0', fontFamily: 'IBM Plex Mono, monospace' }}>{step.detail}</div>
              </div>
              <div className="score-bar">
                <div className="score-fill" style={{ width: `${Math.round(step.value * 100)}%`, background: `linear-gradient(90deg, ${step.color}55, ${step.color})` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Compliance & Ops */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="brutal-card p-4 space-y-2">
          <div className="flex items-center gap-2">
            <Shield size={14} color="#ff4d6d" />
            <span className="text-xs uppercase tracking-widest" style={{ color: '#ff4d6d', fontFamily: 'Space Mono, monospace' }}>Compliance & Security</span>
          </div>
          <ul className="space-y-1">
            {['No PII in embeddings; redact before processing', 'Track version lineage + SUPERSEDES links', 'Role-based access by document_type/domain', 'Audit trails stored in Postgres; cite sources in answers'].map(line => (
              <li key={line} className="text-xs text-[#8a9abb]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>• {line}</li>
            ))}
          </ul>
        </div>
        <div className="brutal-card p-4 space-y-2">
          <div className="flex items-center gap-2">
            <Repeat size={14} color="#00ff9f" />
            <span className="text-xs uppercase tracking-widest" style={{ color: '#00ff9f', fontFamily: 'Space Mono, monospace' }}>Operations</span>
          </div>
          <ul className="space-y-1">
            {['Re-embed changed chunks only; keep old vectors archived', 'Metadata indexes on document_type, version, domain for fast filters', 'Cache common product/customer journeys', 'Monitor latency + answer quality; add human review for compliance queries'].map(line => (
              <li key={line} className="text-xs text-[#8a9abb]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>• {line}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
        <BarChart size={14} color="#00d4ff" />
        Built to answer: “What documents do I need for mortgage approval?” by fusing semantic chunks, graph relationships, and verified metadata.
      </div>
    </div>
  );
}
