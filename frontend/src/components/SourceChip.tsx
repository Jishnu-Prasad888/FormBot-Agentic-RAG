import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import type { Source } from '../types';
import { formatScore, truncate } from '../utils/format';

interface Props {
  sources: Source[];
}

export default function SourceChips({ sources }: Props) {
  const [expanded, setExpanded] = useState(false);
  if (!sources || sources.length === 0) return null;

  const shown = expanded ? sources : sources.slice(0, 3);

  return (
    <div className="mt-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs uppercase tracking-widest text-[#4a5a8e]" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.6rem' }}>
          Sources [{sources.length}]
        </span>
        {sources.length > 3 && (
          <button
            onClick={() => setExpanded(e => !e)}
            className="text-[#00d4ff] flex items-center gap-1"
            style={{ fontSize: '0.65rem', fontFamily: 'Space Mono, monospace' }}
          >
            {expanded ? <><ChevronUp size={10} /> less</> : <><ChevronDown size={10} /> +{sources.length - 3} more</>}
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {shown.map((s, i) => (
          <div
            key={i}
            className="flex items-center gap-1.5 px-2 py-1"
            style={{
              background: 'rgba(0,212,255,0.06)',
              border: '1px solid rgba(0,212,255,0.3)',
              fontFamily: 'IBM Plex Mono, monospace',
            }}
          >
            <FileText size={9} color="#00d4ff" />
            <span style={{ fontSize: '0.65rem', color: '#a0b4d0' }}>
              {truncate(s.filename || s.chunk_id, 30)}
            </span>
            {s.score != null && (
              <span
                className="ml-1 px-1"
                style={{
                  fontSize: '0.6rem',
                  background: 'rgba(0,212,255,0.15)',
                  color: '#00d4ff',
                }}
              >
                {formatScore(s.score)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
