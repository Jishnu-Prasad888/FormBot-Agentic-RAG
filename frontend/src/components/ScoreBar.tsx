import React from 'react';
import { formatScore } from '../utils/format';

interface Props {
  score: number;
  label?: string;
  color?: string;
  showValue?: boolean;
}

export default function ScoreBar({ score, label, color = '#00d4ff', showValue = true }: Props) {
  const s = isNaN(score) ? 0 : score;
  const pct = Math.min(Math.max(s, 0), 1) * 100;
  return (
    <div className="w-full">
      {(label || showValue) && (
        <div className="flex justify-between mb-1">
          {label && (
            <span className="text-xs text-[#6b82b0] uppercase tracking-wider" style={{ fontFamily: 'Space Mono, monospace', fontSize: '0.65rem' }}>
              {label}
            </span>
          )}
          {showValue && (
            <span className="text-xs font-bold" style={{ fontFamily: 'IBM Plex Mono, monospace', color, fontSize: '0.7rem' }}>
              {formatScore(s)}
            </span>
          )}
        </div>
      )}
      <div className="score-bar">
        <div className="score-fill" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})` }} />
      </div>
    </div>
  );
}
