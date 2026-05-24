import React from 'react';

interface Props { size?: number; color?: string; label?: string; }

export default function Spinner({ size = 20, color = '#00d4ff', label }: Props) {
  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="animate-spin">
        <circle cx="12" cy="12" r="9" stroke={color + '33'} strokeWidth="3" />
        <path d="M12 3a9 9 0 0 1 9 9" stroke={color} strokeWidth="3" strokeLinecap="square" />
      </svg>
      {label && (
        <span className="text-xs" style={{ fontFamily: 'IBM Plex Mono, monospace', color, fontSize: '0.7rem' }}>
          {label}
        </span>
      )}
    </div>
  );
}
