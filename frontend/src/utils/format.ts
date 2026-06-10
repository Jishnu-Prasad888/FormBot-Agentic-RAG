export const formatDate = (iso: string): string => {
  const d = new Date(iso);
  return d.toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' });
};

export const formatTime = (iso: string): string => {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit' });
};

export const formatDateTime = (iso: string): string =>
  `${formatDate(iso)} ${formatTime(iso)}`;

export const formatLatency = (ms: number): string => {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
};

export const formatScore = (score: number): string => {
  const s = isNaN(score) ? 0 : score;
  return `${(s * 100).toFixed(1)}%`;
};

export const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
};

export const truncate = (str: string, len: number): string =>
  str.length > len ? str.slice(0, len) + '…' : str;

export const docTypeColor = (type: string): string => {
  switch (type) {
    case 'pdf':      return '#ff4d6d';
    case 'markdown': return '#00d4ff';
    case 'csv':      return '#00ff9f';
    case 'json':     return '#ffe600';
    default:         return '#6b82b0';
  }
};

export const docTypeBg = (type: string): string => {
  switch (type) {
    case 'pdf':      return 'border-coral text-coral bg-coral/10';
    case 'markdown': return 'border-[#00d4ff] text-[#00d4ff] bg-[#00d4ff]/10';
    case 'csv':      return 'border-[#00ff9f] text-[#00ff9f] bg-[#00ff9f]/10';
    case 'json':     return 'border-[#ffe600] text-[#ffe600] bg-[#ffe600]/10';
    default:         return 'border-[#6b82b0] text-[#6b82b0] bg-[#6b82b0]/10';
  }
};

export const strategyColor = (s: string): string => {
  if (s.includes('hybrid'))  return '#00d4ff';
  if (s.includes('table'))   return '#00ff9f';
  if (s.includes('pdf') || s.includes('hierarchical')) return '#ff4d6d';
  if (s.includes('markdown') || s.includes('structure')) return '#a78bfa';
  if (s.includes('vector'))  return '#38bdf8';
  if (s.includes('bm25'))    return '#ffe600';
  return '#6b82b0';
};
