import React, { useState } from 'react';

interface ElasticsearchProps {
  onToast: (type: 'success' | 'error' | 'info', message: string) => void;
}

export default function Elasticsearch({ onToast }: ElasticsearchProps) {
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const handleUpload = async () => {
    if (!file) {
      onToast('error', 'Please select a file');
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const res = await fetch('http://localhost:9000/api/elasticsearch/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Upload failed');
      
      const data = await res.json();
      onToast('success', `Indexed ${data.count} documents to Elasticsearch`);
      setFile(null);
    } catch (err: any) {
      onToast('error', err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Elasticsearch Data Upload</h1>
      
      <div className="bg-white rounded-lg shadow p-6">
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Select File (text, one document per line)</label>
          <input
            type="file"
            accept=".txt,.csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="block w-full text-sm border rounded px-3 py-2"
          />
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {uploading ? 'Uploading...' : 'Upload to Elasticsearch'}
        </button>
      </div>
    </div>
  );
}
