import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  CameraOff,
  MonitorUp,
  Share,
  Mic,
  VolumeX,
  Volume2,
  Sparkles,
  Loader2,
  Upload,
  MessageCircle,
  Languages,
  Clock3,
  RefreshCw,
  Eye,
  PictureInPicture2,
} from 'lucide-react';
import { streamLiveAsk, pushLiveFrame, liveTts, liveTranscribe } from '../api/client';
import type { Toast } from '../types';

interface Props {
  onToast: (type: Toast['type'], msg: string) => void;
}

const FRAME_INTERVAL_MS = 200; // ~5 FPS target
const JPEG_QUALITY = 0.65;
const TARGET_WIDTH = 1280; // ~720p when 16:9

export default function LiveMeet({ onToast }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const detectCanvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const captureTimer = useRef<NodeJS.Timeout | null>(null);
  const recordingRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const backoffUntilRef = useRef<number>(0);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [sourceType, setSourceType] = useState<'camera' | 'screen'>('camera');
  const [ocrOn, setOcrOn] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [inFlightFrame, setInFlightFrame] = useState(false);
  const [paperDetected, setPaperDetected] = useState(false);
  const [fields, setFields] = useState<any[]>([]);
  const [layoutMd, setLayoutMd] = useState('');
  const [diff, setDiff] = useState<any[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  const [question, setQuestion] = useState('');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [answer, setAnswer] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [aiMuted, setAiMuted] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [cams, setCams] = useState<MediaDeviceInfo[]>([]);
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [selectedCam, setSelectedCam] = useState<string>('');
  const [selectedMic, setSelectedMic] = useState<string>('');
  const [pipSupported, setPipSupported] = useState<boolean>(document.pictureInPictureEnabled);
  const [micMuted, setMicMuted] = useState<boolean>(false);
  const [recording, setRecording] = useState(false);
  const [useFormContext, setUseFormContext] = useState(true);
  const [manualContext, setManualContext] = useState('');
  const sessionKey = 'live_session_id';
  const convKey = 'live_conversation_id';

  // ─── Media handling ────────────────────────────────────────────────────────
  const enterPiP = async () => {
    if (!pipSupported || !videoRef.current) return;
    if (document.pictureInPictureElement) return;
    try {
      await videoRef.current.requestPictureInPicture();
    } catch (e) {
      // ignore
    }
  };

  const exitPiP = async () => {
    try {
      if (document.pictureInPictureElement) await document.exitPictureInPicture();
    } catch {
      // ignore
    }
  };

  const startCamera = async () => {
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, deviceId: selectedCam ? { exact: selectedCam } : undefined },
        audio: selectedMic ? { deviceId: { exact: selectedMic } } : true,
      });
      if (videoRef.current) videoRef.current.srcObject = media;
      setStream(media);
      setSourceType('camera');
    } catch (e) {
      onToast('error', 'Failed to start camera');
    }
  };

  const startScreen = async () => {
    try {
      const display = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      // add mic track if available
      let finalStream = display;
      if (selectedMic) {
        try {
          const micStream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: selectedMic } } });
          micStream.getAudioTracks().forEach(t => finalStream.addTrack(t));
        } catch {
          // ignore mic failure
        }
      }
      if (videoRef.current) videoRef.current.srcObject = finalStream;
      setStream(finalStream);
      setSourceType('screen');
      // enter PiP to survive tab switches
      enterPiP();
    } catch (e) {
      onToast('error', 'Failed to start screen share');
    }
  };

  const stopMedia = () => {
    stream?.getTracks().forEach(t => t.stop());
    setStream(null);
    setPaperDetected(false);
    setRecording(false);
    exitPiP();
  };

  const detectPaper = (video: HTMLVideoElement) => {
    const dc = detectCanvasRef.current;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return false;
    const w = 320;
    const h = Math.max(1, Math.round((w / vw) * vh));
    dc.width = w;
    dc.height = h;
    const ctx = dc.getContext('2d');
    if (!ctx) return false;
    ctx.drawImage(video, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;
    let bright = 0;
    let total = w * h;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      if (lum > 200) bright++;
    }
    const fracBright = bright / total;
    return fracBright > 0.35; // crude paper-ish heuristic
  };

  const captureFrame = async () => {
    if (Date.now() < backoffUntilRef.current) return;
    if (!ocrOn || inFlightFrame) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;

    // Paper gate only for camera source
    if (sourceType === 'camera') {
      const hasPaper = detectPaper(video);
      setPaperDetected(hasPaper);
      if (!hasPaper) return; // skip push if no paper-like region
    } else {
      setPaperDetected(false);
    }
    const canvas = canvasRef.current;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;
    const targetW = Math.min(TARGET_WIDTH, vw);
    const targetH = Math.round((targetW / vw) * vh);
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, targetW, targetH);
    setInFlightFrame(true);
    canvas.toBlob(async (blob) => {
      if (!blob) { setInFlightFrame(false); return; }
      try {
        const resp = await pushLiveFrame(blob, sessionId || undefined);
        if (resp.session_id) setSessionId(resp.session_id);
        if (resp.status === 'updated') {
          setFields(resp.fields || []);
          setLayoutMd(resp.layout_markdown || '');
          setDiff(resp.diff || []);
          setUpdatedAt(resp.updated_at || null);
        }
      } catch (e: any) {
        const msg = e?.response?.data?.detail || e?.message || 'Frame upload failed';
        if (msg.toLowerCase().includes('too many frames') || e?.response?.status === 429) {
          backoffUntilRef.current = Date.now() + 1000;
        } else {
          onToast('warning', msg);
        }
      } finally {
        setInFlightFrame(false);
      }
    }, 'image/jpeg', JPEG_QUALITY);
  };

  useEffect(() => {
    const ensureDevices = async () => {
      try {
        await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        setCams(devices.filter(d => d.kind === 'videoinput'));
        setMics(devices.filter(d => d.kind === 'audioinput'));
        if (!selectedCam) setSelectedCam(devices.find(d => d.kind === 'videoinput')?.deviceId || '');
        if (!selectedMic) setSelectedMic(devices.find(d => d.kind === 'audioinput')?.deviceId || '');
      } catch {
        // ignore
      }
    };
    // restore session/conversation if available
    const storedSession = localStorage.getItem(sessionKey);
    if (storedSession) setSessionId(storedSession);
    const storedConv = localStorage.getItem(convKey);
    if (storedConv) setConversationId(storedConv);
    ensureDevices();
    navigator.mediaDevices.addEventListener('devicechange', ensureDevices);
    startCamera();
    return () => {
      stopMedia();
      if (captureTimer.current) clearInterval(captureTimer.current);
      navigator.mediaDevices.removeEventListener('devicechange', ensureDevices);
    };
  }, []);

  useEffect(() => {
    if (captureTimer.current) clearInterval(captureTimer.current);
    if (ocrOn) {
      captureTimer.current = setInterval(captureFrame, FRAME_INTERVAL_MS);
    }
    return () => {
      if (captureTimer.current) clearInterval(captureTimer.current);
    };
  }, [ocrOn, sessionId, stream]);

  // ─── Mic to text (press to speak) ──────────────────────────────────────────
  const startRecording = () => {
    if (!stream) return onToast('warning', 'Camera/mic not started');
    const audioTrack = stream.getAudioTracks()[0];
    if (!audioTrack) return onToast('warning', 'No mic track');
    try {
      const mr = new MediaRecorder(new MediaStream([audioTrack]), { mimeType: 'audio/webm' });
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        try {
          const res = await liveTranscribe(blob);
          setQuestion(res.text || '');
          onToast('info', 'Transcribed');
        } catch (e: any) {
          onToast('error', e?.response?.data?.detail || 'Transcription failed');
        }
      };
      recordingRef.current = mr;
      mr.start();
      setRecording(true);
    } catch {
      onToast('error', 'Recording failed');
    }
  };

  const stopRecording = () => {
    recordingRef.current?.stop();
    recordingRef.current = null;
    setRecording(false);
  };

  // mic mute toggle
  const toggleMic = () => {
    if (!stream) return;
    stream.getAudioTracks().forEach(t => t.enabled = !micMuted);
    setMicMuted(m => !m);
  };

  // PiP on tab switch during screenshare
  useEffect(() => {
    const handler = () => {
      if (document.visibilityState === 'hidden' && sourceType === 'screen' && pipSupported && videoRef.current && !document.pictureInPictureElement) {
        enterPiP();
      }
    };
    document.addEventListener('visibilitychange', handler);
    return () => document.removeEventListener('visibilitychange', handler);
  }, [sourceType, pipSupported]);

  useEffect(() => {
    const onLeavePiP = () => {
      // attempt to re-enter if still screen sharing
      if (sourceType === 'screen') enterPiP();
    };
    document.addEventListener('leavepictureinpicture', onLeavePiP as any);
    return () => document.removeEventListener('leavepictureinpicture', onLeavePiP as any);
  }, [sourceType, pipSupported]);

  useEffect(() => {
    if (sessionId) localStorage.setItem(sessionKey, sessionId);
  }, [sessionId]);

  useEffect(() => {
    if (conversationId) localStorage.setItem(convKey, conversationId);
  }, [conversationId]);

  // ─── Ask & stream answer ───────────────────────────────────────────────────
  const handleAsk = async () => {
    if (!question.trim()) return;
    setStreaming(true);
    setAnswer('');
    try {
      await streamLiveAsk({
        question,
        conversation_id: conversationId || undefined,
        session_id: sessionId || undefined,
        target_language: targetLanguage,
        use_form_context: useFormContext,
        manual_context: manualContext.trim() || undefined,
      }, {
        onToken: (t) => setAnswer(a => a + t),
        onDone: () => setStreaming(false),
        onConversation: (id) => setConversationId(id),
        onError: (e) => { setStreaming(false); onToast('error', e?.message || 'Live ask failed'); },
      });
    } catch (e: any) {
      setStreaming(false);
      onToast('error', e?.message || 'Live ask failed');
    }
  };

  useEffect(() => {
    if (!aiMuted && !streaming && answer.trim()) {
      // fire-and-forget TTS
      (async () => {
        try {
          const audioUrl = await liveTts(answer, undefined);
          const audio = new Audio(audioUrl);
          audio.play();
        } catch (e) {
          onToast('warning', 'TTS failed');
        }
      })();
    }
  }, [streaming, answer, aiMuted]);

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="p-4 md:p-6 space-y-4">
      <div className="flex items-center gap-2">
        <Sparkles size={14} color="#00d4ff" />
        <span className="text-xs uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#00d4ff' }}>
          Live Meet
        </span>
        {updatedAt && (
          <span className="text-[11px] text-[#6b82b0] flex items-center gap-1" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
            <Clock3 size={11} /> updated {updatedAt}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Stage */}
        <div className="lg:col-span-2 brutal-card p-3 space-y-3">
          <div className="relative bg-black rounded-md overflow-hidden" style={{ minHeight: 320 }}>
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-contain" />
            <div className="absolute bottom-2 left-2 flex flex-wrap gap-2 bg-black/50 px-2 py-1 rounded">
              <button
                className="text-white flex items-center gap-1 text-xs"
                onClick={() => { stream ? stopMedia() : startCamera(); }}
              >
                {stream && sourceType === 'camera' ? <CameraOff size={14} /> : <Camera size={14} />} {stream && sourceType === 'camera' ? 'Stop' : 'Camera'}
              </button>
              <button
                className="text-white flex items-center gap-1 text-xs"
                onClick={() => { stream ? stopMedia() : startScreen(); }}
              >
                {stream && sourceType === 'screen' ? <CameraOff size={14} /> : <Share size={14} />} {stream && sourceType === 'screen' ? 'Stop' : 'Share screen'}
              </button>
              {pipSupported && (
                <button
                  className="text-white flex items-center gap-1 text-xs"
                  onClick={() => { if (videoRef.current) videoRef.current.requestPictureInPicture().catch(() => {}); }}
                >
                  <PictureInPicture2 size={14} /> PiP
                </button>
              )}
              <button
                className="text-white flex items-center gap-1 text-xs"
                onClick={() => setOcrOn(v => !v)}
              >
                <RefreshCw size={14} /> {ocrOn ? 'OCR On' : 'OCR Off'}
              </button>
              <button
                className="text-white flex items-center gap-1 text-xs"
                onClick={() => { recording ? stopRecording() : startRecording(); }}
              >
                <Mic size={14} /> {recording ? 'Stop speaking' : 'Tap to speak'}
              </button>
              <button
                className="text-white flex items-center gap-1 text-xs"
                onClick={toggleMic}
              >
                {micMuted ? <VolumeX size={14} /> : <Volume2 size={14} />} {micMuted ? 'Mic muted' : 'Mic live'}
              </button>
            </div>
            {sourceType === 'camera' && (
              <div className="absolute top-2 left-2 flex items-center gap-1 text-[11px] px-2 py-1 rounded" style={{ background: 'rgba(0,0,0,0.55)', color: paperDetected ? '#00ff9f' : '#ff4d6d', fontFamily: 'IBM Plex Mono, monospace' }}>
                <Eye size={12} /> {paperDetected ? 'Paper detected' : 'Looking for paper...'}
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-2 text-xs" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
            <label className="flex items-center gap-1 text-[#6b82b0]">
              Cam
              <select
                className="bg-transparent border border-[#1e2d54] px-2 py-1"
                value={selectedCam}
                onChange={e => setSelectedCam(e.target.value)}
              >
                {cams.map(c => <option key={c.deviceId} value={c.deviceId}>{c.label || 'Camera'}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-1 text-[#6b82b0]">
              Mic
              <select
                className="bg-transparent border border-[#1e2d54] px-2 py-1"
                value={selectedMic}
                onChange={e => setSelectedMic(e.target.value)}
              >
                {mics.map(m => <option key={m.deviceId} value={m.deviceId}>{m.label || 'Mic'}</option>)}
              </select>
            </label>
          </div>

          <div className="flex gap-2 flex-wrap items-center">
            <div className="flex items-center gap-1 text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
              <MonitorUp size={12} /> {sourceType === 'camera' ? 'Camera' : 'Screen'} · 5 FPS @ 720p
            </div>
            {inFlightFrame && <Loader2 size={14} className="animate-spin text-[#00d4ff]" />}
            <div className="flex items-center gap-1 text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
              <Languages size={12} />
              <select
                className="bg-transparent border border-[#1e2d54] px-2 py-1 text-xs"
                value={targetLanguage}
                onChange={e => setTargetLanguage(e.target.value)}
              >
                <option value="en">English</option>
                <option value="kn">Kannada</option>
              </select>
            </div>
            <button
              className="flex items-center gap-1 text-xs"
              onClick={() => setAiMuted(m => !m)}
              style={{ color: aiMuted ? '#ff4d6d' : '#00ff9f' }}
            >
              {aiMuted ? <VolumeX size={14} /> : <Volume2 size={14} />} {aiMuted ? 'AI muted' : 'AI voice'}
            </button>
            <label className="flex items-center gap-1 text-xs text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
              <input type="checkbox" checked={useFormContext} onChange={e => setUseFormContext(e.target.checked)} /> Use form context
            </label>
          </div>

          <div className="flex gap-2">
            <input
              className="input-brutal flex-1 px-3 py-2 text-sm"
              placeholder="Ask a question..."
              value={question}
              onChange={e => setQuestion(e.target.value)}
            />
            <button
              className="btn-brutal px-3 py-2 text-sm flex items-center gap-1"
              onClick={handleAsk}
              disabled={streaming}
            >
              {streaming ? <Loader2 size={16} className="animate-spin" /> : <MessageCircle size={16} />}
              {streaming ? 'Streaming...' : 'Ask'}
            </button>
          </div>

          <div className="p-3 border border-[#1e2d54] bg-[#0f1629] min-h-[100px]">
            <div className="text-xs uppercase tracking-widest text-[#4a5a8e] mb-2" style={{ fontFamily: 'Space Mono, monospace' }}>
              AI Answer
            </div>
            <div className="text-sm text-[#e2e8f0] whitespace-pre-wrap" style={{ fontFamily: 'IBM Plex Sans, sans-serif' }}>
              {answer || (streaming ? '...' : 'No answer yet')}
            </div>
          </div>
        </div>

        {/* Form context */}
        <div className="brutal-card p-3 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Upload size={14} color="#00d4ff" />
              <span className="text-xs uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#00d4ff' }}>
                Form context
              </span>
            </div>
            {diff.length > 0 && (
              <span className="text-[11px] text-[#00ff9f]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                {diff.length} change(s)
              </span>
            )}
            <div className="flex gap-2">
              <button
                className="text-[11px] px-2 py-1 border border-[#1e2d54] text-[#6b82b0]"
                onClick={async () => {
                  if (!sessionId) return;
                  try { await (await import('../api/client')).then(({ clearLiveFrame }) => clearLiveFrame(sessionId));
                    setFields([]); setLayoutMd(''); setDiff([]); setUpdatedAt(null);
                  } catch { onToast('warning', 'Clear failed'); }
                }}
              >
                Clear form context
              </button>
            </div>
          </div>

          {fields.length === 0 ? (
            <div className="text-xs text-[#4a5a8e]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
              Waiting for first frame update...
            </div>
          ) : (
            <div className="space-y-2 max-h-48 overflow-auto">
              {fields.map((f, i) => (
                <div key={i} className="p-2 border border-[#1e2d54] bg-[#0f1629]">
                  <div className="text-xs" style={{ color: '#00d4ff', fontFamily: 'Space Mono, monospace' }}>
                    {f.field || f.name || 'Field'}
                  </div>
                  <div className="text-xs text-[#e2e8f0]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                    {f.value || ''}
                  </div>
                  {typeof f.confidence === 'number' && (
                    <div className="text-[11px] text-[#6b82b0]">conf: {(f.confidence * 100).toFixed(0)}%</div>
                  )}
                </div>
              ))}
            </div>
          )}

          {layoutMd && (
            <div className="mt-2 p-2 border border-[#1e2d54] bg-[#0a0e1a] max-h-48 overflow-auto">
              <pre className="text-[11px] text-[#cbd5e1] whitespace-pre-wrap" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                {layoutMd}
              </pre>
            </div>
          )}

          {diff.length > 0 && (
            <div className="mt-2 p-2 border border-[#1e2d54] bg-[#0a0e1a] max-h-32 overflow-auto">
              <div className="text-[11px] text-[#ff8c00] mb-1" style={{ fontFamily: 'Space Mono, monospace' }}>Recent changes</div>
              {diff.map((d, i) => (
                <div key={i} className="text-[11px] text-[#cbd5e1]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                  {d.change}: {d.field} {d.value || ''}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Manual context */}
        <div className="brutal-card p-3 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#a78bfa' }}>
              Manual context
            </span>
          </div>
          <textarea
            className="input-brutal w-full text-xs p-2 h-24"
            placeholder="Add any extra context you want the AI to use"
            value={manualContext}
            onChange={e => setManualContext(e.target.value)}
            style={{ fontFamily: 'IBM Plex Mono, monospace' }}
          />
          <div className="flex gap-2 text-[11px] text-[#6b82b0]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
            <span>Included in the next question if non-empty.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
