import React, { useState, useEffect, useRef } from 'react';
import {
  Send, Plus, Trash2, MessageSquare, User, Bot,
  StopCircle, ChevronLeft, ChevronRight
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import {
  listConversations, getConversation,
  deleteConversation, streamChat
} from '../api/client';
import Spinner from '../components/Spinner';
import SourceChips from '../components/SourceChip';
import type { Conversation, Message } from '../types';
import { formatDateTime, truncate } from '../utils/format';

interface Props { onToast: (type: any, msg: string) => void; }

export default function Chat({ onToast }: Props) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | undefined>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [streamBuffer, setStreamBuffer] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loadingConv, setLoadingConv] = useState(false);
  const stopRef = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const loadConvs = async () => {
    try {
      const res = await listConversations();
      setConversations(res.conversations || []);
    } catch { onToast('error', 'Failed to load conversations'); }
  };

  useEffect(() => { loadConvs(); }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamBuffer]);

  const selectConv = async (id: string) => {
    setLoadingConv(true);
    setActiveConvId(id);
    setMessages([]);
    try {
      const res = await getConversation(id);
      setMessages(res.messages || []);
    } catch { onToast('error', 'Failed to load conversation'); }
    finally { setLoadingConv(false); }
  };

  const newConversation = () => {
    setActiveConvId(undefined);
    setMessages([]);
    setInput('');
  };

  const deleteConv = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteConversation(id);
      setConversations(c => c.filter(x => x.id !== id));
      if (activeConvId === id) newConversation();
      onToast('success', 'Conversation deleted');
    } catch { onToast('error', 'Delete failed'); }
  };

  const sendMessage = async () => {
    if (!input.trim() || streaming) return;
    const userMsg = input.trim();
    setInput('');

    // Optimistically add user message
    const tempUser: Message = {
      id: 'temp-user',
      conversation_id: activeConvId || '',
      role: 'user',
      content: userMsg,
      sources: [],
      created_at: new Date().toISOString(),
    };
    setMessages(m => [...m, tempUser]);
    setStreaming(true);
    setStreamBuffer('');
    stopRef.current = false;

    let fullText = '';
    let newConvId = activeConvId;

    await streamChat(
      userMsg,
      activeConvId,
      (token) => {
        if (stopRef.current) return;
        fullText += token;
        setStreamBuffer(fullText);
      },
      async () => {
        setStreaming(false);
        setStreamBuffer('');

        // Reload conversation to get proper IDs and sources
        try {
          const convs = await listConversations();
          setConversations(convs.conversations || []);
          if (!activeConvId && convs.conversations?.length > 0) {
            const latest = convs.conversations[0];
            newConvId = latest.id;
            setActiveConvId(latest.id);
            const full = await getConversation(latest.id);
            setMessages(full.messages || []);
          } else if (activeConvId) {
            const full = await getConversation(activeConvId);
            setMessages(full.messages || []);
          }
        } catch {
          // Fallback: add assistant message manually
          setMessages(m => m.map(x => x.id === 'temp-user' ? x : x).concat({
            id: 'temp-asst',
            conversation_id: activeConvId || '',
            role: 'assistant',
            content: fullText,
            sources: [],
            created_at: new Date().toISOString(),
          }));
        }
      },
      (err) => {
        setStreaming(false);
        setStreamBuffer('');
        onToast('error', 'Stream error — is Ollama running?');
        setMessages(m => m.filter(x => x.id !== 'temp-user'));
      }
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  return (
    <div className="flex h-full overflow-hidden" style={{ background: '#0a0e1a' }}>
      {/* Conversations sidebar */}
      <aside
        className={`flex flex-col flex-shrink-0 border-r-2 transition-all duration-200 ${sidebarOpen ? 'w-56 md:w-64' : 'w-0 overflow-hidden'}`}
        style={{ borderColor: '#2a3a6e', background: '#0f1629' }}
      >
        <div className="flex items-center justify-between px-3 py-3 border-b-2" style={{ borderColor: '#2a3a6e' }}>
          <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#6b82b0' }}>
            Chats
          </span>
          <button className="btn-brutal px-2 py-1 text-xs flex items-center gap-1" onClick={newConversation}>
            <Plus size={10} /> New
          </button>
        </div>
        <div className="flex-1 overflow-y-auto py-1">
          {conversations.length === 0 ? (
            <div className="px-3 py-4 text-xs text-[#2a3a6e]" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
              No conversations
            </div>
          ) : (
            conversations.map(c => (
              <button
                key={c.id}
                onClick={() => selectConv(c.id)}
                className={`w-full text-left px-3 py-2.5 flex items-start justify-between gap-2 group border-l-2 transition-all ${activeConvId === c.id ? 'border-[#00d4ff] bg-[#1a2444]' : 'border-transparent hover:bg-[#1a2444]/50 hover:border-[#2a3a6e]'}`}
              >
                <div className="min-w-0">
                  <div className="text-xs truncate" style={{ fontFamily: 'IBM Plex Mono, monospace', color: activeConvId === c.id ? '#e2e8f0' : '#8a9abb', fontSize: '0.7rem' }}>
                    {truncate(c.title || 'Untitled', 28)}
                  </div>
                  <div className="text-xs text-[#4a5a8e] mt-0.5" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.6rem' }}>
                    {formatDateTime(c.updated_at)}
                  </div>
                </div>
                <button
                  onClick={e => deleteConv(c.id, e)}
                  className="opacity-0 group-hover:opacity-100 text-[#ff4d6d] flex-shrink-0 mt-0.5"
                >
                  <Trash2 size={10} />
                </button>
              </button>
            ))
          )}
        </div>
      </aside>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat header */}
        <div className="flex items-center gap-2 px-3 py-2.5 border-b-2 flex-shrink-0" style={{ borderColor: '#2a3a6e', background: '#0f1629' }}>
          <button
            onClick={() => setSidebarOpen(o => !o)}
            className="text-[#6b82b0] hover:text-[#00d4ff] transition-colors"
          >
            {sidebarOpen ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
          </button>
          <MessageSquare size={13} color="#00d4ff" />
          <span className="text-xs font-bold uppercase tracking-widest" style={{ fontFamily: 'Space Mono, monospace', color: '#00d4ff' }}>
            {activeConvId ? `CONV_${activeConvId.slice(0, 8).toUpperCase()}` : 'NEW_CONVERSATION'}
          </span>
          {streaming && (
            <div className="ml-auto flex items-center gap-2">
              <span className="text-xs text-[#ffe600]" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.65rem' }}>GENERATING</span>
              <button
                className="text-[#ff4d6d] hover:text-[#ff4d6d]/70"
                onClick={() => { stopRef.current = true; setStreaming(false); setStreamBuffer(''); }}
              >
                <StopCircle size={12} />
              </button>
            </div>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {loadingConv ? (
            <div className="flex justify-center py-12">
              <Spinner size={24} label="LOADING_MESSAGES..." />
            </div>
          ) : messages.length === 0 && !streamBuffer ? (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-4">
              <div className="p-4 border-2 border-[#2a3a6e]" style={{ background: '#0f1629' }}>
                <Bot size={28} color="#2a3a6e" />
              </div>
              <div>
                <div className="text-sm font-bold" style={{ fontFamily: 'Space Mono, monospace', color: '#4a5a8e' }}>
                  RAG_CHAT_READY
                </div>
                <div className="text-xs text-[#2a3a6e] mt-1" style={{ fontFamily: 'IBM Plex Mono, monospace' }}>
                  Ask a question about your indexed documents
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-sm">
                {[
                  'What documents are indexed?',
                  'Summarize the main topics',
                  'Find information about eligibility',
                  'What schemes are available?',
                ].map(q => (
                  <button
                    key={q}
                    className="p-2 text-left text-xs border border-[#2a3a6e] hover:border-[#00d4ff] hover:bg-[#1a2444] transition-all"
                    style={{ fontFamily: 'IBM Plex Mono, monospace', color: '#6b82b0', fontSize: '0.65rem' }}
                    onClick={() => { setInput(q); textareaRef.current?.focus(); }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <div key={msg.id || i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'assistant' && (
                    <div className="flex-shrink-0 w-7 h-7 flex items-center justify-center border-2 border-[#00ff9f]" style={{ background: '#00ff9f15' }}>
                      <Bot size={12} color="#00ff9f" />
                    </div>
                  )}
                  <div className={`max-w-[80%] ${msg.role === 'user' ? 'msg-user' : 'msg-assistant'} p-3`}>
                    {msg.role === 'user' ? (
                      <p className="text-sm" style={{ fontFamily: 'IBM Plex Sans, sans-serif', color: '#e2e8f0' }}>
                        {msg.content}
                      </p>
                    ) : (
                      <div className="prose prose-invert prose-sm max-w-none text-sm" style={{ fontFamily: 'IBM Plex Sans, sans-serif', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.6' }}>
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    )}
                    {msg.role === 'assistant' && msg.sources?.length > 0 && (
                      <SourceChips sources={msg.sources} />
                    )}
                    <div className="text-xs text-[#2a3a6e] mt-1.5" style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '0.6rem' }}>
                      {formatDateTime(msg.created_at)}
                    </div>
                  </div>
                  {msg.role === 'user' && (
                    <div className="flex-shrink-0 w-7 h-7 flex items-center justify-center border-2 border-[#00d4ff]" style={{ background: '#00d4ff15' }}>
                      <User size={12} color="#00d4ff" />
                    </div>
                  )}
                </div>
              ))}
              {/* Streaming message */}
              {streamBuffer && (
                <div className="flex gap-3 justify-start">
                  <div className="flex-shrink-0 w-7 h-7 flex items-center justify-center border-2 border-[#00ff9f]" style={{ background: '#00ff9f15' }}>
                    <Bot size={12} color="#00ff9f" />
                  </div>
                  <div className="max-w-[80%] msg-assistant p-3">
                    <div className="prose prose-invert prose-sm max-w-none text-sm" style={{ fontFamily: 'IBM Plex Sans, sans-serif', color: '#cbd5e1', fontSize: '0.875rem', lineHeight: '1.6' }}>
                      <ReactMarkdown>{streamBuffer}</ReactMarkdown>
                    </div>
                    <span className="cursor" />
                  </div>
                </div>
              )}
            </>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="flex-shrink-0 p-3 border-t-2" style={{ borderColor: '#2a3a6e', background: '#0f1629' }}>
          <div className="flex gap-2 items-end">
            <textarea
              ref={textareaRef}
              rows={1}
              className="input-brutal flex-1 py-2.5 px-3 text-sm resize-none"
              placeholder="Ask about your documents... (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={streaming}
              style={{ fontFamily: 'IBM Plex Sans, sans-serif', minHeight: 42, maxHeight: 120 }}
            />
            <button
              className="btn-brutal px-3 flex-shrink-0 flex items-center justify-center"
              style={{ height: 42, minWidth: 42 }}
              onClick={sendMessage}
              disabled={streaming || !input.trim()}
            >
              {streaming ? <Spinner size={14} /> : <Send size={14} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
