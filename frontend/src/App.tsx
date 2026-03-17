import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Send, Zap, Search, FolderOpen, Brain, Terminal,
  Activity, ChevronDown, X, Plus, Loader2, Mic,
  Database, Globe, FileText, Code, MessageSquare
} from "lucide-react";

// ── Types ───────────────────────────────────────────────────────────
type Role = "user" | "assistant" | "system";
interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: Date;
  sources?: Source[];
}
interface Source {
  source_type: string;
  platform: string;
  content: string;
  score: number;
  date: string;
}
interface Stats {
  total_documents: number;
  collections: Record<string, { count: number }>;
  llm: string;
}

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = API.replace("http", "ws");

// ── Helpers ─────────────────────────────────────────────────────────
const uid = () => Math.random().toString(36).slice(2);
const SOURCE_ICONS: Record<string, string> = {
  chatgpt: "🤖", cursor: "✏️", claude: "🟠", gemini: "♊",
  local_file: "📄", user_note: "💡", code_files: "💻", default: "📌"
};

// ── Main App ─────────────────────────────────────────────────────────
export default function Jarvis() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: `# JARVIS Online ⚡\n\nI'm fully initialized with your personal knowledge base. I have access to:\n- Your chat histories (Cursor, ChatGPT, Gemini, Claude)\n- Your code projects and files\n- Your research and documents\n\nWhat can I help you with, Manvith?`,
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId] = useState(() => `jarvis-${uid()}`);
  const [stats, setStats] = useState<Stats | null>(null);
  const [activeTab, setActiveTab] = useState<"chat" | "search" | "files" | "memory">("chat");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Source[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [filePath, setFilePath] = useState("~");
  const [fileItems, setFileItems] = useState<any[]>([]);
  const [ingestStatus, setIngestStatus] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ── WebSocket connection ─────────────────────────────────────────
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(`${WS_URL}/ws/${sessionId}`);
      ws.onopen = () => console.log("JARVIS connected");
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === "chunk") {
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last?.role === "assistant" && last.id === "streaming") {
              return [...prev.slice(0, -1), { ...last, content: last.content + data.content }];
            }
            return prev;
          });
        } else if (data.type === "done") {
          setIsStreaming(false);
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last?.id === "streaming") {
              return [...prev.slice(0, -1), { ...last, id: uid() }];
            }
            return prev;
          });
        } else if (data.type === "error") {
          setIsStreaming(false);
          setMessages(prev => [...prev, {
            id: uid(), role: "assistant",
            content: `❌ Error: ${data.content}`, timestamp: new Date()
          }]);
        }
      };
      ws.onclose = () => setTimeout(connect, 3000);
      wsRef.current = ws;
    };
    connect();
    return () => wsRef.current?.close();
  }, [sessionId]);

  // ── Load stats ────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(setStats).catch(() => {});
    browsePath("~");
  }, []);

  // ── Auto scroll ───────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Send message ─────────────────────────────────────────────────
  const sendMessage = useCallback(() => {
    if (!input.trim() || isStreaming) return;
    const msg: Message = { id: uid(), role: "user", content: input.trim(), timestamp: new Date() };

    setMessages(prev => [...prev, msg, {
      id: "streaming", role: "assistant", content: "", timestamp: new Date()
    }]);
    setInput("");
    setIsStreaming(true);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ message: msg.content }));
    }
  }, [input, isStreaming]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  // ── Search ───────────────────────────────────────────────────────
  const runSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const res = await fetch(`${API}/search`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, top_k: 10 })
      });
      const data = await res.json();
      setSearchResults(data.results || []);
    } finally {
      setIsSearching(false);
    }
  };

  // ── Browse files ─────────────────────────────────────────────────
  const browsePath = async (path: string) => {
    try {
      const res = await fetch(`${API}/files/browse?path=${encodeURIComponent(path)}`);
      if (!res.ok) return;
      const data = await res.json();
      setFilePath(data.path);
      setFileItems(data.items || []);
    } catch {}
  };

  // ── Trigger ingestion ─────────────────────────────────────────────
  const triggerAutoIngest = async () => {
    setIngestStatus("Starting auto-ingestion...");
    const res = await fetch(`${API}/ingest/auto`);
    const data = await res.json();
    setIngestStatus(data.message || "Ingestion started in background");
    setTimeout(() => {
      fetch(`${API}/health`).then(r => r.json()).then(setStats).catch(() => {});
      setIngestStatus("");
    }, 3000);
  };

  // ── Render ────────────────────────────────────────────────────────
  return (
    <div className="h-screen bg-[#0a0a0f] text-gray-100 flex flex-col font-mono overflow-hidden">
      {/* Header */}
      <header className="border-b border-[#1e3a5f] bg-[#050510] px-6 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-[0_0_20px_rgba(59,130,246,0.5)]">
            <Zap size={16} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-cyan-400 tracking-wider">JARVIS</h1>
            <p className="text-[10px] text-gray-500">Personal RAG Assistant</p>
          </div>
        </div>

        {/* Stats bar */}
        <div className="flex items-center gap-4 text-xs text-gray-400">
          {stats ? (
            <>
              <span className="flex items-center gap-1">
                <Database size={12} className="text-cyan-500" />
                {stats.total_documents?.toLocaleString()} docs
              </span>
              <span className="flex items-center gap-1">
                <Activity size={12} className="text-green-500" />
                {stats.llm}
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                Online
              </span>
            </>
          ) : (
            <span className="text-gray-600">Connecting...</span>
          )}
        </div>

        {/* Nav tabs */}
        <nav className="flex gap-1">
          {[
            { id: "chat", icon: MessageSquare, label: "Chat" },
            { id: "search", icon: Search, label: "Search" },
            { id: "files", icon: FolderOpen, label: "Files" },
            { id: "memory", icon: Brain, label: "Memory" },
          ].map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-all ${
                activeTab === id
                  ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                  : "text-gray-500 hover:text-gray-300 hover:bg-white/5"
              }`}
            >
              <Icon size={13} />
              {label}
            </button>
          ))}
        </nav>
      </header>

      {/* Main content */}
      <div className="flex-1 overflow-hidden flex">
        {/* ── CHAT TAB ── */}
        {activeTab === "chat" && (
          <div className="flex flex-col flex-1 overflow-hidden">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
                  {/* Avatar */}
                  <div className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                    msg.role === "user"
                      ? "bg-blue-600/30 border border-blue-500/30 text-blue-400"
                      : "bg-cyan-500/20 border border-cyan-500/30 text-cyan-400"
                  }`}>
                    {msg.role === "user" ? "M" : "J"}
                  </div>

                  {/* Bubble */}
                  <div className={`max-w-[75%] rounded-xl px-4 py-3 text-sm ${
                    msg.role === "user"
                      ? "bg-blue-600/20 border border-blue-500/20 text-gray-200"
                      : "bg-[#0d1a2e] border border-[#1e3a5f] text-gray-200"
                  }`}>
                    {msg.id === "streaming" && msg.content === "" ? (
                      <div className="flex items-center gap-2 text-cyan-400">
                        <Loader2 size={14} className="animate-spin" />
                        <span className="text-xs">Processing...</span>
                      </div>
                    ) : (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          code: ({ className, children, ...props }: any) => {
                            const isBlock = className?.includes("language-");
                            return isBlock ? (
                              <pre className="bg-black/50 rounded p-3 overflow-x-auto my-2 border border-gray-700">
                                <code className="text-xs text-green-400 font-mono">{children}</code>
                              </pre>
                            ) : (
                              <code className="bg-black/40 px-1 rounded text-cyan-300 text-xs" {...props}>{children}</code>
                            );
                          },
                          h1: ({ children }) => <h1 className="text-lg font-bold text-cyan-400 mb-2">{children}</h1>,
                          h2: ({ children }) => <h2 className="text-base font-bold text-cyan-300 mb-2 mt-3">{children}</h2>,
                          h3: ({ children }) => <h3 className="text-sm font-bold text-gray-200 mb-1 mt-2">{children}</h3>,
                          ul: ({ children }) => <ul className="list-disc list-inside space-y-1 my-2 text-gray-300">{children}</ul>,
                          ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 my-2 text-gray-300">{children}</ol>,
                          p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                          a: ({ href, children }) => <a href={href} className="text-cyan-400 underline hover:text-cyan-300" target="_blank">{children}</a>,
                          strong: ({ children }) => <strong className="text-white font-semibold">{children}</strong>,
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    )}
                    <p className="text-[10px] text-gray-600 mt-2">
                      {msg.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-[#1e3a5f] bg-[#050510] px-4 py-4">
              {/* Quick prompts */}
              <div className="flex gap-2 mb-3 overflow-x-auto pb-1">
                {[
                  "What have I been working on this week?",
                  "Show my recent GPT conversations about ML",
                  "Find my notes on RAG pipelines",
                  "What's in my thesis Chapter 4?",
                  "List my active projects",
                ].map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => setInput(prompt)}
                    className="shrink-0 text-xs px-3 py-1.5 rounded-full border border-[#1e3a5f] text-gray-400 hover:text-cyan-400 hover:border-cyan-500/50 transition-all whitespace-nowrap"
                  >
                    {prompt}
                  </button>
                ))}
              </div>

              <div className="flex gap-3 items-end">
                <div className="flex-1 relative">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKey}
                    placeholder="Ask JARVIS anything... (Shift+Enter for newline)"
                    rows={1}
                    className="w-full bg-[#0d1a2e] border border-[#1e3a5f] rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-cyan-500/50 resize-none max-h-32 overflow-y-auto transition-colors"
                    style={{ scrollbarWidth: "none" }}
                    onInput={e => {
                      const t = e.target as HTMLTextAreaElement;
                      t.style.height = "auto";
                      t.style.height = Math.min(t.scrollHeight, 128) + "px";
                    }}
                  />
                </div>
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || isStreaming}
                  className="shrink-0 w-11 h-11 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:cursor-not-allowed flex items-center justify-center transition-colors shadow-[0_0_15px_rgba(6,182,212,0.3)]"
                >
                  {isStreaming
                    ? <Loader2 size={18} className="animate-spin text-white" />
                    : <Send size={18} className="text-white" />
                  }
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── SEARCH TAB ── */}
        {activeTab === "search" && (
          <div className="flex-1 overflow-y-auto p-6">
            <h2 className="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2">
              <Search size={18} /> Knowledge Base Search
            </h2>
            <div className="flex gap-3 mb-6">
              <input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === "Enter" && runSearch()}
                placeholder="Search your memories..."
                className="flex-1 bg-[#0d1a2e] border border-[#1e3a5f] rounded-lg px-4 py-3 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-cyan-500/50"
              />
              <button
                onClick={runSearch}
                disabled={isSearching}
                className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
              >
                {isSearching ? <Loader2 size={16} className="animate-spin" /> : "Search"}
              </button>
            </div>

            <div className="space-y-4">
              {searchResults.map((r, i) => (
                <div key={i} className="bg-[#0d1a2e] border border-[#1e3a5f] rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{SOURCE_ICONS[r.source_type] || SOURCE_ICONS.default}</span>
                      <span className="text-xs font-medium text-cyan-400 capitalize">{r.source_type}</span>
                      {r.platform && <span className="text-xs text-gray-500">• {r.platform}</span>}
                      {r.date && <span className="text-xs text-gray-600">• {r.date?.slice(0, 10)}</span>}
                    </div>
                    <span className="text-xs text-gray-600">Score: {(r.score * 100).toFixed(0)}%</span>
                  </div>
                  <p className="text-sm text-gray-300 leading-relaxed line-clamp-4">{r.content}</p>
                </div>
              ))}
              {searchResults.length === 0 && searchQuery && !isSearching && (
                <p className="text-center text-gray-500 mt-8">No results found. Try a different query.</p>
              )}
            </div>
          </div>
        )}

        {/* ── FILES TAB ── */}
        {activeTab === "files" && (
          <div className="flex-1 overflow-y-auto p-6">
            <h2 className="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2">
              <FolderOpen size={18} /> File Browser
            </h2>
            <div className="flex gap-2 mb-4">
              <input
                value={filePath}
                onChange={e => setFilePath(e.target.value)}
                onKeyDown={e => e.key === "Enter" && browsePath(filePath)}
                className="flex-1 bg-[#0d1a2e] border border-[#1e3a5f] rounded-lg px-4 py-2 text-sm font-mono text-gray-300 focus:outline-none focus:border-cyan-500/50"
              />
              <button
                onClick={() => browsePath(filePath)}
                className="px-4 py-2 bg-blue-700 hover:bg-blue-600 rounded-lg text-sm transition-colors"
              >Go</button>
              <button
                onClick={() => browsePath(filePath + "/..")}
                className="px-4 py-2 bg-[#0d1a2e] border border-[#1e3a5f] hover:border-cyan-500/50 rounded-lg text-sm transition-colors"
              >↑ Up</button>
            </div>
            <div className="space-y-1">
              {fileItems.map((item, i) => (
                <div
                  key={i}
                  onClick={() => item.is_dir ? browsePath(item.path) : setInput(`Read the file: ${item.path}`)}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 cursor-pointer transition-colors group"
                >
                  <span className="text-lg">{item.is_dir ? "📁" : "📄"}</span>
                  <span className="flex-1 text-sm text-gray-300 group-hover:text-white">{item.name}</span>
                  {!item.is_dir && (
                    <span className="text-xs text-gray-600">{(item.size / 1024).toFixed(1)}KB</span>
                  )}
                  {!item.is_dir && (
                    <button
                      onClick={(e) => { e.stopPropagation(); setActiveTab("chat"); setInput(`Read the file: ${item.path}`); }}
                      className="opacity-0 group-hover:opacity-100 text-xs text-cyan-400 hover:text-cyan-300 transition-all"
                    >
                      Ask JARVIS →
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── MEMORY TAB ── */}
        {activeTab === "memory" && (
          <div className="flex-1 overflow-y-auto p-6">
            <h2 className="text-lg font-bold text-cyan-400 mb-6 flex items-center gap-2">
              <Brain size={18} /> Knowledge Base & Memory
            </h2>

            {/* Collection stats */}
            {stats && (
              <div className="grid grid-cols-2 gap-4 mb-8">
                {Object.entries(stats.collections).map(([name, info]) => (
                  <div key={name} className="bg-[#0d1a2e] border border-[#1e3a5f] rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-lg">
                        {name === "chat_history" ? "💬" : name === "code_files" ? "💻" : name === "documents" ? "📄" : "🌐"}
                      </span>
                      <h3 className="text-sm font-medium text-gray-200 capitalize">{name.replace("_", " ")}</h3>
                    </div>
                    <p className="text-2xl font-bold text-cyan-400">{info.count?.toLocaleString()}</p>
                    <p className="text-xs text-gray-500">chunks indexed</p>
                  </div>
                ))}
              </div>
            )}

            {/* Ingestion controls */}
            <div className="bg-[#0d1a2e] border border-[#1e3a5f] rounded-xl p-6">
              <h3 className="text-sm font-semibold text-gray-200 mb-4 flex items-center gap-2">
                <Database size={14} className="text-cyan-400" /> Ingest Your Data
              </h3>

              <div className="space-y-3 mb-4">
                {[
                  { label: "🤖 ChatGPT Export", placeholder: "/path/to/conversations.json", key: "chatgpt" },
                  { label: "✏️ Cursor History", placeholder: "~/.cursor/logs", key: "cursor" },
                  { label: "🟠 Claude Export", placeholder: "/path/to/claude_export.json", key: "claude" },
                  { label: "♊ Gemini Takeout", placeholder: "/path/to/Takeout/Gemini", key: "gemini" },
                ].map(({ label, placeholder }) => (
                  <div key={placeholder} className="flex items-center gap-3">
                    <span className="text-xs w-36 text-gray-400 shrink-0">{label}</span>
                    <input
                      placeholder={placeholder}
                      className="flex-1 bg-black/30 border border-gray-700 rounded px-3 py-2 text-xs font-mono text-gray-300 focus:outline-none focus:border-cyan-500/50"
                    />
                  </div>
                ))}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={triggerAutoIngest}
                  className="flex-1 py-2.5 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-sm font-medium transition-colors"
                >
                  ⚡ Auto-Detect & Ingest
                </button>
                <button
                  onClick={() => setInput("What data do you currently have indexed?")}
                  className="px-4 py-2.5 border border-[#1e3a5f] hover:border-cyan-500/50 rounded-lg text-sm transition-colors"
                >
                  Check Status
                </button>
              </div>

              {ingestStatus && (
                <p className="mt-3 text-xs text-cyan-400 flex items-center gap-2">
                  <Loader2 size={12} className="animate-spin" /> {ingestStatus}
                </p>
              )}
            </div>

            {/* Quick actions */}
            <div className="mt-6">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Quick Actions</h3>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Summarize recent work", icon: FileText },
                  { label: "Find duplicate ideas", icon: Search },
                  { label: "Show coding patterns", icon: Code },
                  { label: "Run terminal command", icon: Terminal },
                ].map(({ label, icon: Icon }) => (
                  <button
                    key={label}
                    onClick={() => { setActiveTab("chat"); setInput(label); }}
                    className="flex items-center gap-2 px-3 py-2.5 bg-[#0d1a2e] border border-[#1e3a5f] hover:border-cyan-500/30 rounded-lg text-xs text-gray-400 hover:text-cyan-400 transition-all text-left"
                  >
                    <Icon size={13} /> {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
