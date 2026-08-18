import React, { useEffect, useState, useRef } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Send,
  Sparkles,
  Database,
  Terminal,
  Bot,
  User,
  ShieldCheck,
  TrendingUp,
  FileText,
  Trash2,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  BarChart3,
  Layers,
  ArrowRight
} from 'lucide-react';
import { api } from '../services/api';
import { ChatMessage, Dataset, Report } from '../types';

// Markdown and Evidence formatter for Assistant responses
const FormattedMessage: React.FC<{ content: string; isUser: boolean }> = ({ content, isUser }) => {
  if (isUser) {
    return <div className="whitespace-pre-wrap font-sans text-xs leading-relaxed">{content}</div>;
  }

  // Split lines and render structured blocks
  const lines = content.split('\n');
  const renderedElements: React.ReactNode[] = [];
  let tableBuffer: string[] = [];
  let inCodeBlock = false;
  let codeBuffer: string[] = [];
  let codeLang = '';

  const flushTable = (key: number) => {
    if (tableBuffer.length < 2) {
      tableBuffer.forEach((l, i) => renderedElements.push(<p key={`tbl-fallback-${key}-${i}`} className="my-1">{l}</p>));
      tableBuffer = [];
      return;
    }
    const headers = tableBuffer[0].split('|').map(s => s.trim()).filter(Boolean);
    const rows = tableBuffer.slice(2).map(r => r.split('|').map(s => s.trim()).filter(Boolean));
    renderedElements.push(
      <div key={`tbl-${key}`} className="my-2.5 overflow-x-auto rounded-xl border border-slate-200 bg-slate-50/70 p-1 shadow-2xs">
        <table className="min-w-full divide-y divide-slate-200 text-[11px] font-sans">
          <thead>
            <tr className="bg-slate-100/80 text-slate-700 font-bold">
              {headers.map((h, i) => (
                <th key={i} className="px-3 py-1.5 text-left">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {rows.map((row, rIdx) => (
              <tr key={rIdx} className="hover:bg-slate-50/50">
                {row.map((cell, cIdx) => (
                  <td key={cIdx} className="px-3 py-1.5 text-slate-800 font-mono text-[10.5px]">{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
    tableBuffer = [];
  };

  const flushCode = (key: number) => {
    renderedElements.push(
      <div key={`code-${key}`} className="my-2.5 rounded-xl border border-slate-800 bg-slate-900 p-3 text-slate-100 font-mono text-[11px] overflow-x-auto shadow-sm">
        {codeLang && <div className="text-[9px] uppercase tracking-wider text-emerald-400 font-bold mb-1.5">{codeLang}</div>}
        <pre className="whitespace-pre">{codeBuffer.join('\n')}</pre>
      </div>
    );
    codeBuffer = [];
    inCodeBlock = false;
    codeLang = '';
  };

  lines.forEach((line, idx) => {
    // Code block toggle
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        flushCode(idx);
      } else {
        if (tableBuffer.length > 0) flushTable(idx);
        inCodeBlock = true;
        codeLang = line.trim().replace(/^```/, '');
      }
      return;
    }

    if (inCodeBlock) {
      codeBuffer.push(line);
      return;
    }

    // Markdown Table lines
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      tableBuffer.push(line.trim());
      return;
    } else if (tableBuffer.length > 0) {
      flushTable(idx);
    }

    // Headers
    if (line.startsWith('### ')) {
      renderedElements.push(
        <h4 key={idx} className="font-extrabold text-slate-900 text-xs mt-3 mb-1.5 flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          {line.replace('### ', '')}
        </h4>
      );
      return;
    }
    if (line.startsWith('## ')) {
      renderedElements.push(
        <h3 key={idx} className="font-extrabold text-slate-900 text-sm mt-3.5 mb-2 pb-1 border-b border-slate-100">
          {line.replace('## ', '')}
        </h3>
      );
      return;
    }

    // Evidence tags & Callout alerts
    if (line.includes('> **[Evidence:') || line.includes('> [Evidence:')) {
      const match = line.match(/>\s*\*?\*?\[Evidence:\s*([^\]]+)\]\*?\*?(.*)/);
      const tag = match ? match[1] : 'Verified Analysis Context';
      const extra = match ? match[2] : '';
      renderedElements.push(
        <div key={idx} className="my-2 p-2 rounded-xl bg-emerald-50/90 border border-emerald-200 text-emerald-900 text-[11px] font-sans flex items-start gap-2 shadow-2xs">
          <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-emerald-800 uppercase tracking-wider text-[9.5px] bg-emerald-100 px-1.5 py-0.5 rounded mr-1.5">Evidence Source</span>
            <span className="font-semibold">{tag}</span>
            {extra && <span className="text-emerald-700 ml-1">{extra}</span>}
          </div>
        </div>
      );
      return;
    }

    if (line.startsWith('> [!IMPORTANT]') || line.startsWith('> [!NOTE]') || line.startsWith('> [!WARNING]')) {
      const type = line.includes('IMPORTANT') ? 'IMPORTANT' : (line.includes('WARNING') ? 'WARNING' : 'NOTE');
      renderedElements.push(
        <div key={idx} className="mt-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
          <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
          <span>{type}</span>
        </div>
      );
      return;
    }
    if (line.startsWith('> ')) {
      renderedElements.push(
        <blockquote key={idx} className="my-1.5 pl-3 border-l-2 border-slate-300 text-slate-600 italic text-[11px]">
          {line.replace(/^>\s*/, '')}
        </blockquote>
      );
      return;
    }

    // Bullet points
    if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      const rawItem = line.trim().replace(/^[-*]\s+/, '');
      renderedElements.push(
        <li key={idx} className="text-slate-800 text-xs ml-4 my-0.5 list-disc leading-relaxed">
          <span dangerouslySetInnerHTML={{ __html: formatInlineMarkdown(rawItem) }} />
        </li>
      );
      return;
    }

    // Numbered points
    if (/^\d+\.\s+/.test(line.trim())) {
      const num = line.trim().match(/^(\d+)\.\s+/)?.[1] || '';
      const rawItem = line.trim().replace(/^\d+\.\s+/, '');
      renderedElements.push(
        <div key={idx} className="flex items-start gap-1.5 my-1 text-slate-800 text-xs leading-relaxed">
          <span className="font-bold text-slate-500 shrink-0 font-mono text-[10px]">{num}.</span>
          <span dangerouslySetInnerHTML={{ __html: formatInlineMarkdown(rawItem) }} />
        </div>
      );
      return;
    }

    if (line.trim() === '') {
      renderedElements.push(<div key={idx} className="h-1.5" />);
      return;
    }

    renderedElements.push(
      <p key={idx} className="my-1 leading-relaxed text-slate-800 text-xs" dangerouslySetInnerHTML={{ __html: formatInlineMarkdown(line) }} />
    );
  });

  if (tableBuffer.length > 0) flushTable(9999);
  if (inCodeBlock) flushCode(9999);

  return <div className="space-y-1 font-sans">{renderedElements}</div>;
};

// Helper for inline bold, code, and links
function formatInlineMarkdown(text: string): string {
  let res = text
    .replace(/\*\*([^*]+)\*\*/g, '<strong class="font-bold text-slate-900">$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em class="italic">$1</em>')
    .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 bg-slate-100 border border-slate-200 rounded text-[11px] font-mono text-emerald-800">$1</code>');
  return res;
}

export const ChatPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const urlAnalysisId = searchParams.get('analysisId') || '';
  const urlReportId = searchParams.get('reportId') || '';
  const urlDatasetId = searchParams.get('datasetId') || '';

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(urlDatasetId);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string>(urlAnalysisId);
  const [selectedReportId, setSelectedReportId] = useState<string>(urlReportId);
  
  const [contextData, setContextData] = useState<Record<string, any> | null>(null);
  const [contextLoading, setContextLoading] = useState<boolean>(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Load initial dataset and reports list
  useEffect(() => {
    async function init() {
      try {
        const [dsRes, repRes] = await Promise.all([
          api.getDatasets(),
          api.getReports().catch(() => [])
        ]);
        setDatasets(dsRes.items || []);
        setReports(repRes || []);

        if (!selectedDatasetId && dsRes.items?.length > 0) {
          setSelectedDatasetId(dsRes.items[0].id);
        }
      } catch (err) {
        console.error('Error loading chat dependencies:', err);
      }
    }
    init();
  }, []);

  // Fetch structured context whenever selection changes
  useEffect(() => {
    async function fetchContext() {
      if (!selectedAnalysisId && !selectedReportId && !selectedDatasetId) {
        setContextData(null);
        return;
      }
      setContextLoading(true);
      try {
        const ctx = await api.getAgentContext({
          analysis_id: selectedAnalysisId || undefined,
          report_id: selectedReportId || undefined,
          dataset_id: selectedDatasetId || undefined,
        });
        setContextData(ctx);
      } catch (err) {
        console.error('Failed to fetch agent context:', err);
      } finally {
        setContextLoading(false);
      }
    }
    fetchContext();
  }, [selectedAnalysisId, selectedReportId, selectedDatasetId]);

  useEffect(() => {
    if (chatBottomRef.current && typeof chatBottomRef.current.scrollIntoView === 'function') {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSendMessage = async (textToSend?: string) => {
    const content = textToSend || inputMessage;
    if (!content.trim() || loading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      session_id: sessionId || '',
      role: 'user',
      content: content.trim(),
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await api.sendChatMessage({
        session_id: sessionId,
        dataset_id: selectedDatasetId || undefined,
        analysis_id: selectedAnalysisId || undefined,
        report_id: selectedReportId || undefined,
        content: content.trim(),
      });
      setSessionId(response.session_id);
      setMessages((prev) => [...prev, response]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        session_id: sessionId || '',
        role: 'assistant',
        content: `Error: ${err.message || 'Failed to reach agent.'}`,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearConversation = () => {
    setMessages([]);
    setSessionId(undefined);
  };

  const handleSelectReportContext = (rep: Report) => {
    setSelectedReportId(rep.id);
    setSelectedAnalysisId(rep.analysis_id);
    setSearchParams({ analysisId: rep.analysis_id, reportId: rep.id });
  };

  const samplePrompts = [
    'Why did this model win?',
    'Explain the threshold',
    'What are the biggest risks?',
    'Check for leakage',
    'Summarize the report',
    'Explain like a business manager',
    'What happens if I use 0.50 cutoff?',
    'Which customers are being missed?',
  ];

  const activeDatasetName = contextData?.dataset?.name || 'No Dataset Selected';
  const activeAnalysisId = contextData?.analysis?.id;
  const activeProblemType = contextData?.analysis?.problem_type;
  const activeChampionName = contextData?.champion_model?.name || (contextData?.leaderboard?.[0]?.model_name);
  const activeThreshold = contextData?.threshold_analysis?.selected_threshold;

  return (
    <div className="p-6 md:p-8 w-full h-[calc(100vh-4rem)] flex flex-col space-y-4 max-w-7xl mx-auto">
      {/* ========================================================================= */}
      {/* 1. TOP CONTEXT INDICATOR & SELECTOR HEADER */}
      {/* ========================================================================= */}
      <div className="glass-panel p-4 md:p-5 rounded-3xl border border-slate-200/90 bg-white shadow-2xs shrink-0 flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Left: Indicator Information */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="w-7 h-7 rounded-xl bg-gradient-to-tr from-emerald-600 to-indigo-600 flex items-center justify-center text-white shrink-0 shadow-2xs">
              <Sparkles className="w-4 h-4" />
            </div>
            <h1 className="text-base font-extrabold text-slate-900">
              AutoDS Grounded Data Science Agent
            </h1>
            <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200">
              <ShieldCheck className="w-3 h-3 text-emerald-600" />
              Grounded in Computed Evidence
            </span>
          </div>

          {/* Context Status Banner */}
          {contextData?.has_context ? (
            <div className="flex items-center gap-2 text-xs flex-wrap">
              <span className="font-semibold text-slate-700">Analyzing:</span>
              <span className="font-extrabold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-lg border border-emerald-200 font-mono text-[11px]">
                {activeDatasetName}
              </span>
              {activeAnalysisId && (
                <>
                  <span className="text-slate-300">•</span>
                  <span className="text-slate-500 font-mono text-[11px]">Analysis ID: {activeAnalysisId.substring(0, 8)}</span>
                </>
              )}
              {activeChampionName && (
                <>
                  <span className="text-slate-300">•</span>
                  <span className="text-slate-700">Champion: <strong className="text-indigo-600">{activeChampionName}</strong></span>
                </>
              )}
              {activeProblemType && (
                <>
                  <span className="text-slate-300">•</span>
                  <span className="text-slate-500 uppercase tracking-wider text-[10px] font-bold">{activeProblemType}</span>
                </>
              )}
              {activeThreshold !== undefined && (
                <>
                  <span className="text-slate-300">•</span>
                  <span className="text-slate-600 font-mono text-[10.5px]">Cutoff: {activeThreshold.toFixed(2)}</span>
                </>
              )}
            </div>
          ) : (
            <div className="text-xs text-amber-700 flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5" />
              <span>No active analysis selected. Select a dataset or completed report below to ground responses.</span>
            </div>
          )}
        </div>

        {/* Right: Context Selectors & Clear Action */}
        <div className="flex items-center gap-2.5 flex-wrap shrink-0">
          {/* Report / Analysis Quick Switch */}
          {reports.length > 0 && (
            <select
              value={selectedReportId}
              onChange={(e) => {
                const rep = reports.find(r => r.id === e.target.value);
                if (rep) handleSelectReportContext(rep);
                else {
                  setSelectedReportId('');
                  setSelectedAnalysisId('');
                  setSearchParams({});
                }
              }}
              className="bg-slate-50 border border-slate-200 text-slate-800 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-500 shadow-2xs font-medium max-w-[220px] truncate"
            >
              <option value="">Switch Report / Analysis...</option>
              {reports.map((r) => (
                <option key={r.id} value={r.id}>
                  📄 {r.title || `Report ${r.id.substring(0, 8)}`}
                </option>
              ))}
            </select>
          )}

          {/* Dataset Selector */}
          <div className="flex items-center space-x-1.5">
            <Database className="w-3.5 h-3.5 text-slate-500" />
            <select
              value={selectedDatasetId}
              onChange={(e) => {
                setSelectedDatasetId(e.target.value);
                setSelectedAnalysisId('');
                setSelectedReportId('');
                setSearchParams({ datasetId: e.target.value });
              }}
              className="bg-slate-50 border border-slate-200 text-slate-800 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-emerald-500 shadow-2xs font-medium max-w-[180px] truncate"
            >
              {datasets.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  Dataset: {ds.name}
                </option>
              ))}
            </select>
          </div>

          {/* View Report button if report exists */}
          {(selectedReportId || contextData?.report_summary?.id) && (
            <button
              onClick={() => navigate(`/reports/${selectedReportId || contextData?.report_summary?.id}`)}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 text-xs font-bold shadow-2xs transition"
              title="View full report document"
            >
              <FileText className="w-3.5 h-3.5 text-indigo-600" />
              <span>View Report</span>
            </button>
          )}

          {/* Clear Chat */}
          {messages.length > 0 && (
            <button
              onClick={handleClearConversation}
              className="inline-flex items-center gap-1 px-3 py-2 rounded-xl bg-slate-100 hover:bg-rose-50 hover:text-rose-700 text-slate-600 border border-slate-200 text-xs font-bold shadow-2xs transition"
              title="Clear conversation"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Clear</span>
            </button>
          )}
        </div>
      </div>

      {/* ========================================================================= */}
      {/* 2. MESSAGE STREAM */}
      {/* ========================================================================= */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2 bg-white rounded-3xl border border-slate-200/90 p-4 md:p-6 shadow-sm">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6 text-slate-500 p-8">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-emerald-100 to-indigo-100 flex items-center justify-center text-emerald-700 border border-emerald-200 shadow-2xs">
              <Bot className="w-8 h-8" />
            </div>
            
            <div className="max-w-xl space-y-2">
              <h3 className="font-extrabold text-base text-slate-900">
                Ask AutoDS About {contextData?.has_context ? activeDatasetName : 'Your Machine Learning Pipelines'}
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                {contextData?.has_context ? (
                  <>
                    I have access to the verified analysis context for <strong>{activeDatasetName}</strong> (Analysis ID: <code>{activeAnalysisId?.substring(0, 8)}</code>).
                    I can explain why the champion model was chosen, break down decision thresholds, report on Critic audits and leakage, and translate technical findings for stakeholders.
                  </>
                ) : (
                  <>
                    Inquire about model performance, feature attributions, dataset distributions, or ask questions that trigger automatic SQL queries against raw tables.
                  </>
                )}
              </p>
            </div>

            {/* Quick Suggested Prompts */}
            <div className="space-y-2 max-w-2xl">
              <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Suggested Questions for Active Context
              </div>
              <div className="flex flex-wrap gap-2 justify-center">
                {samplePrompts.map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => handleSendMessage(prompt)}
                    className="text-xs px-3.5 py-2 rounded-xl bg-slate-50 hover:bg-emerald-50 hover:text-emerald-800 hover:border-emerald-200 text-slate-700 border border-slate-200/90 shadow-2xs transition font-medium text-left flex items-center gap-1.5 group cursor-pointer"
                  >
                    <span>{prompt}</span>
                    <ArrowRight className="w-3 h-3 text-slate-400 group-hover:text-emerald-600 transition" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg) => {
            const isUser = msg.role === 'user';
            return (
              <div
                key={msg.id}
                className={`flex items-start space-x-3 ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!isUser && (
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-600 to-indigo-600 flex items-center justify-center text-white shrink-0 text-xs shadow-2xs mt-1">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-3xl p-4 md:p-5 rounded-2xl text-xs space-y-2 leading-relaxed shadow-xs ${
                    isUser
                      ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white rounded-tr-none'
                      : 'bg-slate-50 text-slate-900 rounded-tl-none border border-slate-200/90'
                  }`}
                >
                  <FormattedMessage content={msg.content} isUser={isUser} />

                  {msg.tool_calls_json && (
                    <div className="pt-2.5 mt-2 border-t border-slate-200/60 flex items-center space-x-2 text-[10px] text-emerald-800 font-mono font-semibold">
                      <Terminal className="w-3.5 h-3.5 shrink-0 text-emerald-600" />
                      <span>Executed Safe Tool: {msg.tool_calls_json.tool_name || 'safe_sql_tool'}</span>
                    </div>
                  )}
                </div>

                {isUser && (
                  <div className="w-8 h-8 rounded-xl bg-slate-200 flex items-center justify-center text-slate-700 shrink-0 text-xs border border-slate-300 mt-1">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            );
          })
        )}

        {loading && (
          <div className="flex items-center space-x-3 text-slate-500 text-xs">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-600 to-indigo-600 flex items-center justify-center text-white shrink-0">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-slate-50 border border-slate-200 px-4 py-3 rounded-2xl flex items-center space-x-2 shadow-2xs">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
              <span className="text-slate-700 font-semibold">AutoDS is reasoning over verified evidence...</span>
            </div>
          </div>
        )}
        <div ref={chatBottomRef} />
      </div>

      {/* ========================================================================= */}
      {/* 3. SUGGESTED CHIPS (VISIBLE DURING CHAT) */}
      {/* ========================================================================= */}
      {messages.length > 0 && (
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 shrink-0 scrollbar-none">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider shrink-0 mr-1">Quick:</span>
          {samplePrompts.slice(0, 5).map((prompt, i) => (
            <button
              key={i}
              onClick={() => handleSendMessage(prompt)}
              disabled={loading}
              className="text-[11px] px-3 py-1 rounded-full bg-white hover:bg-slate-100 text-slate-700 border border-slate-200 shadow-2xs transition shrink-0 disabled:opacity-50"
            >
              {prompt}
            </button>
          ))}
        </div>
      )}

      {/* ========================================================================= */}
      {/* 4. INPUT BAR */}
      {/* ========================================================================= */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSendMessage();
        }}
        className="shrink-0 flex items-center space-x-3 bg-white border border-slate-300 p-2.5 rounded-2xl focus-within:border-emerald-500 shadow-2xs transition"
      >
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder={
            contextData?.has_context
              ? `Ask a question about ${activeDatasetName} (e.g. "Why did ${activeChampionName || 'this model'} win?", "Check for leakage")...`
              : "Ask a question about models, datasets, or type a custom SELECT SQL query..."
          }
          className="flex-1 bg-transparent px-3 py-1.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!inputMessage.trim() || loading}
          className="p-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white transition disabled:opacity-40 shadow-2xs cursor-pointer"
          title="Send message"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};
