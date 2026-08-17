import React, { useEffect, useState, useRef } from 'react';
import {
  MessageSquare,
  Send,
  Sparkles,
  Database,
  Terminal,
  Bot,
  User,
  ShieldCheck,
  Zap,
  CheckCircle2
} from 'lucide-react';
import { api } from '../services/api';
import { ChatMessage, ChatSession, Dataset } from '../types';

export const ChatPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function init() {
      try {
        const dsRes = await api.getDatasets();
        setDatasets(dsRes.items || []);
        if (dsRes.items?.length > 0) {
          setSelectedDatasetId(dsRes.items[0].id);
        }
      } catch (err) {
        console.error('Error loading chat datasets:', err);
      }
    }
    init();
  }, []);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
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

  const samplePrompts = [
    'Which model performed best on this dataset?',
    'What is the distribution of the target variable?',
    'Did the critic detect any potential data leakage?',
    'Explain the most important predictive features.',
    'What is the total row count and schema overview?',
  ];

  return (
    <div className="p-8 max-w-5xl mx-auto h-[calc(100vh-4rem)] flex flex-col space-y-4">
      {/* Header & Dataset Context Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-4 border-b border-slate-800 shrink-0">
        <div>
          <h1 className="text-xl font-extrabold text-slate-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            AutoDS Grounded Data Science Agent
          </h1>
          <p className="text-xs text-slate-400">
            Answers are grounded strictly in computed dataset profiles, experiment metrics, and safe SQL tools.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <Database className="w-4 h-4 text-slate-400" />
          <select
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-xl p-2 focus:outline-none focus:border-emerald-500"
          >
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                Context: {ds.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Message Stream */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6 text-slate-500 p-8">
            <div className="w-16 h-16 rounded-2xl bg-slate-900 flex items-center justify-center text-indigo-400 border border-slate-800 shadow-inner">
              <Bot className="w-8 h-8" />
            </div>
            <div className="max-w-md space-y-1">
              <h3 className="font-bold text-sm text-slate-200">Ask AutoDS Anything</h3>
              <p className="text-xs text-slate-400">
                Inquire about model performance, feature attributions, dataset distributions, or ask questions that trigger automatic SQL queries.
              </p>
            </div>

            {/* Quick Prompts */}
            <div className="flex flex-wrap gap-2 justify-center max-w-xl">
              {samplePrompts.map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => handleSendMessage(prompt)}
                  className="text-xs px-3 py-1.5 rounded-full bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition"
                >
                  "{prompt}"
                </button>
              ))}
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
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-600 to-indigo-600 flex items-center justify-center text-white shrink-0 text-xs shadow-sm">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-2xl p-4 rounded-2xl text-xs space-y-2 leading-relaxed ${
                    isUser
                      ? 'bg-indigo-600 text-white rounded-tr-none'
                      : 'glass-panel text-slate-200 rounded-tl-none border-slate-800'
                  }`}
                >
                  <div className="whitespace-pre-wrap font-sans">{msg.content}</div>

                  {msg.tool_calls_json && (
                    <div className="pt-2 border-t border-slate-800/80 flex items-center space-x-2 text-[10px] text-emerald-400 font-mono">
                      <Terminal className="w-3 h-3 shrink-0" />
                      <span>Executed: {msg.tool_calls_json.tool_name || 'safe_tool'}</span>
                    </div>
                  )}
                </div>

                {isUser && (
                  <div className="w-8 h-8 rounded-xl bg-slate-800 flex items-center justify-center text-slate-300 shrink-0 text-xs border border-slate-700">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            );
          })
        )}
        {loading && (
          <div className="flex items-center space-x-3 text-slate-400 text-xs">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-600 to-indigo-600 flex items-center justify-center text-white shrink-0">
              <Bot className="w-4 h-4" />
            </div>
            <div className="glass-panel px-4 py-3 rounded-2xl flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
              <span>AutoDS is reasoning over verified evidence...</span>
            </div>
          </div>
        )}
        <div ref={chatBottomRef} />
      </div>

      {/* Input Bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSendMessage();
        }}
        className="shrink-0 flex items-center space-x-3 bg-slate-900/80 border border-slate-800 p-2 rounded-2xl focus-within:border-emerald-500 transition"
      >
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Ask a question about the dataset, models, or request safe SQL data analysis..."
          className="flex-1 bg-transparent px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!inputMessage.trim() || loading}
          className="p-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white transition disabled:opacity-40"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};
