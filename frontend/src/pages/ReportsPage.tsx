import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Download,
  Sparkles,
  Image as ImageIcon
} from 'lucide-react';
import { api } from '../services/api';
import { Report } from '../types';

export const ReportsPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [reports, setReports] = useState<Report[]>([]);
  const [activeReport, setActiveReport] = useState<Report | null>(null);

  useEffect(() => {
    async function load() {
      try {
        if (id) {
          const rep = await api.getReport(id);
          setActiveReport(rep);
        } else {
          const list = await api.getReports();
          setReports(list || []);
          if (list && list.length > 0) {
            setActiveReport(list[0]);
          }
        }
      } catch (err) {
        console.error('Failed to load reports:', err);
      }
    }
    load();
  }, [id]);

  const handleDownloadMarkdown = (report: Report) => {
    const blob = new Blob([report.full_report_markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AutoDS_Report_${report.id.substring(0, 8)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Evidence-Backed Reports</h1>
          <p className="text-sm text-slate-500">
            Synthesized Data Science reports separating Observed Facts, Model Evidence, and Actionable Business Insights.
          </p>
        </div>
      </div>

      {reports.length === 0 && !activeReport ? (
        <div className="glass-panel p-12 rounded-3xl text-center text-slate-500">
          No reports generated yet. Execute an autonomous Data Science run to generate evidence-backed reports.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left: Reports Selector (if viewing multiple) */}
          {!id && reports.length > 1 && (
            <div className="lg:col-span-4 space-y-3">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">Available Reports</h2>
              <div className="space-y-2">
                {reports.map((rep) => (
                  <div
                    key={rep.id}
                    onClick={() => setActiveReport(rep)}
                    className={`p-4 rounded-2xl border cursor-pointer transition ${
                      activeReport?.id === rep.id
                        ? 'bg-indigo-50 border-indigo-200 text-slate-900 shadow-2xs'
                        : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    <h3 className="font-bold text-sm line-clamp-1">{rep.title}</h3>
                    <p className="text-xs text-slate-500 mt-1 line-clamp-1">{rep.summary_markdown}</p>
                    <p className="text-[10px] text-slate-400 mt-2 font-mono">
                      {new Date(rep.created_at).toLocaleString()}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Right: Active Report Viewer */}
          <div className={`${!id && reports.length > 1 ? 'lg:col-span-8' : 'lg:col-span-12'} space-y-6`}>
            {activeReport && (
              <div className="glass-panel p-8 rounded-3xl space-y-6">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-100">
                  <div className="space-y-1">
                    <h2 className="text-xl font-bold text-slate-900">{activeReport.title}</h2>
                    <p className="text-xs text-slate-500 font-mono">
                      Generated: {new Date(activeReport.created_at).toUTCString()}
                    </p>
                  </div>

                  <button
                    onClick={() => handleDownloadMarkdown(activeReport)}
                    className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
                  >
                    <Download className="w-4 h-4 text-slate-600" />
                    <span>Download Markdown Report</span>
                  </button>
                </div>

                {/* Business Insights Callout */}
                {activeReport.business_insights_json?.insights && activeReport.business_insights_json.insights.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-emerald-600" />
                      Key Business Insights
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {activeReport.business_insights_json.insights.map((ins, i) => (
                        <div key={i} className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-1.5">
                          <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200">
                            {ins.category.replace('_', ' ')}
                          </span>
                          <h4 className="font-bold text-xs text-slate-900">{ins.title}</h4>
                          <p className="text-xs text-slate-600">{ins.finding}</p>
                          <p className="text-[11px] text-emerald-700 font-mono font-medium">Evidence: {ins.evidence}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Markdown Content */}
                <div className="prose max-w-none text-xs leading-relaxed space-y-4 pt-4 border-t border-slate-100">
                  <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 font-sans whitespace-pre-wrap text-slate-800">
                    {activeReport.full_report_markdown}
                  </div>
                </div>

                {/* Visual Artifacts Gallery */}
                {activeReport.artifact_paths && activeReport.artifact_paths.length > 0 && (
                  <div className="pt-6 border-t border-slate-100 space-y-4">
                    <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                      <ImageIcon className="w-4 h-4 text-indigo-600" />
                      Generated Visual Diagnostics & Plots
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {activeReport.artifact_paths.map((path, idx) => (
                        <div key={idx} className="rounded-2xl overflow-hidden border border-slate-200 bg-slate-50 p-2">
                          <img
                            src={`/${path}`}
                            alt={`Plot ${idx + 1}`}
                            className="w-full h-auto object-contain rounded-xl"
                            onError={(e) => {
                              (e.target as HTMLElement).style.display = 'none';
                            }}
                          />
                          <p className="text-[10px] font-mono text-slate-500 mt-2 px-2 truncate">{path}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
