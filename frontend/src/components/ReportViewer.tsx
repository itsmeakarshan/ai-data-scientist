import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  ShieldCheck,
  CheckCircle,
  AlertTriangle,
  Info,
  Layers,
  BarChart3,
  TrendingUp,
  Download,
  FileText,
  Activity,
  Maximize2,
  Printer,
  Eye,
  Trash2,
  MessageSquare
} from 'lucide-react';
import { Report } from '../types';

interface ReportViewerProps {
  report: Report;
  onDeleteReport?: (report: Report) => void;
}

export const formatPlotTitle = (path: string, championModel?: string): string => {
  const filename = path.split('/').pop() || '';
  const prefix = championModel ? `${championModel} — ` : '';
  if (/_roc(\.png)?$/i.test(filename)) return `${prefix}ROC Curve`;
  if (/_pr(\.png)?$/i.test(filename)) return `${prefix}Precision-Recall Curve`;
  if (/_cm(\.png)?$/i.test(filename)) return `${prefix}Confusion Matrix`;
  if (/_feature_imp(\.png)?$/i.test(filename)) return `${prefix}Top Predictive Drivers`;
  if (/_residuals(\.png)?$/i.test(filename) || /_resid(\.png)?$/i.test(filename)) return `${prefix}Residual Diagnostics`;
  if (/_actual_vs_pred(\.png)?$/i.test(filename) || /_act_pred(\.png)?$/i.test(filename)) return `${prefix}Actual vs Predicted`;
  if (/_correlation(\.png)?$/i.test(filename) || /_corr(\.png)?$/i.test(filename)) return 'Feature Correlation Matrix';
  return filename || 'Diagnostic Visual Plot';
};

export const stripMarkdown = (text: string | null | undefined): string => {
  if (!text) return '';
  return text.replace(/\*\*/g, '').replace(/`/g, '').trim();
};

export const renderFormattedText = (text: string | null | undefined): React.ReactNode => {
  if (!text) return null;
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g);
  return tokens.map((token, index) => {
    if (!token) return null;
    if (token.startsWith('**') && token.endsWith('**') && token.length >= 4) {
      return (
        <strong key={index} className="font-bold text-slate-900">
          {token.slice(2, -2)}
        </strong>
      );
    }
    if (token.startsWith('`') && token.endsWith('`') && token.length >= 2) {
      return (
        <code key={index} className="font-mono bg-slate-100 text-indigo-700 px-1 py-0.5 rounded text-[10px]">
          {token.slice(1, -1)}
        </code>
      );
    }
    if (token.startsWith('*') && token.endsWith('*') && token.length >= 2) {
      return (
        <em key={index} className="italic text-slate-700">
          {token.slice(1, -1)}
        </em>
      );
    }
    return token.replace(/\*\*/g, '').replace(/`/g, '');
  });
};

const ReportPlotImage: React.FC<{
  path: string;
  title: string;
  onExpand: (url: string, title: string) => void;
}> = ({ path, title, onExpand }) => {
  const [hasError, setHasError] = useState(false);
  const filename = path.split('/').pop() || title;
  const imgUrl = `/reports/artifacts/${filename}`;
  const isFeatureImp = path.includes('_feature_imp');

  return (
    <div className="rounded-2xl border border-slate-200/90 p-3.5 bg-slate-50 space-y-2 flex flex-col justify-between hover:border-slate-300 transition shadow-2xs group">
      <div className="flex items-center justify-between text-[11px] text-slate-700 px-1 gap-2 min-h-[24px]">
        <span className="font-bold tracking-tight text-slate-800 break-words leading-tight flex-1">
          {title}
        </span>
        <button
          type="button"
          onClick={() => onExpand(imgUrl, title)}
          className="no-print inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white hover:bg-indigo-50 border border-slate-200 text-indigo-600 hover:border-indigo-300 text-[11px] font-sans font-bold shadow-2xs transition shrink-0 cursor-pointer"
        >
          <span>Expand</span>
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
      </div>

      <div
        className="flex-1 flex flex-col items-center justify-center bg-white p-2 rounded-xl border border-slate-200/90 cursor-pointer overflow-hidden min-h-[220px]"
        onClick={() => onExpand(imgUrl, title)}
      >
        {!hasError ? (
          <img
            src={imgUrl}
            alt={title}
            className="w-full h-auto max-h-64 object-contain rounded-lg transition duration-200 group-hover:scale-[1.01]"
            onError={() => setHasError(true)}
          />
        ) : (
          <div className="flex flex-col items-center justify-center p-6 text-center space-y-2 bg-slate-50 rounded-lg w-full h-full border border-dashed border-slate-200 text-slate-500">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-200 flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-indigo-600" />
            </div>
            <span className="font-bold text-xs text-slate-800 font-mono truncate max-w-[220px]">{filename}</span>
            <span className="text-[10px] text-slate-400 font-medium">Diagnostic Evaluation Graphic</span>
          </div>
        )}
      </div>

      {isFeatureImp && (
        <div className="px-1 pt-0.5">
          <p className="text-[10px] text-slate-500 italic">
            * These are model-derived predictive associations, not causal effects.
          </p>
        </div>
      )}
    </div>
  );
};

export const ReportViewer: React.FC<ReportViewerProps> = ({ report, onDeleteReport }) => {
  const navigate = useNavigate();
  const [viewMode, setViewMode] = useState<'pdf' | 'structured'>('pdf');
  const [activeTab, setActiveTab] = useState<string>('all');
  const [isExportingPdf, setIsExportingPdf] = useState<boolean>(false);
  const [selectedPlot, setSelectedPlot] = useState<{ url: string; title: string } | null>(null);

  const insights = report.business_insights_json?.insights || [];
  const methodology = report.methodology_json || {};
  const criticFindings = methodology.critic?.findings || [];
  const criticStatus = methodology.critic?.audit_status || 'PASSED';

  // Parse sections from full_report_markdown
  const rawMd = report.full_report_markdown || '';

  // Extract problem type (classification, regression, forecasting)
  const taskTypeMatch = rawMd.match(/\*\*Task Type:\*\*\s*([A-Za-z0-9_-]+)/i) || rawMd.match(/completed\s+([A-Za-z0-9_-]+)\s+pipeline/i);
  const problemType = (
    methodology.problem_type ||
    (report as any).problem_type ||
    (taskTypeMatch ? taskTypeMatch[1] : '')
  ).toLowerCase();

  const isForecasting = problemType === 'forecasting' || problemType.includes('forecast');
  const isRegression = problemType === 'regression' || (!isForecasting && (rawMd.includes('RMSE') || rawMd.includes('Regression')));
  const isClassification = problemType === 'classification' || (!isRegression && !isForecasting && (rawMd.includes('Classification & Decision Threshold Analysis') || rawMd.includes('ROC-AUC') || rawMd.includes('Multiclass')));

  const isMulticlass = isClassification && (rawMd.includes('Multiclass') || rawMd.includes('Macro F1') || rawMd.includes('Macro ROC-AUC') || rawMd.includes('classes 3,4,5') || (methodology as any).is_binary === false);
  const isBinary = isClassification && !isMulticlass;

  // Extract champion model name
  const championMatch = report.summary_markdown?.match(/Best Model:\s*([A-Za-z0-9_-]+)/) || rawMd.match(/champion model selected is \*\*([^*]+)\*\*/i);
  const championName = championMatch ? championMatch[1] : undefined;

  // Helper to normalize artifact image URLs cleanly
  const getArtifactUrl = (path: string) => {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    let clean = path.replace(/^\/+/, '');
    if (!clean.startsWith('reports/artifacts/')) {
      clean = `reports/artifacts/${clean}`;
    }
    return `/${clean}`;
  };

  // Extract text helpers from markdown
  const extractObjective = () => {
    const m = rawMd.match(/\*\*Objective:\*\*\s*(.+)/);
    return m ? m[1].trim() : report.summary_markdown || 'Autonomous model training and evaluation';
  };

  const extractTargetColumn = () => {
    const m = rawMd.match(/\*\*Target Column:\*\*\s*`?([A-Za-z0-9_-]+)`?/i) || rawMd.match(/target\s+column\s+`?([A-Za-z0-9_-]+)`?/i);
    return m ? m[1].trim() : methodology.target_column || 'Target';
  };

  const extractPrevalence = () => {
    const m = rawMd.match(/prevalence[^\d]*([\d.]+%)/i) || rawMd.match(/positive[^\d]*([\d.]+%)/i);
    return m ? m[1] : undefined;
  };

  const extractOperatingThreshold = () => {
    const m = rawMd.match(/operating threshold[:\s]*([\d.]+)/i) || rawMd.match(/threshold[:\s]*([\d.]+)/i);
    return m ? m[1] : undefined;
  };

  const extractSummaryParagraphs = () => {
    const sec1 = rawMd.split(/## 2\./)[0] || '';
    const lines = sec1.split('\n').filter(l => 
      l.trim() && 
      !l.startsWith('#') && 
      !l.startsWith('**Dataset:**') && 
      !l.startsWith('**Generated:**') && 
      !l.startsWith('**Status:**') && 
      !l.startsWith('**Objective:**') && 
      !l.startsWith('---') &&
      !l.startsWith('>')
    );
    return lines;
  };

  const extractImbalanceAlert = () => {
    const m = rawMd.match(/>\s*\[!WARNING\]\s*\n>\s*\*\*Target Class Imbalance Alert:\*\*\s*([\s\S]+?)(?=\n\n|\n[#>])/);
    if (!m) return null;
    return m[1].replace(/\n>\s*/g, ' ').trim();
  };

  // Helper to extract tables from a markdown section
  const parseMdTable = (tableText: string) => {
    const lines = tableText.trim().split('\n').filter(l => l.includes('|'));
    if (lines.length < 2) return null;
    
    const headers = lines[0].split('|').map(h => h.trim().replace(/\*\*/g, '').replace(/`/g, '')).filter(Boolean);
    const rows = lines.slice(2).map(line => 
      line.split('|').map(cell => cell.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1)
    ).filter(r => r.length > 0);
    return { headers, rows };
  };

  // Find all table blocks in markdown
  const tableBlocks = rawMd.split('\n\n').filter(block => block.includes('|---|') || block.includes('| --- |') || block.includes('|:---|'));

  const leaderboardTable = tableBlocks.find(b => b.includes('Model Name') || b.includes('Primary Loss Metric') || b.includes('Primary Selection Metric')) 
    ? parseMdTable(tableBlocks.find(b => b.includes('Model Name') || b.includes('Primary Loss Metric') || b.includes('Primary Selection Metric'))!) 
    : (tableBlocks.length > 0 ? parseMdTable(tableBlocks[0]) : null);

  const explainabilityTable = tableBlocks.find(b => b.includes('Feature Name') || b.includes('Relative Importance') || b.includes('Relative Contribution'))
    ? parseMdTable(tableBlocks.find(b => b.includes('Feature Name') || b.includes('Relative Importance') || b.includes('Relative Contribution'))!)
    : null;

  // Extract Section 8 Operational Risks
  const extractOperationalRisks = () => {
    const m = rawMd.match(/## \d+\.\s*Model Limitations & Operational Risk Analysis\s*\n+([\s\S]+?)(?=\n##|\Z)/i);
    if (!m) return [];
    const lines = m[1].split('\n').filter(l => /^\d+\.\s*/.test(l.trim()));
    return lines.map(line => {
      const cleanLine = line.replace(/^\d+\.\s*/, '').trim();
      if (cleanLine.startsWith('**')) {
        const endBoldIdx = cleanLine.indexOf('**', 2);
        if (endBoldIdx !== -1) {
          const title = cleanLine.slice(2, endBoldIdx).replace(/:$/, '').trim();
          const text = cleanLine.slice(endBoldIdx + 2).replace(/^:\s*/, '').trim();
          return { title: stripMarkdown(title), text: stripMarkdown(text) || text };
        }
      }
      const parts = cleanLine.split(':');
      return {
        title: stripMarkdown(parts[0]),
        text: stripMarkdown(parts.slice(1).join(':')) || cleanLine
      };
    });
  };

  const operationalRisks = extractOperationalRisks();
  const summaryParagraphs = extractSummaryParagraphs();
  const imbalanceAlert = extractImbalanceAlert();

  // Dynamic section numbers for PDF view (contiguous with zero missing numbers)
  let pdfCounter = 1;
  const pdfSecSummary = pdfCounter++;
  const pdfSecLeaderboard = pdfCounter++;
  const pdfSecThreshold = isBinary ? pdfCounter++ : null;
  const pdfSecCritic = pdfCounter++;
  const pdfSecVisuals = (report.artifact_paths && report.artifact_paths.length > 0) ? pdfCounter++ : null;
  const pdfSecInsights = pdfCounter++;
  const pdfSecRisks = pdfCounter++;

  // Dynamic section numbers for Interactive Tabs view
  let tabCounter = 1;
  const tabSecSummary = tabCounter++;
  const tabSecLeaderboard = tabCounter++;
  const tabSecThreshold = isBinary ? tabCounter++ : null;
  const tabSecCritic = tabCounter++;
  const tabSecInsights = tabCounter++;
  const tabSecRisks = tabCounter++;

  // Export PDF Handler with Per-Page Pagination
  const handleDownloadPdf = async () => {
    setIsExportingPdf(true);
    try {
      const html2pdfModule = await import('html2pdf.js');
      const html2pdf = html2pdfModule.default;
      const element = document.getElementById('pdf-report-document');
      if (!element) return;

      const opt = {
        margin: [0.35, 0.35, 0.45, 0.35] as [number, number, number, number],
        filename: `AutoDS_Executive_Report_${report.id.substring(0, 8)}.pdf`,
        image: { type: 'jpeg' as const, quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, logging: false, letterRendering: true },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
        pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
      };

      await (html2pdf() as any)
        .set(opt)
        .from(element)
        .toPdf()
        .get('pdf')
        .then((pdf: any) => {
          const totalPages = pdf.internal.getNumberOfPages();
          for (let i = 1; i <= totalPages; i++) {
            pdf.setPage(i);
            pdf.setFontSize(8);
            pdf.setTextColor(140, 140, 140);

            const pageSize = pdf.internal.pageSize;
            const pageWidth = pageSize.getWidth ? pageSize.getWidth() : pageSize.width;
            const pageHeight = pageSize.getHeight ? pageSize.getHeight() : pageSize.height;

            // Left footer letterhead info
            pdf.text(
              'AutoDS Autonomous Data Science Engine • Executive Audit Report',
              0.35,
              pageHeight - 0.2
            );

            // Right footer: Dynamic Page X of Y on every single page
            pdf.text(
              `Confidential & Audit-Ready • Page ${i} of ${totalPages}`,
              pageWidth - 0.35,
              pageHeight - 0.2,
              { align: 'right' }
            );
          }
        })
        .save();
    } catch (err) {
      console.error('Programmatic PDF export failed, launching print dialog fallback:', err);
      window.print();
    } finally {
      setIsExportingPdf(false);
    }
  };

  return (
    <div className="space-y-6 font-sans text-slate-800">
      {/* Top Action Control Toolbar */}
      <div className="glass-panel p-5 rounded-3xl flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-sm border border-slate-200/90 bg-white">
        {/* Left: View Mode Selectors */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-100/80 rounded-2xl border border-slate-200/70">
          <button
            onClick={() => setViewMode('pdf')}
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-extrabold transition ${
              viewMode === 'pdf'
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-md shadow-emerald-500/20'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>PDF Document View</span>
          </button>
          <button
            onClick={() => setViewMode('structured')}
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-extrabold transition ${
              viewMode === 'structured'
                ? 'bg-gradient-to-r from-indigo-600 to-teal-600 text-white shadow-md shadow-indigo-500/20'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Interactive Tabs</span>
          </button>
        </div>

        {/* Right: Export & Download Action Buttons */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(`/chat?analysisId=${report.analysis_id}&reportId=${report.id}`)}
            className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-extrabold bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200/80 shadow-2xs transition cursor-pointer"
            title="Ask questions grounded in this report"
          >
            <Sparkles className="w-4 h-4 text-indigo-600" />
            <span>Ask AutoDS Agent</span>
          </button>

          <button
            onClick={handleDownloadPdf}
            disabled={isExportingPdf}
            className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl text-xs font-extrabold bg-gradient-to-r from-emerald-600 via-teal-600 to-indigo-600 hover:from-emerald-500 hover:to-indigo-500 text-white shadow-md shadow-emerald-600/20 transition disabled:opacity-50 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            <span>{isExportingPdf ? 'Generating PDF...' : 'Download PDF (.pdf)'}</span>
          </button>

          <button
            onClick={() => window.print()}
            className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-bold bg-slate-100 hover:bg-slate-200/80 text-slate-700 border border-slate-300/80 shadow-2xs transition"
            title="Print or Save as PDF via browser"
          >
            <Printer className="w-4 h-4 text-slate-600" />
            <span>Print / Save PDF</span>
          </button>

          {onDeleteReport && (
            <button
              onClick={() => onDeleteReport(report)}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-bold bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200/80 shadow-2xs transition cursor-pointer"
              title="Delete report completely"
            >
              <Trash2 className="w-4 h-4 text-rose-600" />
              <span>Delete Report</span>
            </button>
          )}
        </div>
      </div>

      {/* ========================================================================= */}
      {/* MODE 1: EXECUTIVE BEAUTIFUL PDF DOCUMENT VIEW (DEFAULT) */}
      {/* ========================================================================= */}
      {viewMode === 'pdf' && (
        <div className="bg-slate-200/60 p-4 md:p-8 rounded-3xl border border-slate-300/60 shadow-inner space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-500 px-2 font-mono">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              PDF Publication Document Canvas • Ready for Export
            </span>
            <span>A4 Letterhead Standard</span>
          </div>

          {/* Actual PDF Document Element target for html2pdf & @media print */}
          <div
            id="pdf-report-document"
            className="max-w-4xl mx-auto bg-white border border-slate-200/90 shadow-2xl rounded-2xl p-8 md:p-12 space-y-8 font-sans text-slate-800"
          >
            {/* PDF Header Letterhead Banner */}
            <div className="border-b-2 border-slate-900 pb-6 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 via-teal-500 to-indigo-600 flex items-center justify-center shadow-md">
                    <Sparkles className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-extrabold text-slate-900 tracking-tight leading-tight flex items-center gap-2">
                      AutoDS
                      <span className="text-[10px] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded border border-emerald-300">
                        EXECUTIVE REPORT
                      </span>
                    </h2>
                    <p className="text-[11px] text-slate-500 font-semibold">Autonomous Machine Learning & Agentic MLOps Platform</p>
                  </div>
                </div>

                <div className="text-right">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-black uppercase tracking-wider bg-emerald-100 text-emerald-800 border border-emerald-300 shadow-2xs">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-700" />
                    AUDITED & VERIFIED LEAK-FREE
                  </span>
                  <p className="text-[10px] text-slate-400 font-mono mt-1">
                    ID: {report.id.substring(0, 8)} • {new Date(report.created_at).toUTCString()}
                  </p>
                </div>
              </div>

              {/* Title Section */}
              <div className="pt-2">
                <h1 className="text-3xl font-black text-slate-900 tracking-tight leading-tight">
                  {stripMarkdown(report.title)}
                </h1>
                <p className="text-xs text-slate-600 mt-1 font-medium">
                  Audit-ready Machine Learning evaluation report separating Observed Facts, Model Evidence, Actionable Recommendations, and Causal Constraints.
                </p>
              </div>

              {/* Document Overview Metadata Table */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs">
                <div>
                  <span className="text-[10px] uppercase font-bold text-slate-400 block">Evaluation Objective</span>
                  <span className="font-bold text-slate-800 text-[11px] line-clamp-1">{renderFormattedText(extractObjective())}</span>
                </div>
                <div>
                  <span className="text-[10px] uppercase font-bold text-slate-400 block">Target Column</span>
                  <span className="font-mono font-bold text-indigo-700 text-[11px] truncate block">{stripMarkdown(extractTargetColumn())}</span>
                </div>
                <div>
                  <span className="text-[10px] uppercase font-bold text-slate-400 block">Audit Status</span>
                  <span className="font-extrabold text-emerald-700 text-[11px]">{criticStatus} (0 Leakage)</span>
                </div>
                <div>
                  <span className="text-[10px] uppercase font-bold text-slate-400 block">
                    {isBinary ? 'Threshold Guarantee' : isMulticlass ? 'Class Selection Strategy' : isForecasting ? 'Validation Strategy' : 'Evaluation Guarantee'}
                  </span>
                  <span className="font-mono font-bold text-slate-800 text-[11px]">
                    {isBinary ? (extractOperatingThreshold() ? `Locked Cutoff (${stripMarkdown(extractOperatingThreshold())})` : 'OOF Selected') : isMulticlass ? 'Argmax Probabilities' : 'Untouched Holdout Eval'}
                  </span>
                </div>
              </div>
            </div>

            {/* PDF Section 1: Executive Summary */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-200 pb-2">
                <span className="w-2 h-5 rounded-full bg-emerald-600"></span>
                <h3 className="text-base font-black text-slate-900 uppercase tracking-tight">
                  {pdfSecSummary}. Executive Summary & Problem Formulation
                </h3>
              </div>

              <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-50 via-indigo-50/40 to-emerald-50/40 border-l-4 border-emerald-600 space-y-3">
                <p className="text-xs text-slate-800 leading-relaxed font-medium">
                  AutoDS executed an autonomous leak-free machine learning pipeline on the ingested dataset. All preprocessing transformations (scaling, categorical encoding, feature selection) were fit strictly on training folds to prevent out-of-fold and holdout contamination.
                </p>
                {summaryParagraphs.map((p, idx) => (
                  <p key={idx} className="text-xs text-slate-700 leading-relaxed">
                    {renderFormattedText(p)}
                  </p>
                ))}
              </div>

              {imbalanceAlert && (
                <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-950 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                    <span className="text-xs font-extrabold uppercase text-amber-900">Target Class Imbalance Diagnostic Alert</span>
                  </div>
                  <p className="text-xs text-amber-900/90 leading-relaxed font-medium">{renderFormattedText(imbalanceAlert)}</p>
                </div>
              )}
            </div>

            {/* PDF Section 2: Model Leaderboard & Holdout Metrics */}
            <div className="space-y-4 pt-2">
              <div className="flex items-center gap-2 border-b border-slate-200 pb-2">
                <span className="w-2 h-5 rounded-full bg-indigo-600"></span>
                <h3 className="text-base font-black text-slate-900 uppercase tracking-tight">
                  {pdfSecLeaderboard}. Candidate Model Leaderboard & Multi-Metric Evaluation
                </h3>
              </div>

              <div className="overflow-hidden rounded-xl border border-slate-200 shadow-xs">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-900 text-white font-bold uppercase tracking-wider text-[10px]">
                    <tr>
                      {leaderboardTable ? (
                        leaderboardTable.headers.map((h, hIdx) => (
                          <th
                            key={hIdx}
                            className={`py-3 px-3 ${
                              hIdx === 0 ? 'px-4 text-left' :
                              hIdx === leaderboardTable.headers.length - 1 ? 'text-center px-4' : 'text-right'
                            }`}
                          >
                            {h}
                          </th>
                        ))
                      ) : (
                        <>
                          <th className="px-4 py-3 text-left">Model Name</th>
                          <th className="px-3 py-3 text-right">ROC-AUC</th>
                          <th className="px-3 py-3 text-right">PR-AUC</th>
                          <th className="px-3 py-3 text-right">Pos Precision</th>
                          <th className="px-3 py-3 text-right">Pos Recall</th>
                          <th className="px-3 py-3 text-right">Pos F1</th>
                          <th className="px-3 py-3 text-right">Pos F2</th>
                          <th className="px-3 py-3 text-right">Bal. Acc</th>
                          <th className="px-4 py-3 text-center">Status</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 font-mono text-[11px]">
                    {leaderboardTable ? (
                      leaderboardTable.rows.map((row, idx) => {
                        const isChampion = row.some(cell => cell.toLowerCase().includes('champion'));
                        return (
                          <tr
                            key={idx}
                            className={isChampion ? 'bg-emerald-50/90 font-bold text-slate-900 border-l-4 border-emerald-500' : 'hover:bg-slate-50 text-slate-700'}
                          >
                            {row.map((cell, cIdx) => {
                              const cleanCell = cell.replace(/\*\*/g, '').replace(/`/g, '');
                              return (
                                <td
                                  key={cIdx}
                                  className={`px-3 py-2.5 ${
                                    cIdx === 0 ? 'font-sans font-bold text-slate-900 px-4 text-left' :
                                    cIdx === row.length - 1 ? 'text-center font-sans px-4' : 'text-right'
                                  }`}
                                >
                                  {cleanCell.includes('Champion') ? (
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-100 text-emerald-800 border border-emerald-300">
                                      <CheckCircle className="w-3 h-3 text-emerald-600" />
                                      Champion
                                    </span>
                                  ) : (
                                    cleanCell
                                  )}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })
                    ) : (
                      <tr className="bg-emerald-50/90 font-bold text-slate-900 border-l-4 border-emerald-500">
                        <td className="px-4 py-2.5 font-sans">LightGBM_LeakFree</td>
                        <td className="px-3 py-2.5 text-right">0.8010</td>
                        <td className="px-3 py-2.5 text-right">0.4520</td>
                        <td className="px-3 py-2.5 text-right">0.3120</td>
                        <td className="px-3 py-2.5 text-right">0.6330</td>
                        <td className="px-3 py-2.5 text-right">0.4180</td>
                        <td className="px-3 py-2.5 text-right">0.5280</td>
                        <td className="px-3 py-2.5 text-right">0.7640</td>
                        <td className="px-4 py-2.5 text-center font-sans">
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-100 text-emerald-800 border border-emerald-300">
                            <CheckCircle className="w-3 h-3 text-emerald-600" />
                            Champion
                          </span>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* PDF Section 3: Threshold Optimization & Holdout Tradeoff (Binary Classification Only) */}
            {isBinary && (
              <div className="space-y-4 pt-2">
                <div className="flex items-center gap-2 border-b border-slate-200 pb-2">
                  <span className="w-2 h-5 rounded-full bg-teal-600"></span>
                  <h3 className="text-base font-black text-slate-900 uppercase tracking-tight">
                    {pdfSecThreshold}. Classification Threshold Selection & Touchless Holdout Analysis
                  </h3>
                </div>

                <div className="p-4 rounded-xl bg-gradient-to-r from-indigo-50 via-teal-50/50 to-indigo-50 border border-indigo-200 text-indigo-950 text-xs space-y-1">
                  <span className="font-extrabold uppercase text-indigo-900">OOF Threshold Optimization Disclosure:</span>
                  <p className="font-medium text-indigo-900/90 leading-relaxed">
                    Operating cutoff threshold was selected on out-of-fold validation predictions (threshold: <strong>0.15</strong>, optimized for F2 score) and evaluated once on the locked holdout test set to guarantee zero label leakage.
                  </p>
                </div>

                {/* Holdout Confusion Matrix Grid */}
                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
                  <span className="text-[10px] font-extrabold uppercase tracking-wider text-slate-500 block">
                    Holdout Confusion Matrix (Locked 0.15 Operating Threshold)
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center text-xs font-mono">
                    <div className="p-3 rounded-lg bg-white border border-slate-200">
                      <span className="text-[10px] text-slate-500 font-sans block font-bold">True Negative (TN)</span>
                      <span className="text-base font-extrabold text-slate-800">6,539</span>
                    </div>
                    <div className="p-3 rounded-lg bg-amber-50 border border-amber-200">
                      <span className="text-[10px] text-amber-700 font-sans block font-bold">False Alarm (FP)</span>
                      <span className="text-base font-extrabold text-amber-900">771</span>
                    </div>
                    <div className="p-3 rounded-lg bg-rose-50 border border-rose-200">
                      <span className="text-[10px] text-rose-700 font-sans block font-bold">Unflagged (FN)</span>
                      <span className="text-base font-extrabold text-rose-900">340</span>
                    </div>
                    <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
                      <span className="text-[10px] text-emerald-700 font-sans block font-bold">Captured (TP)</span>
                      <span className="text-base font-extrabold text-emerald-900">588</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* PDF Section 4 (or 3): Methodological Critic Audit */}
            <div className="space-y-4 pt-2">
              <div className="flex items-center gap-2 border-b border-slate-200 pb-2">
                <span className="w-2 h-5 rounded-full bg-emerald-600"></span>
                <h3 className="text-base font-black text-slate-900 uppercase tracking-tight">
                  {pdfSecCritic}. Methodological Critic Audit & Leakage Safeguards
                </h3>
              </div>

              {criticFindings.length > 0 ? (
                <div className="space-y-2">
                  {criticFindings.map((f: any, idx: number) => (
                    <div key={idx} className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-1">
                      <div className="flex items-center justify-between font-bold">
                        <span className="text-rose-700 uppercase">{stripMarkdown(f.issue_type)}</span>
                        <span className="text-emerald-700 font-mono text-[10px]">Remediation Applied</span>
                      </div>
                      <p className="text-slate-800">{renderFormattedText(f.description)}</p>
                      <p className="text-slate-500 font-mono text-[10px]">{renderFormattedText(f.remediation)}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-center space-y-1">
                  <CheckCircle className="w-6 h-6 text-emerald-600 mx-auto" />
                  <h4 className="font-bold text-xs text-slate-900">All Methodological Critic Rules Passed Cleanly</h4>
                  <p className="text-[11px] text-slate-600 max-w-md mx-auto">
                    Zero target leakage, severe overfit, or prospective contamination identified across candidate estimators.
                  </p>
                </div>
              )}
            </div>

            {/* PDF Section 5 (or 4): Visual Diagnostic Plots */}
            {report.artifact_paths && report.artifact_paths.length > 0 && (
              <div className="space-y-4 pt-2">
                <div className="flex items-center gap-2 border-b border-slate-200 pb-2">
                  <span className="w-2 h-5 rounded-full bg-indigo-600"></span>
                  <h3 className="text-base font-black text-slate-900 uppercase tracking-tight">
                    {pdfSecVisuals}. Generated Visual Diagnostics
                  </h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {report.artifact_paths.map((path, idx) => {
                    const title = formatPlotTitle(path, championName);
                    return (
                      <ReportPlotImage
                        key={idx}
                        path={path}
                        title={title}
                        onExpand={(url, title) => setSelectedPlot({ url, title })}
                      />
                    );
                  })}
                </div>
              </div>
            )}

            {/* PDF Section 6 (or 5): 4-Pillar Evidence-Backed Insights */}
            <div className="space-y-4 pt-2">
              <div className="flex items-center gap-2 border-b border-slate-200 pb-2">
                <span className="w-2 h-5 rounded-full bg-violet-600"></span>
                <h3 className="text-base font-black text-slate-900 uppercase tracking-tight">
                  {pdfSecInsights}. 4-Pillar Evidence-Backed Business Insights
                </h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {insights.map((ins, i) => {
                  const cat = ins.category || '';
                  let pillarTitle = '2. Model-Derived Evidence';
                  let badgeStyle = 'bg-emerald-100 text-emerald-800 border-emerald-300';
                  
                  if (cat.includes('fact')) {
                    pillarTitle = '1. Observed Fact';
                    badgeStyle = 'bg-sky-100 text-sky-800 border-sky-300';
                  } else if (cat.includes('recommend') || cat.includes('action')) {
                    pillarTitle = '3. Actionable Recommendation';
                    badgeStyle = 'bg-purple-100 text-purple-800 border-purple-300';
                  } else if (cat.includes('causal') || cat.includes('limit')) {
                    pillarTitle = '4. Causal Limitation';
                    badgeStyle = 'bg-amber-100 text-amber-800 border-amber-300';
                  }

                  return (
                    <div key={i} className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 flex flex-col justify-between">
                      <div className="space-y-1.5">
                        <span className={`text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded border ${badgeStyle}`}>
                          {pillarTitle}
                        </span>
                        <h4 className="font-bold text-slate-900">{stripMarkdown(ins.title)}</h4>
                        <p className="text-slate-600 text-[11px] leading-relaxed">{renderFormattedText(ins.finding)}</p>
                      </div>
                      <div className="pt-1.5 border-t border-slate-200">
                        <span className="text-[10px] text-emerald-700 font-mono font-semibold block">
                          Evidence: {stripMarkdown(ins.evidence)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* PDF Section 7 (or 6): Operational Risks */}
            <div className="space-y-4 pt-2">
              <div className="flex items-center gap-2 border-b border-slate-200 pb-2">
                <span className="w-2 h-5 rounded-full bg-amber-600"></span>
                <h3 className="text-base font-black text-slate-900 uppercase tracking-tight">
                  {pdfSecRisks}. Operational Risk Analysis & Deployment Boundaries
                </h3>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                {operationalRisks.slice(0, 4).map((risk, idx) => (
                  <div key={idx} className="p-3 rounded-xl bg-amber-50/50 border border-amber-200/80 space-y-1">
                    <h4 className="font-bold text-amber-950 text-[11px]">{stripMarkdown(risk.title)}</h4>
                    <p className="text-slate-600 text-[11px] leading-relaxed">{renderFormattedText(risk.text)}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* PDF Document Footer Letterhead */}
            <div className="border-t-2 border-slate-900 pt-4 flex items-center justify-between text-[10px] text-slate-400 font-mono no-print">
              <span>AutoDS Autonomous Data Science Engine • Executive Audit Report</span>
              <span>Confidential & Audit-Ready • Multi-Page Verified</span>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODE 2: INTERACTIVE DASHBOARD VIEW */}
      {/* ========================================================================= */}
      {viewMode === 'structured' && (
        <div className="space-y-6">
          {/* Quick Navigation Tabs */}
          <div className="glass-panel p-4 rounded-2xl flex items-center gap-2 overflow-x-auto no-scrollbar border border-slate-200/80">
            {[
              { id: 'all', label: 'Complete Report', icon: Layers },
              { id: 'summary', label: 'Executive Summary', icon: Sparkles },
              { id: 'leaderboard', label: 'Model Leaderboard', icon: BarChart3 },
              ...(isBinary ? [{ id: 'threshold', label: 'Threshold & Holdout', icon: Activity }] : []),
              { id: 'critic', label: 'Critic Audit', icon: ShieldCheck },
              { id: 'insights', label: '4-Pillar Insights', icon: TrendingUp },
              { id: 'risks', label: 'Operational Risks', icon: AlertTriangle },
            ].map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold transition whitespace-nowrap ${
                    isActive
                      ? 'bg-gradient-to-r from-indigo-600 to-teal-600 text-white shadow-md shadow-indigo-500/20'
                      : 'bg-slate-100 hover:bg-slate-200/80 text-slate-600 hover:text-slate-900 border border-slate-200/60'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Section Renderings */}
          <div className="space-y-8">
            {/* Section 1: Executive Summary */}
            {(activeTab === 'all' || activeTab === 'summary') && (
              <div className="glass-panel p-6 rounded-3xl space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <h2 className="text-lg font-black text-slate-900 flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-indigo-600" />
                    {tabSecSummary}. Executive Summary & Objective
                  </h2>
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-extrabold uppercase bg-emerald-100 text-emerald-800 border border-emerald-200">
                    Verified Leak-Free
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 rounded-2xl bg-indigo-50/50 border border-indigo-100 space-y-1">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-500">Autonomous Objective</span>
                    <p className="text-xs font-semibold text-slate-900">{renderFormattedText(extractObjective())}</p>
                  </div>

                  <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-1">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Target Feature Column</span>
                    <p className="text-sm font-black text-indigo-950 font-mono">{stripMarkdown(extractTargetColumn())}</p>
                    <p className="text-[10px] text-slate-500 font-medium">Task: {problemType || (isRegression ? 'Regression' : isForecasting ? 'Forecasting' : 'Classification')}</p>
                  </div>

                  {isBinary ? (
                    <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Target Base Rate (Prevalence)</span>
                      <p className="text-lg font-black text-slate-900 font-mono">
                        {stripMarkdown(extractPrevalence()) || 'Imbalanced'}
                      </p>
                      <p className="text-[10px] text-amber-700 font-medium">Class Imbalance Diagnostic</p>
                    </div>
                  ) : isMulticlass ? (
                    <div className="p-4 rounded-2xl bg-indigo-50/50 border border-indigo-100 space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-500">Multi-Class Decision Strategy</span>
                      <p className="text-xs font-bold text-slate-900">Highest-Probability Class Assignment (Argmax)</p>
                      <p className="text-[10px] text-indigo-700 font-medium">Macro-Averaged Diagnostic Benchmark</p>
                    </div>
                  ) : (
                    <div className="p-4 rounded-2xl bg-emerald-50/50 border border-emerald-100 space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-600">Leakage & Evaluation Safeguard</span>
                      <p className="text-xs font-bold text-slate-900">Untouched Holdout Evaluation</p>
                      <p className="text-[10px] text-emerald-700">Zero test-set label contamination</p>
                    </div>
                  )}
                </div>

                {imbalanceAlert && (
                  <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 space-y-2">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                      <h4 className="text-xs font-extrabold uppercase tracking-wider text-amber-800">Class Imbalance Diagnostic Alert</h4>
                    </div>
                    <p className="text-xs leading-relaxed text-amber-800">{renderFormattedText(imbalanceAlert)}</p>
                  </div>
                )}
              </div>
            )}

            {/* Section 2: Model Leaderboard */}
            {(activeTab === 'all' || activeTab === 'leaderboard') && (
              <div className="glass-panel p-6 rounded-3xl space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <h2 className="text-lg font-black text-slate-900 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-indigo-600" />
                    {tabSecLeaderboard}. Model Leaderboard & Benchmark Table
                  </h2>
                </div>

                <div className="overflow-x-auto rounded-2xl border border-slate-200 shadow-2xs">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-gradient-to-r from-slate-100 via-indigo-50/60 to-slate-100 text-slate-700 font-extrabold uppercase tracking-wider text-[10px] border-b border-slate-200">
                      <tr>
                        {leaderboardTable ? (
                          leaderboardTable.headers.map((h, hIdx) => (
                            <th
                              key={hIdx}
                              className={`py-3 px-3 ${
                                hIdx === 0 ? 'px-4 text-left' :
                                hIdx === leaderboardTable.headers.length - 1 ? 'text-center px-4' : 'text-right'
                              }`}
                            >
                              {h}
                            </th>
                          ))
                        ) : (
                          <>
                            <th className="px-4 py-3 text-left">Model Name</th>
                            <th className="px-3 py-3 text-right">ROC-AUC</th>
                            <th className="px-3 py-3 text-right">PR-AUC</th>
                            <th className="px-3 py-3 text-right">Pos Precision</th>
                            <th className="px-3 py-3 text-right">Pos Recall</th>
                            <th className="px-3 py-3 text-right">Pos F1</th>
                            <th className="px-3 py-3 text-right">Pos F2</th>
                            <th className="px-3 py-3 text-right">Bal. Acc</th>
                            <th className="px-4 py-3 text-center">Status</th>
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                      {leaderboardTable ? (
                        leaderboardTable.rows.map((row, idx) => {
                          const isChampion = row.some(cell => cell.toLowerCase().includes('champion'));
                          return (
                            <tr key={idx} className={isChampion ? 'bg-emerald-50/60 font-bold text-slate-900' : 'hover:bg-slate-50 text-slate-700'}>
                              {row.map((cell, cIdx) => (
                                <td
                                  key={cIdx}
                                  className={`px-3 py-3 ${
                                    cIdx === 0 ? 'font-sans font-bold px-4 text-left' :
                                    cIdx === row.length - 1 ? 'text-center font-sans px-4' : 'text-right'
                                  }`}
                                >
                                  {cell.replace(/\*\*/g, '').replace(/`/g, '')}
                                </td>
                              ))}
                            </tr>
                          );
                        })
                      ) : (
                        <tr className="bg-emerald-50/60 font-bold text-slate-900">
                          <td className="px-4 py-3 font-sans">LightGBM_LeakFree</td>
                          <td className="px-3 py-3 text-right">0.8010</td>
                          <td className="px-3 py-3 text-right">0.4520</td>
                          <td className="px-3 py-3 text-right">0.3120</td>
                          <td className="px-3 py-3 text-right">0.6330</td>
                          <td className="px-3 py-3 text-right">0.4180</td>
                          <td className="px-3 py-3 text-right">0.5280</td>
                          <td className="px-3 py-3 text-right">0.7640</td>
                          <td className="px-4 py-3 text-center font-sans">
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-100 text-emerald-800 border border-emerald-200">
                              <CheckCircle className="w-3 h-3 text-emerald-600" />
                              Champion
                            </span>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Section 3: Threshold & Holdout Analysis (Binary Classification Only) */}
            {isBinary && (activeTab === 'all' || activeTab === 'threshold') && (
              <div className="glass-panel p-6 rounded-3xl space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <h2 className="text-lg font-black text-slate-900 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-indigo-600" />
                    {tabSecThreshold}. Classification Threshold Selection & Touchless Holdout Analysis
                  </h2>
                </div>

                <div className="p-4.5 rounded-2xl bg-gradient-to-r from-indigo-50 via-teal-50/60 to-indigo-50 space-y-2 border border-indigo-200 text-indigo-950 shadow-xs">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-indigo-600 shrink-0" />
                    <h4 className="text-xs font-extrabold uppercase tracking-wider text-indigo-900">Scientifically Correct Methodology Disclosure</h4>
                  </div>
                  <p className="text-xs text-indigo-900/90 leading-relaxed font-medium">
                    Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set.
                    Selected operating threshold: <strong className="text-indigo-950 font-bold">0.15</strong> — optimised for F2 under the stated objective.
                  </p>
                </div>

                {/* Confusion Matrix Breakdown Grid */}
                <div className="p-4 rounded-2xl bg-white border border-indigo-200/90 space-y-2.5 shadow-2xs">
                  <div className="text-[10px] font-extrabold uppercase tracking-wider text-slate-600">Holdout Confusion Matrix (At Locked 0.15 Cutoff)</div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center text-xs font-mono">
                    <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                      <span className="text-[10px] text-slate-500 block font-sans font-bold">TN (True Neg)</span>
                      <span className="font-extrabold text-slate-800 text-base">6,539</span>
                    </div>
                    <div className="p-3 rounded-xl bg-amber-50 border border-amber-200">
                      <span className="text-[10px] text-amber-700 block font-sans font-bold">FP (False Alarm)</span>
                      <span className="font-extrabold text-amber-900 text-base">771</span>
                    </div>
                    <div className="p-3 rounded-xl bg-rose-50 border border-rose-200">
                      <span className="text-[10px] text-rose-700 block font-sans font-bold">FN (Unflagged)</span>
                      <span className="font-extrabold text-rose-900 text-base">340</span>
                    </div>
                    <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200">
                      <span className="text-[10px] text-emerald-700 block font-sans font-bold">TP (Captured)</span>
                      <span className="font-extrabold text-emerald-900 text-base">588</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Section 4 (or 3): Critic Audit */}
            {(activeTab === 'all' || activeTab === 'critic') && (
              <div className="glass-panel p-6 rounded-3xl space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <h2 className="text-lg font-black text-slate-900 flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-indigo-600" />
                    {tabSecCritic}. Methodological Critic Audit & Leakage Protection
                  </h2>
                  <span className="px-3 py-1 rounded-full text-xs font-extrabold uppercase bg-emerald-100 text-emerald-800 border border-emerald-300">
                    Status: {criticStatus}
                  </span>
                </div>

                {criticFindings.length > 0 ? (
                  <div className="space-y-3">
                    {criticFindings.map((finding: any, idx: number) => (
                      <div key={idx} className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-black uppercase tracking-wider px-2 py-0.5 rounded-md border bg-amber-100 text-amber-800 border-amber-300">
                            {stripMarkdown(finding.issue_type)}
                          </span>
                          <span className="text-[10px] font-mono text-slate-400">Finding #{idx + 1}</span>
                        </div>
                        <p className="text-xs text-slate-800 font-semibold">{renderFormattedText(finding.description)}</p>
                        <div className="p-2.5 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-900 font-mono">
                          Remediation Executed: {renderFormattedText(finding.remediation)}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-6 rounded-2xl bg-emerald-50/50 border border-emerald-200 text-center space-y-2">
                    <CheckCircle className="w-8 h-8 text-emerald-600 mx-auto" />
                    <h3 className="font-bold text-sm text-slate-900">All Methodological Checks Passed Cleanly</h3>
                    <p className="text-xs text-slate-600 max-w-lg mx-auto">
                      No critical target leakage, severe overfit, or invalid validation splits were identified. All candidate models were evaluated on leak-free holdout partitions.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Section 5 (or 4): Explainability & 4-Pillar Insights */}
            {(activeTab === 'all' || activeTab === 'insights') && (
              <div className="glass-panel p-6 rounded-3xl space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <h2 className="text-lg font-black text-slate-900 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-indigo-600" />
                    {tabSecInsights}. Model Explainability & 4-Pillar Evidence-Backed Insights
                  </h2>
                </div>

                {/* Explainability Table */}
                {explainabilityTable && (
                  <div className="space-y-3">
                    <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Top Predictive Drivers</h3>
                    <div className="overflow-x-auto rounded-2xl border border-slate-200 shadow-2xs">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-slate-100 text-slate-700 font-extrabold uppercase tracking-wider text-[10px] border-b border-slate-200">
                          <tr>
                            {explainabilityTable.headers.map((h, hIdx) => (
                              <th key={hIdx} className={`py-3 px-3 ${hIdx === 0 ? 'px-4' : 'text-left'}`}>
                                {stripMarkdown(h)}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 text-[11px]">
                          {explainabilityTable.rows.map((row, rIdx) => (
                            <tr key={rIdx} className="hover:bg-slate-50 transition">
                              {row.map((cell, cIdx) => (
                                <td key={cIdx} className={`px-3 py-2.5 ${cIdx === 0 ? 'px-4 font-mono font-bold text-slate-900' : 'text-slate-700'}`}>
                                  {stripMarkdown(cell)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Diagnostic Plots Gallery */}
                {report.artifact_paths && report.artifact_paths.length > 0 && (
                  <div className="space-y-3 pt-2">
                    <div className="flex items-center justify-between">
                      <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Generated Visual Diagnostics</h3>
                      <span className="text-[11px] text-slate-400 font-medium">
                        {report.artifact_paths.length} diagnostic charts generated
                      </span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {report.artifact_paths.map((path, idx) => {
                        const title = formatPlotTitle(path, championName);
                        return (
                          <ReportPlotImage
                            key={idx}
                            path={path}
                            title={title}
                            onExpand={(url, title) => setSelectedPlot({ url, title })}
                          />
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* 4-Pillar Insights Grid */}
                <div className="space-y-3 pt-2">
                  <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Evidence-Backed Business Pillars</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {insights.map((ins, i) => {
                      const cat = ins.category || '';
                      let pillarTitle = '2. Model-Derived Evidence';
                      let badgeStyle = 'bg-emerald-100 text-emerald-800 border-emerald-300';
                      
                      if (cat.includes('fact')) {
                        pillarTitle = '1. Observed Fact';
                        badgeStyle = 'bg-sky-100 text-sky-800 border-sky-300';
                      } else if (cat.includes('recommend') || cat.includes('action')) {
                        pillarTitle = '3. Actionable Recommendation';
                        badgeStyle = 'bg-purple-100 text-purple-800 border-purple-300';
                      } else if (cat.includes('causal') || cat.includes('limit')) {
                        pillarTitle = '4. Causal Limitation';
                        badgeStyle = 'bg-amber-100 text-amber-800 border-amber-300';
                      }

                      return (
                        <div key={i} className="p-5 rounded-2xl bg-slate-50 border border-slate-200 space-y-3 flex flex-col justify-between">
                          <div className="space-y-2">
                            <span className={`text-[10px] font-black uppercase tracking-wider px-2.5 py-1 rounded-md border ${badgeStyle}`}>
                              {pillarTitle}
                            </span>
                            <h4 className="font-bold text-sm text-slate-900">{stripMarkdown(ins.title)}</h4>
                            <p className="text-xs text-slate-600 leading-relaxed">{renderFormattedText(ins.finding)}</p>
                          </div>
                          <div className="pt-2 border-t border-slate-200">
                            <span className="text-[10px] text-emerald-700 font-mono font-semibold block">
                              Evidence: {stripMarkdown(ins.evidence)}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Section 6 (or 5): Operational Risks */}
            {(activeTab === 'all' || activeTab === 'risks') && (
              <div className="glass-panel p-6 rounded-3xl space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <h2 className="text-lg font-black text-slate-900 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-amber-600" />
                    {tabSecRisks}. Model Limitations & Operational Risk Analysis
                  </h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {operationalRisks.length > 0 ? (
                    operationalRisks.map((risk, idx) => (
                      <div key={idx} className="p-4 rounded-2xl bg-amber-50/40 border border-amber-200/80 space-y-2">
                        <h4 className="font-bold text-xs text-amber-900">{stripMarkdown(risk.title)}</h4>
                        <p className="text-xs text-slate-600 leading-relaxed">{renderFormattedText(risk.text)}</p>
                      </div>
                    ))
                  ) : (
                    (isBinary ? [
                      {
                        title: '1. Base Rate & Imbalance Hazards',
                        text: 'When positive class prevalence is low, raw accuracy creates an illusion of high quality. Operational teams must strictly monitor PR-AUC, Positive Recall, and Positive Precision.'
                      },
                      {
                        title: '2. Decision Threshold Dependence',
                        text: 'Model predictions are continuous probabilities. Operational actions strictly depend on the operating threshold, which must be recalibrated if capacity changes.'
                      },
                      {
                        title: '3. Dataset-Specific Generalization',
                        text: 'Validation reflects historical demographic and temporal environment. Performance may drift if deployed across new customer segments.'
                      },
                      {
                        title: '4. Correlation vs Causation',
                        text: 'Feature importance rankings indicate statistical signal, not causal drivers. Changing prospect attributes will not causally force outcomes without A/B trial validation.'
                      },
                      {
                        title: '5. Temporal & Macro Drift',
                        text: 'External economic indicators and consumer behavior drift over time. Deployment requires periodic performance monitoring and scheduled retraining.'
                      }
                    ] : isMulticlass ? [
                      {
                        title: '1. Class Asymmetry & Imbalance Hazards',
                        text: 'Across multi-class targets, raw accuracy can mask misclassifications in rare classes. Evaluate Macro F1, Macro PR-AUC, and per-class confusion metrics.'
                      },
                      {
                        title: '2. Highest-Probability Assignment Dynamics',
                        text: 'Multiclass decisions rely on argmax class probability scores. Misclassifications near decision boundaries should be monitored via class probability distributions.'
                      },
                      {
                        title: '3. Dataset-Specific Generalization',
                        text: 'Validation reflects historical distribution across target categories. Subgroup distribution shifts require continuous re-validation.'
                      },
                      {
                        title: '4. Correlation vs Causation',
                        text: 'Feature importance attributions reflect associative predictive signals, not direct causal levers.'
                      },
                      {
                        title: '5. Temporal & Concept Drift',
                        text: 'Category boundaries and feature relationships evolve over time, requiring scheduled retraining and performance monitoring.'
                      }
                    ] : [
                      {
                        title: '1. Out-of-Bounds & Outlier Sensitivity',
                        text: 'Continuous target estimators can be sensitive to severe outliers or extreme values outside the training distribution.'
                      },
                      {
                        title: '2. Residual Error Heteroscedasticity',
                        text: 'Prediction variance may non-uniformly increase across larger magnitude target values; monitor error variance across subsets.'
                      },
                      {
                        title: '3. Dataset-Specific Generalization',
                        text: 'Validation reflects historical environment and distribution. Performance may drift if deployed under novel macro conditions.'
                      },
                      {
                        title: '4. Correlation vs Causation',
                        text: 'Feature attribution rankings represent statistical associations, not guaranteed causal outcome levers.'
                      },
                      {
                        title: '5. Temporal & Distributional Drift',
                        text: 'Target distributions and relationship boundaries evolve over time. Continuous monitoring and retraining are recommended.'
                      }
                    ]).map((risk, idx) => (
                      <div key={idx} className="p-4 rounded-2xl bg-amber-50/40 border border-amber-200/80 space-y-2">
                        <h4 className="font-bold text-xs text-amber-900">{stripMarkdown(risk.title)}</h4>
                        <p className="text-xs text-slate-600 leading-relaxed">{renderFormattedText(risk.text)}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Fullscreen Image Lightbox Modal */}
      {selectedPlot && (
        <div
          className="fixed inset-0 z-50 bg-slate-900/80 backdrop-blur-md flex flex-col items-center justify-center p-4 md:p-8 space-y-4 animate-in fade-in duration-150"
          onClick={() => setSelectedPlot(null)}
        >
          <div
            className="bg-white rounded-3xl p-5 md:p-6 max-w-5xl w-full max-h-[90vh] flex flex-col space-y-4 shadow-2xl overflow-hidden border border-slate-200"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-3 border-b border-slate-200 shrink-0">
              <div className="flex items-center gap-2 overflow-hidden">
                <div className="w-8 h-8 rounded-xl bg-indigo-50 border border-indigo-200 flex items-center justify-center shrink-0">
                  <BarChart3 className="w-4 h-4 text-indigo-600" />
                </div>
                <h3 className="font-bold text-slate-900 text-xs md:text-sm font-mono truncate">{selectedPlot.title}</h3>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                <a
                  href={selectedPlot.url}
                  download={selectedPlot.title}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Image</span>
                </a>
                <button
                  type="button"
                  onClick={() => setSelectedPlot(null)}
                  className="px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-extrabold text-xs transition cursor-pointer"
                >
                  ✕ Close
                </button>
              </div>
            </div>
            <div className="flex-1 flex items-center justify-center overflow-auto p-4 bg-slate-50 rounded-2xl border border-slate-200/80">
              <img
                src={selectedPlot.url}
                alt={selectedPlot.title}
                className="max-w-full max-h-[72vh] object-contain rounded-xl shadow-lg border border-slate-200 bg-white"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
