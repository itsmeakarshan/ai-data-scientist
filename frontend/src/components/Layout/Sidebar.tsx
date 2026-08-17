import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Database,
  PlayCircle,
  FlaskConical,
  Boxes,
  FileText,
  MessageSquare,
  Settings as SettingsIcon,
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { HealthStatus } from '../../types';

interface SidebarProps {
  health?: HealthStatus | null;
}

export const Sidebar: React.FC<SidebarProps> = ({ health }) => {
  const navItems = [
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/datasets', label: 'Datasets', icon: Database },
    { to: '/analysis', label: 'Autonomous DS', icon: PlayCircle },
    { to: '/experiments', label: 'Experiments', icon: FlaskConical },
    { to: '/models', label: 'Model Registry', icon: Boxes },
    { to: '/reports', label: 'Reports', icon: FileText },
    { to: '/chat', label: 'Agent Chat', icon: MessageSquare },
    { to: '/settings', label: 'Settings', icon: SettingsIcon },
  ];

  return (
    <aside className="w-64 bg-slate-900/90 border-r border-slate-800 flex flex-col h-screen sticky top-0 shrink-0 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800/80 flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 via-teal-500 to-indigo-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="font-bold text-lg text-slate-100 tracking-tight leading-tight flex items-center gap-1.5">
            AutoDS
            <span className="text-[10px] uppercase font-bold bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-500/30">
              v0.1
            </span>
          </h1>
          <p className="text-xs text-slate-400 font-medium">Autonomous Data Science</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* System Status Banner */}
      <div className="p-3.5 m-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs">
        <div className="flex items-center justify-between mb-2">
          <span className="text-slate-400 font-medium">Engine Status</span>
          <span className="flex items-center gap-1 text-emerald-400 font-semibold">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            Online
          </span>
        </div>
        
        <div className="space-y-1.5 text-[11px] text-slate-400">
          <div className="flex items-center justify-between">
            <span>Gemini AI:</span>
            <span className={health?.gemini_api_configured ? "text-emerald-400" : "text-amber-400 font-medium"}>
              {health?.gemini_api_configured ? "Active" : "Deterministic Mode"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>MLflow Tracking:</span>
            <span className="text-emerald-400">Active</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Critic Audit:</span>
            <span className="text-indigo-400 flex items-center gap-0.5">
              <ShieldCheck className="w-3 h-3" /> Enforced
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
