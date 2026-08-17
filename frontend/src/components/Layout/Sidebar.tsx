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
  ShieldCheck
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
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col h-screen sticky top-0 shrink-0 select-none shadow-sm">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-100 flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 via-teal-500 to-indigo-600 flex items-center justify-center shadow-md shadow-emerald-500/20">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="font-extrabold text-lg text-slate-900 tracking-tight leading-tight flex items-center gap-1.5">
            AutoDS
            <span className="text-[10px] uppercase font-bold bg-emerald-50 text-emerald-700 px-1.5 py-0.5 rounded border border-emerald-200">
              v0.1
            </span>
          </h1>
          <p className="text-xs text-slate-500 font-medium">Autonomous Data Science</p>
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
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-indigo-50/90 text-indigo-700 border border-indigo-100 font-semibold shadow-xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
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
      <div className="p-3.5 m-3 rounded-2xl bg-slate-50 border border-slate-200/80 text-xs">
        <div className="flex items-center justify-between mb-2">
          <span className="text-slate-500 font-medium">Engine Status</span>
          <span className="flex items-center gap-1.5 text-emerald-600 font-bold">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            Online
          </span>
        </div>
        
        <div className="space-y-1.5 text-[11px] text-slate-600">
          <div className="flex items-center justify-between">
            <span>Gemini AI:</span>
            <span className={health?.gemini_api_configured ? "text-emerald-700 font-semibold" : "text-amber-700 font-semibold"}>
              {health?.gemini_api_configured ? "Active" : "Deterministic Mode"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>MLflow Tracking:</span>
            <span className="text-emerald-700 font-semibold">Active</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Critic Audit:</span>
            <span className="text-indigo-700 font-semibold flex items-center gap-0.5">
              <ShieldCheck className="w-3 h-3" /> Enforced
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
