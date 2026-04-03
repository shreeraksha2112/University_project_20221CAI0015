import React from 'react';
import { BroadcastIcon, WarningIcon, PulseIcon } from '@phosphor-icons/react';

const Dashboard = ({ stats }) => {
  const getRiskColor = (level) => {
    switch (level) {
      case 'LOW':
        return 'text-[#10B981]';
      case 'MEDIUM':
        return 'text-[#F59E0B]';
      case 'HIGH':
        return 'text-[#EF4444]';
      default:
        return 'text-slate-400';
    }
  };

  const getRiskGlow = (level) => {
    switch (level) {
      case 'LOW':
        return 'risk-glow-low';
      case 'MEDIUM':
        return 'risk-glow-medium';
      case 'HIGH':
        return 'risk-glow-high';
      default:
        return '';
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg sm:text-xl font-medium tracking-wide uppercase text-slate-400" style={{ fontFamily: 'Chivo, sans-serif' }}>
        Mission Control
      </h3>

      {/* Satellite Count */}
      <div className="bg-[#090A0F] border border-[#1E2330] p-4 rounded-sm" data-testid="satellite-count-widget">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Satellites</span>
          <BroadcastIcon size={20} weight="duotone" className="text-[#06B6D4]" />
        </div>
        <div className="metric-display text-slate-100" data-testid="satellite-count">
          {stats.satellite_count}
        </div>
      </div>

      {/* Debris Count */}
      <div className="bg-[#090A0F] border border-[#1E2330] p-4 rounded-sm" data-testid="debris-count-widget">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Debris</span>
          <WarningIcon size={20} weight="duotone" className="text-[#F59E0B]" />
        </div>
        <div className="metric-display text-slate-100" data-testid="debris-count">
          {stats.debris_count}
        </div>
      </div>

      {/* Risk Level */}
      <div className={`bg-[#090A0F] border border-[#1E2330] p-4 rounded-sm ${getRiskGlow(stats.risk_level)}`} data-testid="risk-level-widget">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Risk Level</span>
          <PulseIcon size={20} weight="duotone" className={getRiskColor(stats.risk_level)} />
        </div>
        <div className={`text-3xl font-black font-mono tracking-tight ${getRiskColor(stats.risk_level)}`} data-testid="risk-level">
          {stats.risk_level}
        </div>
        <div className="mt-2 h-2 bg-[#1E2330] rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${stats.risk_level === 'HIGH' ? 'bg-[#EF4444]' : stats.risk_level === 'MEDIUM' ? 'bg-[#F59E0B]' : 'bg-[#10B981]'}`}
            style={{ width: `${stats.max_risk * 100}%` }}
          />
        </div>
      </div>

      {/* Additional Stats */}
      <div className="bg-[#090A0F] border border-[#1E2330] p-4 rounded-sm">
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Collisions</span>
            <span className="text-xl font-mono font-bold text-slate-300" data-testid="collision-count">{stats.total_collisions}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Uptime</span>
            <span className="text-xl font-mono font-bold text-slate-300">{stats.uptime}s</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Frame</span>
            <span className="text-xl font-mono font-bold text-slate-300">{stats.frame_count}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
