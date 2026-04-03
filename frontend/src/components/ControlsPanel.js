import React from 'react';
import { PlayIcon, PauseIcon, ArrowsClockwiseIcon, CaretUpIcon, CaretDownIcon } from '@phosphor-icons/react';

const ControlsPanel = ({ stats, predictorMode, onTogglePause, onSpeedAdjust, onReset, onPredictorModeChange }) => {
  return (
    <div className="bg-[#090A0F] border border-[#1E2330] p-4 rounded-sm" data-testid="controls-panel">
      <h3 className="text-lg sm:text-xl font-medium tracking-wide uppercase text-slate-400 mb-4" style={{ fontFamily: 'Chivo, sans-serif' }}>
        Controls
      </h3>

      {/* Main Controls */}
      <div className="space-y-3 mb-6">
        <button
          onClick={onTogglePause}
          className="w-full bg-[#3B82F6] hover:bg-[#2563EB] text-white font-bold py-3 px-4 border border-[#1E2330] transition-colors flex items-center justify-center gap-2"
          data-testid="pause-resume-button"
        >
          {stats.is_paused ? <PlayIcon size={20} weight="fill" /> : <PauseIcon size={20} weight="fill" />}
          {stats.is_paused ? 'RESUME' : 'PAUSE'}
        </button>

        <button
          onClick={onReset}
          className="w-full bg-[#090A0F] hover:bg-[#11131A] text-slate-300 font-bold py-3 px-4 border border-[#1E2330] transition-colors flex items-center justify-center gap-2"
          data-testid="reset-button"
        >
          <ArrowsClockwiseIcon size={20} weight="bold" />
          RESET
        </button>
      </div>

      {/* Speed Control */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Speed</span>
          <span className="text-sm font-mono font-bold text-slate-300" data-testid="simulation-speed">{stats.simulation_speed.toFixed(1)}x</span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => onSpeedAdjust(-0.5)}
            className="flex-1 bg-[#090A0F] hover:bg-[#11131A] text-slate-300 font-bold py-2 px-3 border border-[#1E2330] transition-colors flex items-center justify-center"
            data-testid="speed-down-button"
          >
            <CaretDownIcon size={20} weight="bold" />
          </button>
          <button
            onClick={() => onSpeedAdjust(0.5)}
            className="flex-1 bg-[#090A0F] hover:bg-[#11131A] text-slate-300 font-bold py-2 px-3 border border-[#1E2330] transition-colors flex items-center justify-center"
            data-testid="speed-up-button"
          >
            <CaretUpIcon size={20} weight="bold" />
          </button>
        </div>
      </div>

      {/* Predictor Mode Toggle */}
      <div className="mb-6">
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500 block mb-2">Predictor Mode</span>
        <div className="flex gap-2">
          <button
            onClick={() => onPredictorModeChange('rule-based')}
            className={`flex-1 font-bold py-2 px-3 border border-[#1E2330] transition-colors ${
              predictorMode === 'rule-based'
                ? 'bg-[#3B82F6] text-white'
                : 'bg-[#090A0F] text-slate-400 hover:bg-[#11131A]'
            }`}
            data-testid="predictor-rule-based-button"
          >
            RULE
          </button>
          <button
            onClick={() => onPredictorModeChange('ml')}
            className={`flex-1 font-bold py-2 px-3 border border-[#1E2330] transition-colors ${
              predictorMode === 'ml'
                ? 'bg-[#3B82F6] text-white'
                : 'bg-[#090A0F] text-slate-400 hover:bg-[#11131A]'
            }`}
            data-testid="predictor-ml-button"
          >
            ML
          </button>
        </div>
      </div>

      {/* Keyboard Shortcuts */}
      <div>
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500 block mb-3">Keyboard Shortcuts</span>
        <div className="space-y-2 text-xs text-slate-400">
          <div className="flex items-center justify-between">
            <span>Pause/Resume</span>
            <span className="keycap">SPACE</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Speed Up</span>
            <span className="keycap">↑</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Speed Down</span>
            <span className="keycap">↓</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Reset</span>
            <span className="keycap">R</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlsPanel;
