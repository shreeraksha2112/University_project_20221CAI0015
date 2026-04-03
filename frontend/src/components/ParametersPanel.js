import React, { useState } from 'react';
import { XIcon } from '@phosphor-icons/react';

const ParametersPanel = ({ onClose, onReset }) => {
  const [useCustom, setUseCustom] = useState(false);
  const [params, setParams] = useState({
    satellite_count: 12,
    max_debris: 500,
    kessler_threshold: 100,
    collision_distance: 15.0,
    debris_per_collision: 8,
    simulation_speed: 1.0
  });

  const handleReset = () => {
    if (useCustom) {
      onReset(params);
    } else {
      onReset(null);
    }
  };

  const handleChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 p-4">
      <div className="bg-[#090A0F] border-2 border-[#1E2330] rounded-sm max-w-2xl w-full max-h-[90vh] overflow-y-auto" data-testid="parameters-panel">
        <div className="sticky top-0 bg-[#090A0F] border-b border-[#1E2330] p-4 flex items-center justify-between">
          <h2 className="text-2xl font-bold tracking-tight" style={{ fontFamily: 'Chivo, sans-serif' }}>SIMULATION PARAMETERS</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors" data-testid="close-params-button">
            <XIcon size={24} weight="bold" />
          </button>
        </div>

        <div className="p-6">
          <div className="mb-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useCustom}
                onChange={(e) => setUseCustom(e.target.checked)}
                className="w-4 h-4"
                data-testid="use-custom-params-checkbox"
              />
              <span className="text-sm text-slate-300 font-medium">Use Custom Parameters</span>
            </label>
          </div>

          {useCustom && (
            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">
                  Satellite Count
                </label>
                <input
                  type="number"
                  value={params.satellite_count}
                  onChange={(e) => handleChange('satellite_count', e.target.value)}
                  className="w-full bg-[#11131A] border border-[#1E2330] text-slate-100 px-3 py-2 rounded-sm font-mono"
                  min="1"
                  max="50"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">
                  Max Debris
                </label>
                <input
                  type="number"
                  value={params.max_debris}
                  onChange={(e) => handleChange('max_debris', e.target.value)}
                  className="w-full bg-[#11131A] border border-[#1E2330] text-slate-100 px-3 py-2 rounded-sm font-mono"
                  min="100"
                  max="2000"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">
                  Kessler Threshold
                </label>
                <input
                  type="number"
                  value={params.kessler_threshold}
                  onChange={(e) => handleChange('kessler_threshold', e.target.value)}
                  className="w-full bg-[#11131A] border border-[#1E2330] text-slate-100 px-3 py-2 rounded-sm font-mono"
                  min="50"
                  max="500"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">
                  Collision Distance
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={params.collision_distance}
                  onChange={(e) => handleChange('collision_distance', e.target.value)}
                  className="w-full bg-[#11131A] border border-[#1E2330] text-slate-100 px-3 py-2 rounded-sm font-mono"
                  min="5"
                  max="50"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">
                  Debris Per Collision
                </label>
                <input
                  type="number"
                  value={params.debris_per_collision}
                  onChange={(e) => handleChange('debris_per_collision', e.target.value)}
                  className="w-full bg-[#11131A] border border-[#1E2330] text-slate-100 px-3 py-2 rounded-sm font-mono"
                  min="3"
                  max="20"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">
                  Initial Speed
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={params.simulation_speed}
                  onChange={(e) => handleChange('simulation_speed', e.target.value)}
                  className="w-full bg-[#11131A] border border-[#1E2330] text-slate-100 px-3 py-2 rounded-sm font-mono"
                  min="0.1"
                  max="5"
                />
              </div>
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleReset}
              className="flex-1 bg-[#3B82F6] hover:bg-[#2563EB] text-white font-bold py-3 px-4 transition-colors"
              data-testid="confirm-reset-button"
            >
              RESET SIMULATION
            </button>
            <button
              onClick={onClose}
              className="flex-1 bg-[#090A0F] hover:bg-[#11131A] text-slate-300 font-bold py-3 px-4 border border-[#1E2330] transition-colors"
            >
              CANCEL
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ParametersPanel;
