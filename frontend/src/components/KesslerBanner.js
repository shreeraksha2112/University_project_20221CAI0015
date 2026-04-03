import React from 'react';
import { WarningOctagonIcon } from '@phosphor-icons/react';

const KesslerBanner = () => {
  return (
    <div className="sticky top-0 z-50 kessler-shake" data-testid="kessler-banner">
      <div className="bg-gradient-to-r from-black via-[#B91C1C] to-black py-4 px-6 border-b-4 border-[#B91C1C]">
        <div className="container mx-auto flex items-center justify-center gap-3">
          <WarningOctagonIcon size={32} weight="fill" className="text-white animate-pulse" />
          <div className="text-center">
            <h2 className="text-2xl font-black tracking-tight text-white" style={{ fontFamily: 'Chivo, sans-serif', textShadow: '0 0 10px rgba(255,255,255,0.5)' }}>
              ⚠️ KESSLER SYNDROME DETECTED ⚠️
            </h2>
            <p className="text-sm text-red-200 font-mono mt-1">
              DEBRIS CASCADE IN PROGRESS - COLLISION RATE CRITICAL
            </p>
          </div>
          <WarningOctagonIcon size={32} weight="fill" className="text-white animate-pulse" />
        </div>
      </div>
    </div>
  );
};

export default KesslerBanner;
