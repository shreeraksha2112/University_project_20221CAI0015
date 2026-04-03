import React, { useRef, useEffect, useState } from 'react';

const SimulationCanvas = ({ simulationState, onAddSatellite }) => {
  const canvasRef = useRef(null);
  const [earthImage, setEarthImage] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const img = new Image();
    img.src = 'https://images.unsplash.com/photo-1663427929868-3941f957bb36?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA1OTV8MHwxfHNlYXJjaHwxfHxoaWdoJTIwcmVzb2x1dGlvbiUyMGVhcnRoJTIwZnJvbSUyMHNwYWNlJTIwZGFyayUyMGJhY2tncm91bmR8ZW58MHx8fHwxNzc1MjM1MDQxfDA&ixlib=rb-4.1.0&q=85';
    img.onload = () => setEarthImage(img);
  }, []);

  useEffect(() => {
    const updateCanvasSize = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        const container = canvas.parentElement;
        const width = container.clientWidth;
        const height = Math.max(600, window.innerHeight * 0.7);
        setCanvasSize({ width, height });
      }
    };

    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !simulationState) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = canvasSize;
    
    canvas.width = width;
    canvas.height = height;

    const centerX = width / 2;
    const centerY = height / 2;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Draw stars background (simple)
    ctx.fillStyle = '#ffffff';
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * width;
      const y = Math.random() * height;
      const size = Math.random() * 1.5;
      ctx.globalAlpha = Math.random() * 0.8 + 0.2;
      ctx.fillRect(x, y, size, size);
    }
    ctx.globalAlpha = 1.0;

    // Draw Earth at center
    if (earthImage) {
      const earthRadius = 60;
      ctx.save();
      ctx.beginPath();
      ctx.arc(centerX, centerY, earthRadius, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(earthImage, centerX - earthRadius, centerY - earthRadius, earthRadius * 2, earthRadius * 2);
      ctx.restore();
      
      // Earth glow
      ctx.strokeStyle = 'rgba(100, 150, 255, 0.3)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(centerX, centerY, earthRadius, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw connection lines between nearby satellites
    const satellites = simulationState.satellites || [];
    for (let i = 0; i < satellites.length; i++) {
      for (let j = i + 1; j < satellites.length; j++) {
        const sat1 = satellites[i];
        const sat2 = satellites[j];
        
        const dx = sat2.x - sat1.x;
        const dy = sat2.y - sat1.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 150) {
          const x1 = centerX + sat1.x;
          const y1 = centerY + sat1.y;
          const x2 = centerX + sat2.x;
          const y2 = centerY + sat2.y;
          
          // Color based on distance
          let color;
          if (distance < 50) {
            color = 'rgba(239, 68, 68, 0.6)'; // Red
          } else if (distance < 100) {
            color = 'rgba(245, 158, 11, 0.4)'; // Amber
          } else {
            color = 'rgba(16, 185, 129, 0.3)'; // Green
          }
          
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    // Draw satellites
    satellites.forEach((sat) => {
      const x = centerX + sat.x;
      const y = centerY + sat.y;
      const scale = 1 + (sat.z / 500); // 3D depth effect
      const size = sat.radius * scale;
      
      // Satellite body
      ctx.fillStyle = '#06B6D4';
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();
      
      // Risk indicator ring
      if (sat.collision_risk > 0.3) {
        let riskColor;
        if (sat.collision_risk > 0.6) {
          riskColor = '#EF4444';
        } else if (sat.collision_risk > 0.3) {
          riskColor = '#F59E0B';
        }
        
        ctx.strokeStyle = riskColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, size + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
      
      // Orbital path hint
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, sat.orbit_radius, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Draw debris
    const debris = simulationState.debris || [];
    debris.forEach((d) => {
      const x = centerX + d.x;
      const y = centerY + d.y;
      const scale = 1 + (d.z / 500);
      const size = d.radius * scale;
      
      ctx.fillStyle = '#94A3B8';
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;
    });

    // Kessler syndrome effect
    if (simulationState.stats?.kessler_active) {
      ctx.fillStyle = 'rgba(185, 28, 28, 0.05)';
      ctx.fillRect(0, 0, width, height);
    }

  }, [simulationState, canvasSize, earthImage]);

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - canvasSize.width / 2;
    const y = e.clientY - rect.top - canvasSize.height / 2;
    
    onAddSatellite(x, y);
  };

  return (
    <div className="relative border border-[#1E2330] bg-black overflow-hidden rounded-sm" style={{ minHeight: '60vh' }}>
      <div className="starfield"></div>
      <canvas
        ref={canvasRef}
        className="space-canvas relative z-10"
        onClick={handleCanvasClick}
        data-testid="simulation-canvas"
      />
      
      <div className="absolute bottom-4 right-4 bg-[#090A0F] border border-[#1E2330] px-3 py-2 rounded-sm z-20">
        <p className="text-xs text-slate-400 font-mono">CLICK TO ADD SATELLITE</p>
      </div>
    </div>
  );
};

export default SimulationCanvas;
