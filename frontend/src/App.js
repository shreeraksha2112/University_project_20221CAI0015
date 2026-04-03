import { useState, useEffect } from 'react';
import '@/App.css';
import SimulationCanvas from './components/SimulationCanvas';
import Dashboard from './components/Dashboard';
import ControlsPanel from './components/ControlsPanel';
import KesslerBanner from './components/KesslerBanner';
import ParametersPanel from './components/ParametersPanel';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [ws, setWs] = useState(null);
  const [simulationState, setSimulationState] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [showParams, setShowParams] = useState(false);

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Keyboard controls
  useEffect(() => {
    const handleKeyPress = (e) => {
      switch (e.key) {
        case ' ':
          e.preventDefault();
          handleTogglePause();
          break;
        case 'ArrowUp':
          e.preventDefault();
          handleSpeedAdjust(0.5);
          break;
        case 'ArrowDown':
          e.preventDefault();
          handleSpeedAdjust(-0.5);
          break;
        case 'r':
        case 'R':
          e.preventDefault();
          setShowParams(true);
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [ws]);

  const connectWebSocket = () => {
    const wsUrl = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://');
    const websocket = new WebSocket(`${wsUrl}/api/ws/simulation`);

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setSimulationState(data);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      // Reconnect after 2 seconds
      setTimeout(connectWebSocket, 2000);
    };

    setWs(websocket);
  };

  const sendCommand = (command) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(command));
    }
  };

  const handleTogglePause = () => {
    sendCommand({ type: 'toggle_pause' });
  };

  const handleSpeedAdjust = (delta) => {
    sendCommand({ type: 'adjust_speed', delta });
  };

  const handleReset = (params = null) => {
    sendCommand({ type: 'reset', params });
    setShowParams(false);
  };

  const handleAddSatellite = (x, y) => {
    sendCommand({ type: 'add_satellite', x, y });
  };

  const handlePredictorModeChange = (mode) => {
    sendCommand({ type: 'set_predictor_mode', mode });
  };

  if (!simulationState) {
    return (
      <div className="min-h-screen bg-[#030305] flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-slate-400 font-mono text-sm">INITIALIZING SIMULATION...</p>
        </div>
      </div>
    );
  }

  const kesslerActive = simulationState?.stats?.kessler_active || false;

  return (
    <div className="App min-h-screen bg-[#030305] text-slate-100">
      {kesslerActive && <KesslerBanner />}
      
      <div className="container mx-auto p-4 md:p-6">
        <header className="mb-6">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tighter mb-2" style={{ fontFamily: 'Chivo, sans-serif' }}>
            ORBITAL DEBRIS MONITOR
          </h1>
          <div className="flex items-center gap-3">
            <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
            <p className="text-xs text-slate-500 uppercase tracking-widest font-bold">
              {isConnected ? 'SYSTEM ONLINE' : 'RECONNECTING...'}
            </p>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 md:gap-6">
          <div className="lg:col-span-1 flex flex-col gap-4">
            <Dashboard stats={simulationState.stats} />
            <ControlsPanel
              stats={simulationState.stats}
              predictorMode={simulationState.predictor_mode}
              onTogglePause={handleTogglePause}
              onSpeedAdjust={handleSpeedAdjust}
              onReset={() => setShowParams(true)}
              onPredictorModeChange={handlePredictorModeChange}
            />
          </div>

          <div className="lg:col-span-3">
            <SimulationCanvas
              simulationState={simulationState}
              onAddSatellite={handleAddSatellite}
            />
          </div>
        </div>
      </div>

      {showParams && (
        <ParametersPanel
          onClose={() => setShowParams(false)}
          onReset={handleReset}
        />
      )}
    </div>
  );
}

export default App;
