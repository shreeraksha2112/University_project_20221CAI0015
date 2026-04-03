from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Dict, List
from simulation import SpaceSimulation

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Global simulation instance
simulation = SpaceSimulation()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                dead_connections.append(connection)
        
        for conn in dead_connections:
            self.disconnect(conn)

manager = ConnectionManager()

# Pydantic models for API
class SimulationParams(BaseModel):
    satellite_count: Optional[int] = 12
    max_debris: Optional[int] = 500
    kessler_threshold: Optional[int] = 100
    collision_distance: Optional[float] = 15.0
    debris_per_collision: Optional[int] = 8
    simulation_speed: Optional[float] = 1.0

class AddSatelliteRequest(BaseModel):
    x: float
    y: float

class SpeedAdjustRequest(BaseModel):
    delta: float

class PredictorModeRequest(BaseModel):
    mode: str

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Space Debris Simulation API", "version": "1.0"}

@api_router.get("/simulation/state")
async def get_simulation_state():
    """Get current simulation state"""
    return simulation.get_state()

@api_router.post("/simulation/reset")
async def reset_simulation(params: Optional[SimulationParams] = None):
    """Reset simulation with optional custom parameters"""
    custom_params = params.model_dump() if params else None
    simulation.reset(custom_params)
    return {"status": "reset", "params": custom_params}

@api_router.post("/simulation/toggle-pause")
async def toggle_pause():
    """Toggle pause state"""
    is_paused = simulation.toggle_pause()
    return {"is_paused": is_paused}

@api_router.post("/simulation/speed")
async def adjust_speed(request: SpeedAdjustRequest):
    """Adjust simulation speed"""
    new_speed = simulation.adjust_speed(request.delta)
    return {"simulation_speed": new_speed}

@api_router.post("/simulation/add-satellite")
async def add_satellite(request: AddSatelliteRequest):
    """Add a new satellite at position"""
    satellite = simulation.add_satellite(request.x, request.y)
    return {"status": "added", "satellite": satellite.to_dict()}

@api_router.post("/simulation/predictor-mode")
async def set_predictor_mode(request: PredictorModeRequest):
    """Set collision predictor mode"""
    success = simulation.set_predictor_mode(request.mode)
    model_info = simulation.predictor.get_model_info()
    return {
        "status": "success" if success else "failed",
        "mode": request.mode,
        "model_info": model_info
    }

@api_router.get("/ml/model-info")
async def get_model_info():
    """Get ML model information"""
    return simulation.predictor.get_model_info()

@api_router.post("/ml/train-model")
async def train_model():
    """Train a new ML model (runs in background)"""
    try:
        from ml_model import train_and_save_model
        import threading
        
        def train_in_background():
            try:
                train_and_save_model()
                # Reload model in predictor
                simulation.predictor.load_ml_model('/app/backend/models/collision_model.pth')
            except Exception as e:
                logging.error(f"Background training failed: {e}")
        
        thread = threading.Thread(target=train_in_background)
        thread.start()
        
        return {
            "status": "training_started",
            "message": "Model training started in background. This may take a few minutes."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@api_router.post("/ml/load-model")
async def load_model(model_path: str = "/app/backend/models/collision_model.pth"):
    """Load ML model from specified path"""
    success = simulation.predictor.load_ml_model(model_path)
    return {
        "status": "success" if success else "failed",
        "model_info": simulation.predictor.get_model_info()
    }

@api_router.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_json(simulation.get_state())
        
        # Listen for commands from client
        async def listen_for_commands():
            try:
                while True:
                    data = await websocket.receive_text()
                    command = json.loads(data)
                    
                    if command.get('type') == 'toggle_pause':
                        simulation.toggle_pause()
                    elif command.get('type') == 'adjust_speed':
                        simulation.adjust_speed(command.get('delta', 0))
                    elif command.get('type') == 'reset':
                        simulation.reset(command.get('params'))
                    elif command.get('type') == 'add_satellite':
                        simulation.add_satellite(command.get('x', 0), command.get('y', 0))
                    elif command.get('type') == 'set_predictor_mode':
                        simulation.set_predictor_mode(command.get('mode', 'rule-based'))
            except WebSocketDisconnect:
                pass
        
        # Start listening for commands
        listen_task = asyncio.create_task(listen_for_commands())
        
        # Simulation loop - send updates at ~30 FPS
        while True:
            simulation.update()
            state = simulation.get_state()
            await websocket.send_json(state)
            await asyncio.sleep(1/30)  # ~30 FPS
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
