import math
import random
from typing import List, Dict, Tuple, Optional
from models import Satellite, Debris, SpaceObject, create_default_satellites
from predictor import CollisionPredictor
import asyncio
import time


class SpaceSimulation:
    """Core simulation engine for orbital mechanics and collision dynamics"""
    
    def __init__(self, custom_params: Optional[Dict] = None):
        """
        Initialize simulation
        
        Args:
            custom_params: Optional custom parameters for simulation
        """
        # Use custom params or defaults
        params = custom_params or {}
        
        self.satellites: List[Satellite] = []
        self.debris: List[Debris] = []
        self.predictor = CollisionPredictor(mode='rule-based')
        
        # Simulation parameters
        self.max_debris = params.get('max_debris', 500)
        self.kessler_threshold = params.get('kessler_threshold', 100)
        self.collision_distance = params.get('collision_distance', 15.0)
        self.debris_per_collision = params.get('debris_per_collision', 8)
        self.simulation_speed = params.get('simulation_speed', 1.0)
        
        # State tracking
        self.is_paused = False
        self.kessler_syndrome_active = False
        self.total_collisions = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize with satellites
        initial_count = params.get('satellite_count', 12)
        self.satellites = create_default_satellites(initial_count)
    
    def add_satellite(self, x: float, y: float) -> Satellite:
        """Add a new satellite at specified position"""
        # Calculate orbit parameters from position
        orbit_radius = math.sqrt(x*x + y*y)
        orbit_angle = math.atan2(y, x)
        
        satellite = Satellite(
            orbit_radius=max(orbit_radius, 100),
            orbit_speed=random.uniform(0.015, 0.025),
            orbit_angle=orbit_angle,
            eccentricity=random.uniform(0.0, 0.1)
        )
        self.satellites.append(satellite)
        return satellite
    
    def update(self, delta_time: float = 1.0):
        """Main simulation update loop"""
        if self.is_paused:
            return
        
        # Apply simulation speed
        effective_delta = delta_time * self.simulation_speed
        
        # Update all satellites
        for satellite in self.satellites:
            if satellite.active:
                satellite.update_position()
        
        # Update all debris
        debris_to_remove = []
        for debris in self.debris:
            debris.update_position()
            if debris.is_expired():
                debris_to_remove.append(debris)
        
        # Remove expired debris
        for debris in debris_to_remove:
            self.debris.remove(debris)
        
        # Collision detection and risk assessment
        self._check_collisions()
        self._calculate_collision_risks()
        
        # Check for Kessler syndrome
        if len(self.debris) > self.kessler_threshold and not self.kessler_syndrome_active:
            self.kessler_syndrome_active = True
        elif len(self.debris) <= self.kessler_threshold * 0.7:
            self.kessler_syndrome_active = False
        
        # Cap debris count
        if len(self.debris) > self.max_debris:
            # Remove oldest debris
            self.debris = sorted(self.debris, key=lambda d: d.lifetime, reverse=True)[:self.max_debris]
        
        self.frame_count += 1
    
    def _calculate_collision_risks(self):
        """Calculate collision risks for all satellite pairs"""
        satellites = [s for s in self.satellites if s.active]
        
        for i, sat1 in enumerate(satellites):
            max_risk = 0.0
            
            # Check against other satellites
            for j, sat2 in enumerate(satellites):
                if i >= j:
                    continue
                
                distance = sat1.distance_to(sat2)
                rel_velocity = sat1.relative_velocity(sat2)
                
                risk = self.predictor.predict_collision_risk(
                    distance, rel_velocity, sat1.mass, sat2.mass
                )
                
                max_risk = max(max_risk, risk)
                
                # Apply avoidance if needed
                if self.predictor.should_avoid(risk):
                    dx = sat2.x - sat1.x
                    dy = sat2.y - sat1.y
                    sat1.apply_avoidance_maneuver(dx, dy)
            
            # Check against debris
            for debris in self.debris[:50]:  # Only check nearest debris for performance
                distance = sat1.distance_to(debris)
                if distance < 100:
                    rel_velocity = sat1.relative_velocity(debris)
                    risk = self.predictor.predict_collision_risk(
                        distance, rel_velocity, sat1.mass, debris.mass
                    )
                    max_risk = max(max_risk, risk)
            
            sat1.collision_risk = max_risk
    
    def _check_collisions(self):
        """Detect and handle collisions"""
        all_objects: List[SpaceObject] = []
        all_objects.extend([s for s in self.satellites if s.active])
        all_objects.extend(self.debris)
        
        collisions = []
        
        # Check all pairs
        for i in range(len(all_objects)):
            for j in range(i + 1, len(all_objects)):
                obj1 = all_objects[i]
                obj2 = all_objects[j]
                
                distance = obj1.distance_to(obj2)
                collision_threshold = obj1.radius + obj2.radius
                
                if distance < collision_threshold:
                    collisions.append((obj1, obj2))
        
        # Handle collisions
        for obj1, obj2 in collisions:
            self._handle_collision(obj1, obj2)
    
    def _handle_collision(self, obj1: SpaceObject, obj2: SpaceObject):
        """Handle collision between two objects"""
        self.total_collisions += 1
        
        # Calculate collision center
        cx = (obj1.x + obj2.x) / 2
        cy = (obj1.y + obj2.y) / 2
        cz = (obj1.z + obj2.z) / 2
        
        # Create debris fragments
        num_fragments = random.randint(self.debris_per_collision - 2, self.debris_per_collision + 2)
        
        for _ in range(num_fragments):
            # Random velocity based on collision energy
            speed = random.uniform(0.5, 2.0)
            angle = random.uniform(0, 2 * math.pi)
            
            debris = Debris(
                x=cx + random.uniform(-10, 10),
                y=cy + random.uniform(-10, 10),
                z=cz + random.uniform(-5, 5),
                vx=speed * math.cos(angle) + random.uniform(-0.2, 0.2),
                vy=speed * math.sin(angle) + random.uniform(-0.2, 0.2),
                vz=random.uniform(-0.3, 0.3),
                lifetime=random.uniform(800, 1200),
                creation_time=self.frame_count
            )
            self.debris.append(debris)
        
        # Deactivate/remove collided objects
        if isinstance(obj1, Satellite):
            obj1.active = False
            self.satellites = [s for s in self.satellites if s.id != obj1.id]
        elif isinstance(obj1, Debris):
            if obj1 in self.debris:
                self.debris.remove(obj1)
        
        if isinstance(obj2, Satellite):
            obj2.active = False
            self.satellites = [s for s in self.satellites if s.id != obj2.id]
        elif isinstance(obj2, Debris):
            if obj2 in self.debris:
                self.debris.remove(obj2)
    
    def get_state(self) -> Dict:
        """Get current simulation state for transmission"""
        # Calculate overall risk
        risks = [s.collision_risk for s in self.satellites if s.active]
        avg_risk = sum(risks) / len(risks) if risks else 0.0
        max_risk = max(risks) if risks else 0.0
        
        risk_level = self.predictor.get_risk_level(max_risk)
        
        return {
            'satellites': [s.to_dict() for s in self.satellites if s.active],
            'debris': [d.to_dict() for d in self.debris],
            'stats': {
                'satellite_count': len([s for s in self.satellites if s.active]),
                'debris_count': len(self.debris),
                'total_collisions': self.total_collisions,
                'risk_level': risk_level,
                'avg_risk': round(avg_risk, 3),
                'max_risk': round(max_risk, 3),
                'kessler_active': self.kessler_syndrome_active,
                'simulation_speed': self.simulation_speed,
                'is_paused': self.is_paused,
                'frame_count': self.frame_count,
                'uptime': round(time.time() - self.start_time, 1)
            },
            'predictor_mode': self.predictor.mode
        }
    
    def toggle_pause(self):
        """Toggle simulation pause state"""
        self.is_paused = not self.is_paused
        return self.is_paused
    
    def adjust_speed(self, delta: float):
        """Adjust simulation speed"""
        self.simulation_speed = max(0.1, min(self.simulation_speed + delta, 5.0))
        return self.simulation_speed
    
    def reset(self, custom_params: Optional[Dict] = None):
        """Reset simulation to initial state"""
        self.__init__(custom_params)
    
    def set_predictor_mode(self, mode: str) -> bool:
        """Change collision prediction mode"""
        return self.predictor.set_mode(mode)
