import math
import random
from typing import List, Tuple
from dataclasses import dataclass, field
import uuid


@dataclass
class SpaceObject:
    """Base class for all space objects"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # For 3D depth simulation
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    radius: float = 5.0
    mass: float = 1.0
    
    def get_position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def get_velocity(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)
    
    def distance_to(self, other: 'SpaceObject') -> float:
        """Calculate 3D Euclidean distance to another object"""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def relative_velocity(self, other: 'SpaceObject') -> float:
        """Calculate relative velocity magnitude"""
        dvx = self.vx - other.vx
        dvy = self.vy - other.vy
        dvz = self.vz - other.vz
        return math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)


@dataclass
class Satellite(SpaceObject):
    """Satellite with orbital parameters"""
    orbit_radius: float = 200.0
    orbit_speed: float = 0.02
    orbit_angle: float = 0.0
    orbit_inclination: float = 0.0  # Angle in XY plane
    eccentricity: float = 0.0  # 0 = circular, >0 = elliptical
    active: bool = True
    collision_risk: float = 0.0
    
    def __post_init__(self):
        if self.orbit_angle == 0.0:
            self.orbit_angle = random.uniform(0, 2 * math.pi)
        if self.orbit_inclination == 0.0:
            self.orbit_inclination = random.uniform(0, 2 * math.pi)
        self.radius = 8.0
        self.mass = 100.0
        self.update_position()
    
    def update_position(self):
        """Update position based on orbital parameters"""
        # Elliptical orbit calculation
        r = self.orbit_radius * (1 - self.eccentricity * math.cos(self.orbit_angle))
        
        # Calculate position in orbital plane
        x_orbit = r * math.cos(self.orbit_angle)
        y_orbit = r * math.sin(self.orbit_angle)
        
        # Apply 3D rotation for inclination
        self.x = x_orbit * math.cos(self.orbit_inclination) - y_orbit * math.sin(self.orbit_inclination)
        self.y = x_orbit * math.sin(self.orbit_inclination) + y_orbit * math.cos(self.orbit_inclination)
        self.z = y_orbit * 0.3  # Slight Z variation for depth
        
        # Update velocity components
        self.orbit_angle += self.orbit_speed
        if self.orbit_angle > 2 * math.pi:
            self.orbit_angle -= 2 * math.pi
        
        # Calculate velocity vector (tangent to orbit)
        v_mag = self.orbit_speed * r
        self.vx = -v_mag * math.sin(self.orbit_angle)
        self.vy = v_mag * math.cos(self.orbit_angle)
        self.vz = 0.0
    
    def apply_avoidance_maneuver(self, dx: float, dy: float):
        """Apply collision avoidance by adjusting orbit"""
        # Adjust orbital parameters slightly
        self.orbit_speed *= 0.98  # Slow down slightly
        self.orbit_inclination += random.uniform(-0.1, 0.1)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': 'satellite',
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'z': round(self.z, 2),
            'vx': round(self.vx, 4),
            'vy': round(self.vy, 4),
            'vz': round(self.vz, 4),
            'radius': self.radius,
            'orbit_radius': round(self.orbit_radius, 2),
            'active': self.active,
            'collision_risk': round(self.collision_risk, 3)
        }


@dataclass
class Debris(SpaceObject):
    """Debris fragment from collisions"""
    lifetime: float = 1000.0  # Frames until decay
    creation_time: float = 0.0
    
    def __post_init__(self):
        self.radius = random.uniform(2.0, 5.0)
        self.mass = random.uniform(1.0, 20.0)
    
    def update_position(self, drag_factor: float = 0.9995):
        """Update position with linear motion and slight drag"""
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        
        # Apply slight drag to prevent infinite acceleration
        self.vx *= drag_factor
        self.vy *= drag_factor
        self.vz *= drag_factor
        
        # Decay lifetime
        self.lifetime -= 1
    
    def is_expired(self) -> bool:
        return self.lifetime <= 0
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': 'debris',
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'z': round(self.z, 2),
            'vx': round(self.vx, 4),
            'vy': round(self.vy, 4),
            'vz': round(self.vz, 4),
            'radius': self.radius,
            'lifetime': round(self.lifetime, 0)
        }


def create_default_satellites(count: int = 12) -> List[Satellite]:
    """Create default satellite configuration"""
    satellites = []
    for i in range(count):
        orbit_radius = random.uniform(150, 350)
        orbit_speed = random.uniform(0.015, 0.03)
        eccentricity = random.uniform(0.0, 0.15)
        
        satellite = Satellite(
            orbit_radius=orbit_radius,
            orbit_speed=orbit_speed,
            eccentricity=eccentricity
        )
        satellites.append(satellite)
    
    return satellites
