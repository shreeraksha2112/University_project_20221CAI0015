import math
from typing import Tuple, Optional
import numpy as np


class CollisionPredictor:
    """Collision prediction with rule-based and ML-ready modes"""
    
    def __init__(self, mode: str = 'rule-based'):
        """
        Initialize predictor
        
        Args:
            mode: 'rule-based' or 'ml' (ML requires model to be loaded)
        """
        self.mode = mode
        self.ml_model = None
        
        # Rule-based thresholds
        self.distance_threshold = 50.0  # Critical distance in pixels
        self.velocity_threshold = 0.5   # Critical relative velocity
        self.warning_distance = 100.0   # Warning zone
        
    def load_ml_model(self, model_path: str):
        """
        Load PyTorch model for ML-based prediction
        
        Args:
            model_path: Path to .pth model file
        """
        try:
            import torch
            self.ml_model = torch.load(model_path)
            self.ml_model.eval()
            self.mode = 'ml'
            return True
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            return False
    
    def predict_collision_risk(self, distance: float, relative_velocity: float,
                              obj1_mass: float = 1.0, obj2_mass: float = 1.0) -> float:
        """
        Predict collision risk between two objects
        
        Args:
            distance: Distance between objects
            relative_velocity: Relative velocity magnitude
            obj1_mass: Mass of first object
            obj2_mass: Mass of second object
            
        Returns:
            Risk score from 0.0 (safe) to 1.0 (imminent collision)
        """
        if self.mode == 'ml' and self.ml_model is not None:
            return self._ml_prediction(distance, relative_velocity, obj1_mass, obj2_mass)
        else:
            return self._rule_based_prediction(distance, relative_velocity, obj1_mass, obj2_mass)
    
    def _rule_based_prediction(self, distance: float, relative_velocity: float,
                               obj1_mass: float, obj2_mass: float) -> float:
        """
        Rule-based collision risk calculation
        
        Risk factors:
        - Distance (closer = higher risk)
        - Relative velocity (faster approach = higher risk)
        - Combined mass (larger objects = higher impact)
        """
        # Distance component (inverse relationship)
        if distance < self.distance_threshold:
            distance_risk = 1.0 - (distance / self.distance_threshold)
        elif distance < self.warning_distance:
            distance_risk = 0.5 * (1.0 - (distance / self.warning_distance))
        else:
            distance_risk = 0.0
        
        # Velocity component
        velocity_risk = min(relative_velocity / self.velocity_threshold, 1.0)
        
        # Mass component (normalized)
        total_mass = obj1_mass + obj2_mass
        mass_factor = min(total_mass / 200.0, 1.0) * 0.3  # Mass contributes up to 30%
        
        # Combined risk (weighted average)
        risk = (distance_risk * 0.6 + velocity_risk * 0.3 + mass_factor * 0.1)
        
        return min(max(risk, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _ml_prediction(self, distance: float, relative_velocity: float,
                       obj1_mass: float, obj2_mass: float) -> float:
        """
        ML-based collision risk prediction
        
        Expected model input: [distance, relative_velocity, obj1_mass, obj2_mass]
        Expected output: risk score [0, 1]
        """
        try:
            import torch
            
            # Normalize inputs (example normalization)
            normalized_distance = distance / 500.0
            normalized_velocity = relative_velocity / 2.0
            normalized_mass1 = obj1_mass / 100.0
            normalized_mass2 = obj2_mass / 100.0
            
            # Create input tensor
            inputs = torch.tensor([[normalized_distance, normalized_velocity,
                                   normalized_mass1, normalized_mass2]], dtype=torch.float32)
            
            # Get prediction
            with torch.no_grad():
                output = self.ml_model(inputs)
                risk = output.item()
            
            return min(max(risk, 0.0), 1.0)
        except Exception as e:
            print(f"ML prediction failed, falling back to rule-based: {e}")
            return self._rule_based_prediction(distance, relative_velocity, obj1_mass, obj2_mass)
    
    def should_avoid(self, risk: float) -> bool:
        """Determine if avoidance maneuver is needed"""
        return risk > 0.6
    
    def get_risk_level(self, risk: float) -> str:
        """Convert risk score to categorical level"""
        if risk < 0.3:
            return 'LOW'
        elif risk < 0.6:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def set_mode(self, mode: str):
        """Switch between rule-based and ML modes"""
        if mode in ['rule-based', 'ml']:
            if mode == 'ml' and self.ml_model is None:
                print("Warning: No ML model loaded, staying in rule-based mode")
                return False
            self.mode = mode
            return True
        return False
