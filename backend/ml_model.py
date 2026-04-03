import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


class CollisionPredictionModel(nn.Module):
    """
    Neural network for predicting satellite collision risk
    
    Input features:
    - distance: Distance between objects (normalized)
    - relative_velocity: Relative velocity magnitude (normalized)
    - obj1_mass: Mass of first object (normalized)
    - obj2_mass: Mass of second object (normalized)
    
    Output:
    - risk: Collision risk score [0, 1]
    """
    
    def __init__(self, input_size=4, hidden_sizes=[32, 16, 8], dropout=0.2):
        super(CollisionPredictionModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ModelTrainer:
    """Trainer for collision prediction model"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate_synthetic_training_data(self, num_samples=10000):
        """
        Generate synthetic training data based on physics rules
        This simulates what would happen in real collisions
        """
        X = []
        y = []
        
        for _ in range(num_samples):
            # Generate random features
            distance = np.random.uniform(0, 500)  # pixels
            rel_velocity = np.random.uniform(0, 5)  # units
            mass1 = np.random.uniform(1, 200)
            mass2 = np.random.uniform(1, 200)
            
            # Calculate risk using physics-based rules (ground truth)
            # Distance component
            if distance < 50:
                distance_risk = 1.0 - (distance / 50)
            elif distance < 150:
                distance_risk = 0.5 * (1.0 - (distance / 150))
            else:
                distance_risk = 0.0
            
            # Velocity component
            velocity_risk = min(rel_velocity / 2.0, 1.0)
            
            # Mass component
            total_mass = mass1 + mass2
            mass_factor = min(total_mass / 200.0, 1.0) * 0.3
            
            # Combined risk
            risk = distance_risk * 0.6 + velocity_risk * 0.3 + mass_factor * 0.1
            risk = max(0.0, min(risk, 1.0))
            
            # Normalize features
            X.append([
                distance / 500.0,
                rel_velocity / 5.0,
                mass1 / 200.0,
                mass2 / 200.0
            ])
            y.append(risk)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def train(self, epochs=100, batch_size=32):
        """Train the model on synthetic data"""
        print("Generating training data...")
        X_train, y_train = self.generate_synthetic_training_data()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training on {len(X_train)} samples for {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        print("Training complete!")
    
    def save_model(self, filepath):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': {
                'input_size': 4,
                'hidden_sizes': [32, 16, 8],
                'dropout': 0.2
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        arch = checkpoint['model_architecture']
        model = CollisionPredictionModel(
            input_size=arch['input_size'],
            hidden_sizes=arch['hidden_sizes'],
            dropout=arch['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


def train_and_save_model(output_path='/app/backend/models/collision_model.pth'):
    """Train and save a collision prediction model"""
    # Create models directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create and train model
    model = CollisionPredictionModel()
    trainer = ModelTrainer(model)
    trainer.train(epochs=50, batch_size=64)
    
    # Save model
    trainer.save_model(output_path)
    print(f"\nModel ready at: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Train a model when run directly
    train_and_save_model()
