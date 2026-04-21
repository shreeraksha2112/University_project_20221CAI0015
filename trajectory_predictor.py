import torch
import torch.nn as nn
import numpy as np
import os

# ==========================================================
# DEVICE CONFIGURATION
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# POSITIONAL ENCODING (OPTIONAL MODULE)
# ==========================================================

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for trajectory sequences.
    """

    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


# ==========================================================
# TRANSFORMER MODEL
# ==========================================================

class TrajectoryTransformer(nn.Module):
    """
    Transformer model for trajectory prediction.
    Predicts future states based on (state + action) sequences.
    """

    def __init__(self, state_dim, action_dim, d_model=64, nhead=4, num_layers=2, horizon=5):
        super(TrajectoryTransformer, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        # Input embedding
        self.embedding = nn.Linear(state_dim + action_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=horizon)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output decoder
        self.decoder = nn.Linear(d_model, state_dim)

    def forward(self, src):
        """
        src: [batch, seq_len, state_dim + action_dim]
        returns: [batch, seq_len, state_dim]
        """

        seq_len = src.size(1)

        # Embedding
        x = self.embedding(src)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer forward pass
        out = self.transformer(x)

        # Predict deltas
        deltas = self.decoder(out)

        # Residual cumulative trajectory
        states = src[:, :, :self.state_dim]

        return states + torch.cumsum(deltas, dim=1)


# ==========================================================
# TRAJECTORY PREDICTOR WRAPPER
# ==========================================================

class TrajectoryPredictor:
    """
    Wrapper for training, inference, and utilities.
    """

    def __init__(self, state_dim=6, action_dim=3, horizon=5, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.model = TrajectoryTransformer(
            state_dim,
            action_dim,
            horizon=horizon
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    def predict(self, state, action):
        """
        Predict trajectory over horizon using constant action.
        """

        self.model.eval()

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            a = torch.FloatTensor(action).unsqueeze(0).to(device)

            x = torch.cat([s, a], dim=-1).unsqueeze(1)
            x_seq = x.repeat(1, self.horizon, 1)

            preds = self.model(x_seq)

            return preds.squeeze(0).cpu().numpy()

    # ------------------------------------------------------
    # TRAINING STEP (SINGLE STEP)
    # ------------------------------------------------------

    def train_step(self, states, actions, next_states):
        """
        Train on single-step transitions.
        """

        self.model.train()

        s = torch.FloatTensor(states).to(device)
        a = torch.FloatTensor(actions).to(device)
        ns = torch.FloatTensor(next_states).unsqueeze(1).to(device)

        x = torch.cat([s, a], dim=-1).unsqueeze(1)

        preds = self.model(x)

        loss = self.criterion(preds, ns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------
    # SEQUENCE TRAINING (IMPORTANT UPGRADE)
    # ------------------------------------------------------

    def train_sequence(self, state_seq, action_seq, next_state_seq):
        """
        Train on full sequences instead of single steps.
        """

        self.model.train()

        s = torch.FloatTensor(state_seq).to(device)
        a = torch.FloatTensor(action_seq).to(device)
        ns = torch.FloatTensor(next_state_seq).to(device)

        x = torch.cat([s, a], dim=-1)

        preds = self.model(x)

        loss = self.criterion(preds, ns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------

    def evaluate(self, state_seq, action_seq, next_state_seq):
        """
        Evaluate model performance without training.
        """

        self.model.eval()

        with torch.no_grad():
            s = torch.FloatTensor(state_seq).to(device)
            a = torch.FloatTensor(action_seq).to(device)
            ns = torch.FloatTensor(next_state_seq).to(device)

            x = torch.cat([s, a], dim=-1)

            preds = self.model(x)

            loss = self.criterion(preds, ns)

        return loss.item()

    # ------------------------------------------------------
    # MODEL SAVE / LOAD
    # ------------------------------------------------------

    def save_model(self, path="trajectory_model.pth"):
        """
        Saves model weights.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="trajectory_model.pth"):
        """
        Loads model weights.
        """
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=device))
            self.model.to(device)
        else:
            print("Model file not found!")

    # ------------------------------------------------------
    # UTILITY: RANDOM DATA GENERATION (FOR TESTING)
    # ------------------------------------------------------

    def generate_dummy_data(self, batch_size=32):
        """
        Generates synthetic data for testing.
        """

        states = np.random.randn(batch_size, self.state_dim)
        actions = np.random.randn(batch_size, self.action_dim)
        next_states = states + 0.1 * actions[:, :self.state_dim]

        return states, actions, next_states


# ==========================================================
# DEMO / TEST RUN
# ==========================================================

def run_demo():
    """
    Demonstrates prediction and training.
    """

    predictor = TrajectoryPredictor()

    # Generate dummy data
    states, actions, next_states = predictor.generate_dummy_data()

    # Train for few iterations
    print("Training...")
    for i in range(10):
        loss = predictor.train_step(states, actions, next_states)
        print(f"Step {i}, Loss: {loss:.4f}")

    # Predict trajectory
    test_state = states[0]
    test_action = actions[0]

    pred_traj = predictor.predict(test_state, test_action)

    print("\nPredicted Trajectory:")
    print(pred_traj)

    # Save model
    predictor.save_model()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run_demo()
