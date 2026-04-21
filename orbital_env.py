import numpy as np
import hapsira
import plotly.graph_objects as go
from hapsira.bodies import Earth
from hapsira.twobody import Orbit
from astropy import units as u
from astropy.time import Time

class OrbitalEnvironment:
    """
    Simulates orbital dynamics using Relative Orbital Elements (ROE).
    """
    def __init__(self, chief_orbit=None, deputy_roe=None, dt=60*u.s, noise_std=0.001):
        self.dt = dt
        self.earth = Earth
        self.noise_std = noise_std
        
        if chief_orbit is None:
            # Default LEO orbit
            self.chief_orbit = Orbit.circular(Earth, alt=500 * u.km)
        else:
            self.chief_orbit = chief_orbit
            
        if deputy_roe is None:
            # Default ROE: [da, dlambda, dex, dey, dix, diy]
            # Units: km for da, rad for others (approximated for Hill frame)
            self.deputy_roe = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        else:
            self.deputy_roe = deputy_roe
            
        self.current_time = Time.now()
        self.history = []

    def _roe_to_cartesian_relative(self, roe, a):
        """
        Converts ROE to Hill frame (RTN) relative position and velocity.
        Simplified quasi-ROE to Hill frame conversion.
        """
        da, dlambda, dex, dey, dix, diy = roe
        
        # Position in Hill frame (x: radial, y: along-track, z: cross-track)
        x = da - a * np.cos(0) * dex - a * np.sin(0) * dey # Simplified theta=0
        y = dlambda + 1.5 * (da/a) * 0 * a # Simplified time-dependent
        z = dix # Simplified
        
        # This is a placeholder for a more robust ROE to Cartesian conversion
        # In a real scenario, we'd use the full GVE or linearized equations (HCW)
        pos = np.array([x, y, z])
        vel = np.array([0.0, 0.0, 0.0]) # Simplified velocity
        
        return pos, vel

    def step(self, action=None):
        """
        Propagates the orbit by dt.
        action: [dv_r, dv_t, dv_n] impulse maneuvers in km/s.
        """
        a = self.chief_orbit.a.to(u.km).value
        n = np.sqrt(Earth.k.to(u.km**3 / u.s**2).value / a**3)
        theta = 0.0 # Simplified argument of latitude for quasi-ROE

        # 1. Apply action using Gauss Variational Equations (GVE) for ROE
        if action is not None:
            # Linearized ROE state updates from impulsive maneuvers [dv_r, dv_t, dv_n]
            # Reference: D'Amico, S., "Autonomous Formation Flying in LEO", 2010
            dv_r, dv_t, dv_n = action
            
            # Change in Semi-major axis (da)
            self.deputy_roe[0] += (2.0 / n) * dv_t
            
            # Change in Along-track separation (dlambda)
            self.deputy_roe[1] += (-2.0 / (n * a)) * dv_r
            
            # Change in Relative Eccentricity Vector (dex, dey)
            self.deputy_roe[2] += (1.0 / (n * a)) * (np.sin(theta) * dv_r + 2 * np.cos(theta) * dv_t)
            self.deputy_roe[3] += (1.0 / (n * a)) * (-np.cos(theta) * dv_r + 2 * np.sin(theta) * dv_t)
            
            # Change in Relative Inclination Vector (dix, diy)
            self.deputy_roe[4] += (1.0 / (n * a)) * (np.cos(theta) * dv_n)
            self.deputy_roe[5] += (1.0 / (n * a)) * (np.sin(theta) * dv_n)

        # 2. Propagate chief orbit
        self.chief_orbit = self.chief_orbit.propagate(self.dt)
        self.current_time += self.dt
        
        # 3. Propagate relative motion (linearized drift)
        # dlambda_dot = -1.5 * n * (da/a)
        self.deputy_roe[1] -= 1.5 * n * (self.deputy_roe[0] / a) * self.dt.to(u.s).value
        
        true_state = self._get_state()
        
        # Record Cartesian History
        pos, _ = self._roe_to_cartesian_relative(true_state, a)
        self.history.append(pos)
        
        observation = self._add_noise(true_state)
        reward = self._calculate_reward(true_state)
        
        return observation, true_state, reward, False, {}

    def _get_state(self):
        return self.deputy_roe.copy()

    def _add_noise(self, state):
        # Add debris uncertainty and sensor noise using configurable std dev
        noise = np.random.normal(0, self.noise_std, size=state.shape)
        return state + noise

    def _calculate_reward(self, state):
        # Reward = Avoid collision + Minimize Fuel
        # da, dlambda, dex, dey, dix, diy = state
        dist = np.linalg.norm(state[0:3]) # Simplified distance metric
        
        reward = 0
        if dist < 0.01: # Collision threshold
            reward -= 100
        else:
            reward += 1.0 # Living reward
            
        return reward

    def render(self, mode='human'):
        """
        Generates an interactive 3D Plotly visualization of the orbital state.
        """
        # Fetch relative coordinates
        a_chief = self.chief_orbit.a.to(u.km).value
        current_roe = self._get_state()
        current_pos, _ = self._roe_to_cartesian_relative(current_roe, a_chief)
        
        fig = go.Figure()
        
        # Chief Spacecraft (Center)
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            name='Chief Spacecraft',
            marker=dict(size=10, color='orange', symbol='diamond'),
            hovertemplate="<b>%{name}</b><br>X: %{x:.4f} km<br>Y: %{y:.4f} km<br>Z: %{z:.4f} km<extra></extra>"
        ))
        
        # Deputy History (Trajectory)
        if len(self.history) > 1:
            hist = np.array(self.history)
            fig.add_trace(go.Scatter3d(
                x=hist[:,0], y=hist[:,1], z=hist[:,2],
                mode='lines',
                name='True Trajectory',
                line=dict(color='cyan', width=4),
                hovertemplate="<b>%{name} Point</b><br>X: %{x:.4f} km<br>Y: %{y:.4f} km<br>Z: %{z:.4f} km<extra></extra>"
            ))
            
        # Current Deputy State (Point)
        fig.add_trace(go.Scatter3d(
            x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
            mode='markers+text',
            name='Current Deputy',
            marker=dict(size=8, color='red', symbol='x'),
            text=[f"({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}) km"],
            textposition="top center",
            hovertemplate="<b>%{name}</b><br>X: %{x:.4f} km<br>Y: %{y:.4f} km<br>Z: %{z:.4f} km<extra></extra>"
        ))
        
        # Axis configurations matching dashboard aesthetics
        max_v = max(np.max(np.abs(self.history)) if len(self.history) > 0 else 0.1, 0.1)
        fig.update_layout(
            title="Orbital Dynamics Sandbox: Relative Motion (Hill Frame)",
            scene=dict(
                xaxis=dict(title='Radial (km)', range=[-max_v, max_v], backgroundcolor="rgb(10, 10, 20)", gridcolor="#444"),
                yaxis=dict(title='Along-track (km)', range=[-max_v, max_v], backgroundcolor="rgb(10, 10, 20)", gridcolor="#444"),
                zaxis=dict(title='Cross-track (km)', range=[-max_v, max_v], backgroundcolor="rgb(10, 10, 20)", gridcolor="#444"),
                aspectmode='cube'
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        if mode == 'human':
            fig.show()
            
        return fig

    def reset(self):
        self.chief_orbit = Orbit.circular(Earth, alt=500 * u.km)
        self.deputy_roe = np.random.normal(0, 0.05, size=6)
        self.deputy_roe[1] = 0.5 # Initial along-track separation
        self.history = [] # Reset history on reset
        return self._get_state()

if __name__ == "__main__":
    env = OrbitalEnvironment()
    print("Initial State:", env.reset())
    for _ in range(20):
        obs, true_s, rew, done, _ = env.step()
    print(f"Final Obs: {obs}")
    # Generate and display the gym visualization
    fig = env.render()
    print("Rendered Plotly Figure object generated.")
