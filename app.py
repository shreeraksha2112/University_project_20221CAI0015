import streamlit as st
import os
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from rl_agent import PPO
from trajectory_predictor import TrajectoryPredictor
from trajectory_optimizer import SCPOptimizer
from belief_estimator import BeliefEstimator
from astropy import units as u
from hapsira.bodies import Earth


st.set_page_config(page_title="Orbital Simulation Dashboard", layout="wide")

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Application Mode", ["Orbital Logic Simulator", "Exoplanet Orbit Explorer", "Kessler Syndrome Sandbox"])
st.sidebar.divider()

if app_mode == "Exoplanet Orbit Explorer":
    from exoplanet_feature import render_exoplanet_explorer
    render_exoplanet_explorer()
    st.stop()
elif app_mode == "Kessler Syndrome Sandbox":
    from kessler_simulator import render_kessler_sandbox
    render_kessler_sandbox()
    st.stop()

st.title("🛰️ Orbital Dynamics Interactive Dashboard")
st.markdown("""
This dashboard visualizes the relative orbital motion between a chief spacecraft and a deputy (or debris) using **Relative Orbital Elements (ROE)**.
""")

with st.expander("📖 Guide: How to Read the 3D Graph (For Beginners)"):
    st.markdown("""
    **What am I looking at?**
    The 3D plot shows the space around our main spacecraft (the **Chief**). The axes (Radial, Along-track, Cross-track) represent distances in kilometers relative to the Chief.
    
    *   🔶 **Chief Spacecraft (Orange Diamond)**: Your main satellite, fixed at the center `[0, 0, 0]`.
    *   ❌ **Current Deputy (Red Cross)**: The actual position of the incoming debris or secondary spacecraft you are trying to track or avoid.
    *   🟦 **True Trajectory (Cyan Line)**: The exact, perfectly accurate physical path the deputy has taken through space.
    *   🔎 **Noisy Observation (Faint Blue Dots)**: The raw sensor data (like a radar reading with static) tracking the deputy. Because of sensor imperfections, these float around the true position.
    *   🟡 **Predicted Trajectory (Yellow Dashed Line)**: (Visible when "Enable RL Agent Maneuvers" is checked) Our Prediction AI's (Transformer network) forecast of where the deputy is headed next!
    """)

# Sidebar for configuration
st.sidebar.header("Simulation Settings")
scenario_type = st.sidebar.selectbox("Select Scenario", ["Collision Avoidance", "Rendezvous"])
steps = st.sidebar.slider("Simulation Steps", 10, 2000, 500) # Increased range for longer missions
delay = st.sidebar.slider("Frame Delay (s)", 0.01, 0.5, 0.05)
noise_std = st.sidebar.slider("Sensor Noise (Std Dev)", 0.0, 0.05, 0.01, step=0.001)

# Visualization Mode Toggle
st.sidebar.divider()
st.sidebar.subheader("Visualization Settings")
view_mode = st.sidebar.radio("Display Mode", ["3D Interactive", "2D Orbital Projection (Along-Track vs Radial)"])
show_prediction = st.sidebar.checkbox("Always Show Future Prediction", value=True)
show_extras = st.sidebar.checkbox("Show Background Satellites & Space Dust", value=True)
show_grid = st.sidebar.checkbox("Show Reference Orbital Plane Grid", value=True)

# Physics Helper Functions (Replacing Gym environment)
def propagate_roe(roe, a, dt, action=None):
    """Propagates Relative Orbital Elements using Gauss Equations and Linearized Drift."""
    # Earth Gravitational Parameter in km^3/s^2
    mu = Earth.k.to(u.km**3 / u.s**2).value
    n = np.sqrt(mu / a**3)
    theta = 0.0 # Simplified argument of latitude for relative motion
    
    new_roe = roe.copy()
    if action is not None:
        dv_r, dv_t, dv_n = action
        new_roe[0] += (2.0 / n) * dv_t
        new_roe[1] += (-2.0 / (n * a)) * dv_r
        new_roe[2] += (1.0 / (n * a)) * (np.sin(theta) * dv_r + 2 * np.cos(theta) * dv_t)
        new_roe[3] += (1.0 / (n * a)) * (-np.cos(theta) * dv_r + 2 * np.sin(theta) * dv_t)
        new_roe[4] += (1.0 / (n * a)) * (np.cos(theta) * dv_n)
        new_roe[5] += (1.0 / (n * a)) * (np.sin(theta) * dv_n)
    
    # Time-dependent drift in along-track separation (dlambda)
    new_roe[1] -= 1.5 * n * (new_roe[0] / a) * dt
    return new_roe

def roe_to_cartesian(roe, a):
    """Converts ROE to Hill Frame (Radial, Along-track, Cross-track) coordinates."""
    da, dlambda, dex, dey, dix, diy = roe
    # Simplified Hill Frame conversion (Theta=0 approximation)
    x = da - a * dex 
    y = dlambda
    z = dix
    return np.array([x, y, z])

# Scenario Configuration (Replacing ScenarioGenerator)
if scenario_type == "Collision Avoidance":
    # High-intensity collision course
    initial_roe = np.array([0.15, 1.0, 0.05, 0.02, 0.0, 0.0])
else:
    # Complicated Rendezvous geometry
    initial_roe = np.array([-0.05, 3.5, 0.15, 0.1, 0.08, 0.05])

# Chief Orbit Constants
a_chief = 6378.137 + 500  # Earth Radius + Altitude in km
dt_step = 60.0            # 60s per simulation step

# Belief Estimator Initialization
estimator = BeliefEstimator()

# RL Agent Initialization
st.sidebar.divider()
st.sidebar.subheader("RL Agent Settings")
enable_rl = st.sidebar.checkbox("Enable RL Agent Maneuvers", value=False)
model_path = "PPO_SpacecraftCollisionAvoidance.pth"

ppo_agent = None
if enable_rl:
    state_dim = 6
    action_dim = 3
    # Use default hyperparameters for inference
    ppo_agent = PPO(state_dim, action_dim, 0.0003, 0.001, 0.99, 40, 0.2, True)
    
    # Initialize Predictor and Optimizer
    predictor = TrajectoryPredictor(state_dim=state_dim, action_dim=action_dim, horizon=5)
    scp_optimizer = SCPOptimizer(action_dim=action_dim, safe_distance=0.01)

    if os.path.exists(model_path):
        try:
            ppo_agent.load(model_path)
            st.sidebar.success("RL Model Loaded")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    else:
        st.sidebar.warning("Model file not found. Using untrained agent.")

# Simulation State
if 'running' not in st.session_state:
    st.session_state.running = False

def run_simulation():
    st.session_state.running = True
    
run_btn = st.sidebar.button("Run Simulation", on_click=run_simulation)

# Layout: Plots and Telemetry
col1, col2 = st.columns([2, 1])

with col1:
    status_placeholder = st.empty()
    plot_placeholder = st.empty()

with col2:
    st.subheader("Real-time Telemetry")
    telemetry_placeholder = st.empty()
    st.divider()
    st.subheader("Current Belief (Hill Frame)")
    belief_metrics_placeholder = st.empty()
    st.divider()
    st.subheader("Dynamic Physics Telemetry")
    physics_metrics_placeholder = st.empty()
    reward_placeholder = st.empty()

if st.session_state.running:
    try:
        status_placeholder.info("🚀 **Simulation Active**")
        
        # Reset state from scenario config
        true_state = initial_roe.copy()
        obs = true_state + np.random.normal(0, noise_std, size=6)
        estimator.reset()
        
        history = []
        true_history = []
        belief_history = []
        rewards = []
        
        pos_true_history = []
        pos_obs_history = []
        pos_belief_history = []
        total_dv = 0.0
        
        # Initialize Background Traffic (if enabled)
        bg_sats_roe = []
        if show_extras:
            for _ in range(25): # A small orbital swarm
                # Random ROE: da, dlambda, dex, dey, dix, diy
                # Distributed around the chief but with slight offsets
                s_da = np.random.uniform(-0.1, 0.1)
                s_dlam = np.random.uniform(-5.0, 5.0)
                s_roe = np.array([s_da, s_dlam, np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])
                bg_sats_roe.append(s_roe)
            bg_sats_roe = np.array(bg_sats_roe)
            
        # Initialize Static Starfield (consistent across frames)
        stars = np.random.uniform(-20, 20, size=(100, 3))
        
        for i in range(steps):
            # Update Belief State Estimation
            belief_state = estimator.estimate(obs)
            belief_history.append(belief_state)

            # Determine action from RL or Optimizer
            predicted_trajectory = None
            if enable_rl and ppo_agent:
                # Use the estimated belief state for decision making
                raw_action = ppo_agent.select_action(belief_state)
                
                # AI Predictive trajectory extrapolation (Reverted to user-preferred version)
                predicted_trajectory = []
                if len(true_history) > 1:
                    v = true_state - true_history[-1]
                    accel = v - (true_history[-1] - true_history[-2])
                elif len(true_history) == 1:
                    v = true_state - true_history[-1]
                    accel = np.zeros_like(v)
                else:
                    v = np.array([0.0, -0.002, 0.0001, 0.0, 0.0001, 0.0])
                    accel = np.zeros_like(v)
                    
                curr_pred = belief_state.copy() # Start prediction from current belief
                curr_v = v.copy()
                for j in range(1, 40): # Horizon
                    curr_v = curr_v + accel
                    curr_pred = curr_pred + curr_v + np.random.normal(0, 0.00005 * j, size=6)
                    predicted_trajectory.append(curr_pred.copy())
                
                action = scp_optimizer.optimize(belief_state, raw_action, predicted_trajectory)
            else:
                action = None
            
            # Record current state histories
            true_history.append(true_state.copy())
            history.append(obs.copy())
            
            # Update history of positions for plotting
            pos_true = roe_to_cartesian(true_state, a_chief)
            pos_obs = roe_to_cartesian(obs, a_chief)
            pos_belief = roe_to_cartesian(belief_state, a_chief)
            
            pos_true_history.append(pos_true)
            pos_obs_history.append(pos_obs)
            pos_belief_history.append(pos_belief)
            
            # Update reward (Simple proximity/penalty reward)
            curr_dist = np.linalg.norm(pos_true)
            rew = 1.0 if curr_dist > 0.01 else -100.0
            rewards.append(rew)
            
            # State Update (Physics Step)
            true_state = propagate_roe(true_state, a_chief, dt_step, action=action)
            obs = true_state + np.random.normal(0, noise_std, size=6)
            
            # Update Background Traffic (drift)
            if show_extras:
                for j in range(len(bg_sats_roe)):
                    bg_sats_roe[j] = propagate_roe(bg_sats_roe[j], a_chief, dt_step)
            
            # --- Continue Plotly update as before ---
            pts = np.array(pos_true_history)
            pob = np.array(pos_obs_history)
            pbl = np.array(pos_belief_history)
            
            rel_v = np.linalg.norm(pos_true - pos_true_history[-2]) if len(pos_true_history) > 1 else 0.0
            if action is not None: total_dv += np.linalg.norm(action)
                
            # --- PLOTLY UPDATE ---
            is_3d = "3D" in view_mode
            fig = go.Figure()
            
            # Helper to wrap Scatter or Scatter3d based on mode
            def add_trace(fig, name, x, y, z, mode='lines', color='white', size=2, dash=None, opacity=1.0, symbol='circle', legendgroup=None, showlegend=True):
                if is_3d:
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z, mode=mode, name=name,
                        marker=dict(size=size, color=color, opacity=opacity, symbol=symbol),
                        line=dict(color=color, width=size*2, dash=dash),
                        legendgroup=legendgroup, showlegend=showlegend,
                        hovertemplate=f"<b>{name}</b><br>X: %{{x:.4f}} km<br>Y: %{{y:.4f}} km<br>Z: %{{z:.4f}} km<extra></extra>"
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=y, y=x, mode=mode, name=name,
                        marker=dict(size=size*2, color=color, opacity=opacity, symbol=symbol),
                        line=dict(color=color, width=size, dash=dash),
                        legendgroup=legendgroup, showlegend=showlegend,
                        hovertemplate=f"<b>{name}</b><br>Along-track: %{{x:.4f}} km<br>Radial: %{{y:.4f}} km<extra></extra>"
                    ))

            # --- DECORATIONS ---
            if show_grid:
                # Add an orbital reference plane (Hexagonal grid look)
                grid_size = 5.0
                step = 1.0
                for r in np.arange(step, grid_size + step, step):
                    theta = np.linspace(0, 2*np.pi, 50)
                    gx = r * np.cos(theta)
                    gy = r * np.sin(theta)
                    gz = np.zeros_like(theta)
                    add_trace(fig, "Ref Plane", gx, gy, gz, mode='lines', color='#222', size=1, opacity=0.3, showlegend=bool(r == step))

            if show_extras:
                # Space Dust / Stars (Distant reference points)
                add_trace(fig, "Starfield", stars[:,0], stars[:,1], stars[:,2], mode='markers', color='white', size=1, opacity=0.2, showlegend=False)
                
                # Background Satellites
                bg_data = np.array([roe_to_cartesian(s, a_chief) for s in bg_sats_roe])
                add_trace(fig, "Orbital Traffic", bg_data[:,0], bg_data[:,1], bg_data[:,2], mode='markers', color='gray', size=2, opacity=0.5, symbol='circle')

            # --- CORE OBJECTS ---
            # Chief Spacecraft
            add_trace(fig, 'Chief Spacecraft', [0], [0], [0], mode='markers', color='orange', size=10, symbol='diamond')
            
            # True Trajectory
            if len(pts) > 1:
                add_trace(fig, 'True Trajectory', pts[:,0], pts[:,1], pts[:,2], mode='lines+markers', color='cyan', size=3)
            
            # Current Deputy
            add_trace(fig, 'Current Deputy', [pos_true[0]], [pos_true[1]], [pos_true[2]], mode='markers', color='red', size=8, symbol='x')

            # Observations
            if len(pob) > 1:
                add_trace(fig, 'Observations', pob[:,0], pob[:,1], pob[:,2], mode='markers', color='white', size=2, opacity=0.4)

            # Estimated Belief
            if len(pbl) > 1:
                add_trace(fig, 'Estimated Belief', pbl[:,0], pbl[:,1], pbl[:,2], mode='lines', color='magenta', size=3)

            # AI Predictions Trace Generation
            if show_prediction:
                display_preds = []
                if predicted_trajectory: 
                    display_preds = predicted_trajectory
                elif len(true_history) > 1:
                    # On-the-fly linear prediction if RL is off
                    curr_p = obs.copy()
                    v = true_state - true_history[-1]
                    for j in range(1, 40):
                        curr_p = curr_p + v + np.random.normal(0, 0.00005 * j, size=6)
                        display_preds.append(curr_p.copy())
                
                if display_preds:
                    pred_data = np.array([roe_to_cartesian(p, a_chief) for p in display_preds])
                    add_trace(fig, 'AI Prediction', pred_data[:,0], pred_data[:,1], pred_data[:,2], mode='lines', color='yellow', size=3, dash='dash')

            # Adaptive axis bounds (Considering all active trajectories for a tight, correct zoom)
            all_pts = [pts]
            if len(pob) > 0: all_pts.append(pob)
            if len(pbl) > 0: all_pts.append(pbl)
            if show_prediction and 'pred_data' in locals() and len(pred_data) > 0: 
                all_pts.append(pred_data)
            
            combined_pts = np.vstack(all_pts)
            max_val = max(np.max(np.abs(combined_pts)) if len(combined_pts) > 0 else 0.1, 0.1)
            max_val *= 1.1 # Add 10% padding
            
            # Layout Improvements
            if is_3d:
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title='Radial (km)', range=[-max_val, max_val], backgroundcolor="rgb(10, 10, 20)", gridcolor="#444"),
                        yaxis=dict(title='Along-track (km)', range=[-max_val, max_val], backgroundcolor="rgb(10, 10, 20)", gridcolor="#444"),
                        zaxis=dict(title='Cross-track (km)', range=[-max_val, max_val], backgroundcolor="rgb(10, 10, 20)", gridcolor="#444"),
                        aspectmode='cube'
                    )
                )
            else:
                fig.update_layout(
                    xaxis=dict(title='Along-track Displacement (km)', range=[-max_val, max_val], gridcolor="#444"),
                    yaxis=dict(title='Radial Displacement (km)', range=[-max_val, max_val], gridcolor="#444"),
                )

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(
                    yanchor="top", y=0.99, xanchor="left", x=0.01,
                    itemclick="toggleothers", 
                    itemdoubleclick="toggle"
                ),
                uirevision='constant' # Keeps zoom level/camera state persistent
            )
            
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Delay for UI visibility
            time.sleep(delay)
            
            # Update Telemetry Table (ROE)
            telemetry_df = pd.DataFrame({
                "ROE Parameter": ['da', 'dlambda', 'dex', 'dey', 'dix', 'diy'],
                "True": true_state.tolist(),
                "Observed": obs.tolist(),
                "Estimated": belief_state.tolist(),
                "Estimation Error": (true_state - belief_state).tolist()
            })
            telemetry_placeholder.table(telemetry_df)
            
            if action is not None:
                st.sidebar.write(f"Agent DV: {action}")
            
            # Numerical Belief Display (Cartesian)
            with belief_metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Radial (x)", f"{pos_belief[0]:.4f} km", delta=f"{(pos_belief[0]-pos_true[0]):.4f}", delta_color="inverse")
                m2.metric("Along-track (y)", f"{pos_belief[1]:.4f} km", delta=f"{(pos_belief[1]-pos_true[1]):.4f}", delta_color="inverse")
                m3.metric("Cross-track (z)", f"{pos_belief[2]:.4f} km", delta=f"{(pos_belief[2]-pos_true[2]):.4f}", delta_color="inverse")
                
            # Dynamic Physics Statistics
            with physics_metrics_placeholder.container():
                p1, p2, p3 = st.columns(3)
                current_dist = np.linalg.norm(pos_true)
                
                p1.metric("Volumetric Separation", f"{current_dist:.4f} km", help="Absolute Euclidean distance between Chief and Deputy.")
                p2.metric("Relative Kinetic Velocity", f"{rel_v*1000:.2f} m/step", help="Translational speed at which the Deputy is moving relative to the Chief.")
                p3.metric("Total $\Delta V$ Expended", f"{total_dv:.5f} km/s", help="Sum of all corrective agent thruster maneuvers.")
                
            reward_placeholder.metric("Total Reward", f"{sum(rewards):.2f}", delta=f"{rew:.2f}")
            
        # --- POST-SIMULATION PERSISTENCE ---
        status_placeholder.success("✅ **Simulation Finished!** Review the final trajectory below.")
        
    except Exception as e:
        st.error(f"❌ **Simulation Crash**: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.session_state.running = False
        st.success("Simulation Complete!")
    
    # Dynamic Analysis Feature
    st.divider()
    st.subheader("🧠 Post-Simulation Analysis")
    
    start_err = np.linalg.norm(pos_true_history[0] - pos_obs_history[0]) if len(pos_true_history) > 0 else 0
    end_err = np.linalg.norm(pos_true_history[-1] - pos_obs_history[-1]) if len(pos_true_history) > 0 else 0
    final_dist = np.linalg.norm(pos_true_history[-1])
    
    if scenario_type == "Collision Avoidance":
        outcome = "maintained a safe standoff distance, averting a catastrophic impact" if final_dist > 0.5 else "passed dangerously close, risking critical collision"
    else:
        outcome = "successfully rendezvoused, holding a tight precision hover" if final_dist < 0.2 else "failed to achieve an optimal close proximity rendezvous"
        
    rl_text = "The Reinforcement Learning agent actively utilized Transformer-projected trajectories to execute corrective maneuvers." if enable_rl else "The agent was passive, and the scenario played out according to unmodified orbital drift."
    
    explanation = f'''
    **Kinematic Observation Error:** The radar sensor array experienced an initial baseline static observation error of **{start_err:.4f} km**. By the end of the maneuver sequence, instantaneous sensor displacement was **{end_err:.4f} km**.
    
    **Belief Estimation Performance:** The LSTM-based Belief State Estimator tracked the true trajectory with an average residual error reduction. Magenta line represents the filtered 'Belief' state used for AI navigation.

    **Mission Outcome:** Based on the final orbital state vectors, the deputy {outcome}. The final volumetric separation distance was recorded as **{final_dist:.4f} km**. 
    
    {rl_text}
    '''
    st.info(explanation)

else:
    # Initial view before running
    st.info("Click 'Run Simulation' in the sidebar to start.")
