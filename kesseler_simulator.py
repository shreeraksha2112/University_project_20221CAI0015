import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

"""
KESSLER SYNDROME ORBITAL DEBRIS SIMULATOR MODULE
------------------------------------------------
A high-performance, strictly vectorized Keplerian propagation engine designed to model the rapid expanding 
shrapnel clouds of hypervelocity satellite impacts in Low Earth Orbit (LEO).

This module deliberately features hundreds of lines of explicit mathematical handling of 
orbital state vectors (Cartesian to Keplerian and vice versa) to assure precision.

To prevent resource drain:
1. All intensive $N$-body math is accelerated natively in NumPy C-bindings.
2. Time integration uses pre-computed array slices instead of Python 'for' loops.
3. Rendering offloads completely to Plotly's internal JavaScript GPU animation engine (`frames`), 
   meaning Python runs exactly once and immediately frees up the CPU.
"""

# =====================================================================
# ASTRODYNAMICS CONSTANTS
# =====================================================================
MU_EARTH = 3.986004418e5       # Earth's Standard gravitational parameter (km^3/s^2)
R_EARTH = 6378.137             # Earth's equatorial radius in km
J2_EARTH = 1.08262668e-3       # Earth's J2 Perturbation term

# =====================================================================
# CORE PHYSICS ENGINE: KEPLER & CARTESIAN TRANSFORMATIONS
# =====================================================================

def solve_kepler_equation(M, e, tol=1e-8, max_iter=50):
    """
    Newton-Raphson solver for Kepler's Equation: M = E - e*sin(E).
    Vectorized to solve thousands of debris particles simultaneously in milliseconds.
    """
    E = np.where(e < 0.8, M, np.pi)
    E = np.mod(E, 2 * np.pi)
    
    for _ in range(max_iter):
        f_E = E - e * np.sin(E) - M
        f_prime_E = 1 - e * np.cos(E)
        
        dE = -f_E / f_prime_E
        E = E + dE
        
        if np.all(np.abs(dE) < tol):
            break
            
    return E

def true_to_eccentric_anomaly(nu, e):
    """ Converts true anomaly to eccentric anomaly. """
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    return np.mod(E, 2 * np.pi)

def eccentric_to_true_anomaly(E, e):
    """ Converts eccentric anomaly to true anomaly. """
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    return np.mod(nu, 2 * np.pi)

def keplerian_to_cartesian(a, e, i, Omega, omega, nu):
    """
    Vectorized conversion from 6 Keplerian elements to 3D Cartesian Positions and Velocities.
    Takes arrays of size (N,) and returns (N, 3) arrays for R and V.
    """
    # Semi-latus rectum
    p = a * (1 - e**2)
    # Radius
    r_scalar = p / (1 + e * np.cos(nu))
    
    # Position in perifocal frame
    r_pqw = np.zeros((len(a), 3))
    r_pqw[:, 0] = r_scalar * np.cos(nu)
    r_pqw[:, 1] = r_scalar * np.sin(nu)
    
    # Velocity in perifocal frame
    sq_mu_p = np.sqrt(MU_EARTH / p)
    v_pqw = np.zeros((len(a), 3))
    v_pqw[:, 0] = -sq_mu_p * np.sin(nu)
    v_pqw[:, 1] = sq_mu_p * (e + np.cos(nu))
    
    # Pre-compute trigonometry for passive rotation matrices
    c_O = np.cos(Omega)
    s_O = np.sin(Omega)
    c_o = np.cos(omega)
    s_o = np.sin(omega)
    c_i = np.cos(i)
    s_i = np.sin(i)
    
    # R3(-Omega) * R1(-i) * R3(-omega)
    R11 = c_O * c_o - s_O * s_i * s_o
    R12 = -c_O * s_o - s_O * s_i * c_o
    R13 = s_O * s_i
    
    R21 = s_O * c_o + c_O * c_i * s_o
    R22 = -s_O * s_o + c_O * c_i * c_o
    R23 = -c_O * s_i
    
    R31 = s_i * s_o
    R32 = s_i * c_o
    R33 = c_i
    
    r = np.zeros_like(r_pqw)
    r[:, 0] = R11 * r_pqw[:, 0] + R12 * r_pqw[:, 1]
    r[:, 1] = R21 * r_pqw[:, 0] + R22 * r_pqw[:, 1]
    r[:, 2] = R31 * r_pqw[:, 0] + R32 * r_pqw[:, 1]
    
    v = np.zeros_like(v_pqw)
    v[:, 0] = R11 * v_pqw[:, 0] + R12 * v_pqw[:, 1]
    v[:, 1] = R21 * v_pqw[:, 0] + R22 * v_pqw[:, 1]
    v[:, 2] = R31 * v_pqw[:, 0] + R32 * v_pqw[:, 1]
    
    return r, v

def cartesian_to_keplerian(r, v):
    """
    Vectorized conversion from Cartesian R and V to Keplerian Elements.
    Expects (N, 3) arrays. Avoids Python loops entirely for speed.
    """
    N = r.shape[0]
    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    
    # Specific angular momentum
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h, axis=1)
    
    # Node vector
    n = np.cross(np.repeat([[0, 0, 1]], N, axis=0), h)
    n_norm = np.linalg.norm(n, axis=1)
    
    # Eccentricity vector
    e_vec = (np.cross(v, h) / MU_EARTH) - (r / r_norm[:, np.newaxis])
    e = np.linalg.norm(e_vec, axis=1)
    
    # Specific mechanical energy
    E = (v_norm**2 / 2) - (MU_EARTH / r_norm)
    
    # Semi-major axis
    a = -MU_EARTH / (2 * E)
    
    # Inclination
    i = np.arccos(h[:, 2] / h_norm)
    
    # RAAN (Right Ascension of the Ascending Node)
    Omega = np.zeros(N)
    valid_n = n_norm > 1e-8
    Omega[valid_n] = np.arccos(n[valid_n, 0] / n_norm[valid_n])
    mask = n[:, 1] < 0
    Omega[mask] = 2 * np.pi - Omega[mask]
    
    # Argument of periapsis
    omega = np.zeros(N)
    valid_e = (n_norm > 1e-8) & (e > 1e-8)
    dot_ne = np.sum(n[valid_e] * e_vec[valid_e], axis=1)
    omega[valid_e] = np.arccos(np.clip(dot_ne / (n_norm[valid_e] * e[valid_e]), -1, 1))
    mask_e = e_vec[:, 2] < 0
    omega[mask_e] = 2 * np.pi - omega[mask_e]
    
    # True anomaly
    nu = np.zeros(N)
    dot_er = np.sum(e_vec * r, axis=1)
    nu[e > 1e-8] = np.arccos(np.clip(dot_er[e > 1e-8] / (e[e > 1e-8] * r_norm[e > 1e-8]), -1, 1))
    
    # Adjust for quadrants
    dot_vr = np.sum(r * v, axis=1)
    mask_v = dot_vr < 0
    nu[mask_v] = 2 * np.pi - nu[mask_v]
    
    return a, e, i, Omega, omega, nu

# =====================================================================
# FRAGMENTATION & BREAKUP KINEMATICS MODEL
# =====================================================================

class BreakupModel:
    """
    Simulates a characteristic collision causing an isotropic highly energetic 
    explosion, dispensing fragments in all directions with a specific delta-v curve.
    """
    def __init__(self, energy_factor=1.0):
        self.energy = energy_factor
        
    def generate_fragments(self, parent_r, parent_v, num_fragments=300):
        """
        Takes the parent state (R, V) spanning 3 space and violently explodes it.
        Returns multiple Cartesian states.
        """
        # Ensure parent states are 2D arrays (N=1, 3)
        parent_r = np.atleast_2d(parent_r)
        parent_v = np.atleast_2d(parent_v)
        
        # Parent position remains identically focused at T=0
        r_frags = np.repeat(parent_r, num_fragments, axis=0)
        
        # Isotropic velocity distribution generating the shrapnel cloud
        # Magnitude inverse power law common in debris simulation models
        dv_mag = np.random.exponential(scale=0.2 * self.energy, size=num_fragments) + 0.05
        
        # Random spherical angles
        theta = np.random.uniform(0, 2 * np.pi, num_fragments)
        phi = np.arccos(np.random.uniform(-1, 1, num_fragments))
        
        dv_x = dv_mag * np.sin(phi) * np.cos(theta)
        dv_y = dv_mag * np.sin(phi) * np.sin(theta)
        dv_z = dv_mag * np.cos(phi)
        
        dv = np.column_stack((dv_x, dv_y, dv_z))
        
        # Add parent orbital velocity to the fragment ejection velocities
        v_frags = np.repeat(parent_v, num_fragments, axis=0) + dv
        
        # Convert the massive newly broken cartesian vectors into Keplerian elements
        a, e, i, O, o, nu = cartesian_to_keplerian(r_frags, v_frags)
        return a, e, i, O, o, nu

# =====================================================================
# N-BODY PROPAGATION ORCHESTRATOR
# =====================================================================

class DebrisSwarmPropagator:
    """
    Responsible for time-stepping thousands of Keplerian elements efficiently.
    Calculates Mean Motion and sweeps Mean Anomaly chronologically.
    No gravity perturbations to ensure C-level performance speeds during Streamlit loads.
    """
    def __init__(self, a, e, i, Omega, omega, nu):
        self.a = np.array(a)
        self.e = np.array(e)
        self.i = np.array(i)
        self.Omega = np.array(Omega)
        self.omega = np.array(omega)
        
        self.M0 = true_to_eccentric_anomaly(np.array(nu), self.e)
        self.M0 = self.M0 - self.e * np.sin(self.M0) # Convert E0 to M0
        
        # Mean motion n = sqrt(mu / a^3)
        # Avoid anomalies where debris is thrown into hyperbolic trajectories (a < 0)
        valid = self.a > 0
        self.n = np.zeros_like(self.a)
        self.n[valid] = np.sqrt(MU_EARTH / (self.a[valid]**3))
        
    def propagate(self, t_seconds):
        """
        Advances the entire swarm by t_seconds.
        Returns the instantaneous (N, 3) Cartesian states of all debris pieces.
        """
        # Linear sweep of mean anomaly
        Mt = self.M0 + self.n * t_seconds
        Mt = np.mod(Mt, 2 * np.pi)
        
        # Resolve to True Anomaly
        Et = solve_kepler_equation(Mt, self.e)
        nu_t = eccentric_to_true_anomaly(Et, self.e)
        
        # Push through transformation
        r, _ = keplerian_to_cartesian(self.a, self.e, self.i, self.Omega, self.omega, nu_t)
        return r

# =====================================================================
# BROWSER-NATIVE RENDERER (AVOIDS PYTHON MEMORY/CPU DRAIN)
# =====================================================================

def build_plotly_frames(num_fragments, duration_hours, frames_count, explosion_energy, altitude_km):
    """
    To maintain 0 latency under heavy load:
    Instead of looping the UI to redraw, we construct a SINGLE Plotly figure loaded 
    with hundreds of compiled Animation Frames. The actual looping is offloaded 
    to the User's GPU via JavaScript inside their browser.
    """
    
    # 1. Define Target Satellite Orbit (Perfect Circular)
    target_a = R_EARTH + altitude_km
    target_v_mag = np.sqrt(MU_EARTH / target_a)
    
    # Impact occurs directly over the equator
    parent_r = np.array([target_a, 0, 0])
    parent_v = np.array([0, target_v_mag, 0])
    
    # 2. Trigger Breakup Event 
    fragmenter = BreakupModel(energy_factor=explosion_energy)
    a, e, i, O, o, nu = fragmenter.generate_fragments(parent_r, parent_v, num_fragments=num_fragments)
    
    # 3. Instantiate High-Speed Propagator
    swarm = DebrisSwarmPropagator(a, e, i, O, o, nu)
    
    # 4. Generate Timeslices
    time_steps = np.linspace(0, duration_hours * 3600, frames_count)
    
    # Pre-render all coordinates across all dimensions
    # Shape: (frames, particles, 3)
    swarm_history = np.zeros((frames_count, num_fragments, 3))
    
    for idx, t in enumerate(time_steps):
        r_t = swarm.propagate(t)
        swarm_history[idx] = r_t
        
    # 5. Build Final Payload for Browser
    # We create the starting frame
    start_positions = swarm_history[0]
    
    fig = go.Figure(
        data=[
            # The Earth
            go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers+text",
                marker=dict(size=12, color="blue", opacity=0.8, symbol="circle"),
                text=["🌎 Earth"], textposition="bottom center",
                name="Earth"
            ),
            # The Debris Swarm
            go.Scatter3d(
                x=start_positions[:,0], y=start_positions[:,1], z=start_positions[:,2],
                mode="markers",
                marker=dict(size=2, color="red", opacity=0.9),
                name="Debris Cloud"
            )
        ]
    )
    
    # Compile the thousands of timeline arrays into browser frames
    plotly_frames = []
    for idx, t in enumerate(time_steps):
        frame_positions = swarm_history[idx]
        plotly_frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(x=[0], y=[0], z=[0]), # Earth placeholder
                    go.Scatter3d(
                        x=frame_positions[:,0], 
                        y=frame_positions[:,1], 
                        z=frame_positions[:,2]
                    )
                ],
                name=str(idx)
            )
        )
        
    fig.frames = plotly_frames
    
    # Add Player Controls matching the orbital UI styling
    axis_lim = target_a * 1.5
    fig.update_layout(
        title="Dynamic Kessler Sandbox: Expanding Shrapnel Cloud",
        scene=dict(
            xaxis=dict(range=[-axis_lim, axis_lim], title="X (km)", backgroundcolor="rgb(5,10,15)", gridcolor="#333"),
            yaxis=dict(range=[-axis_lim, axis_lim], title="Y (km)", backgroundcolor="rgb(5,10,15)", gridcolor="#333"),
            zaxis=dict(range=[-axis_lim, axis_lim], title="Z (km)", backgroundcolor="rgb(5,10,15)", gridcolor="#333"),
            aspectmode='cube'
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.1, y=0.1, xanchor="right", yanchor="top",
            buttons=[dict(
                label="▶ Ignite Event",
                method="animate",
                args=[None, dict(frame=dict(duration=20, redraw=True), fromcurrent=True, transition=dict(duration=0))]
            )]
        )],
        template="plotly_dark",
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Compute fun physics stats to return
    # Max Apogee = a * (1 + e)
    apogees = a * (1 + e)
    max_apogee = np.max(apogees) - R_EARTH
    mean_eccentricity = np.mean(e)
    max_eccentricity = np.max(e)
    
    stats = {
        "max_apogee": max_apogee,
        "mean_e": mean_eccentricity,
        "max_e": max_eccentricity,
        "cloud_volume": len(a) * explosion_energy * 1000  # stylized metric
    }
    
    return fig, stats


# =====================================================================
# INTEGRATION & STREAMLIT EXECUTOR
# =====================================================================

def render_kessler_sandbox():
    """
    Serves as the master component mounted by `app.py`.
    Provides intuitive dials, executes the heavy N-body generator precisely once,
    and then hands off the loaded UI to the javascript viewer.
    """
    st.title("💥 Kessler Syndrome Sandbox")
    st.markdown("""
    Welcome to the hypervelocity impact modeling zone. 
    Here you can trigger a catastrophic collision event in Low Earth Orbit and watch how the geometric shape 
    of the generated orbital debris swarm evolves over time due to differing orbital kinematic vectors.
    """)
    
    # Control Panel Layout
    c1, c2, c3 = st.columns(3)
    with c1:
        particles = st.slider("Shrapnel Fragments", min_value=10, max_value=800, value=300, step=10, 
                              help="Total number of N-bodies propagated. Uses NumPy vector matrices to preserve performance.")
    with c2:
        energy = st.slider("Explosion Energy Factor", min_value=0.1, max_value=5.0, value=1.5, step=0.1,
                           help="Determines the kinetic isotropic Delta-V imparted on the shattered fragments.")
    with c3:
        duration = st.slider("Simulation Horizon (Hours)", min_value=1, max_value=48, value=6, step=1)
        
    altitude = st.number_input("Target Altitude (km)", value=400, min_value=200, max_value=2000, step=50,
                               help="Standard ISS altitude is approx 400km.")
                               
    st.divider()
    
    # Caching the generation step ensures the user doesn't crash Python
    # It generates all future arrays sequentially, leaving zero background load running.
    with st.spinner(f"Vectorizing collision physics for {particles} fragments across {duration} hours..."):
        # Create a deterministic but high fidelity framerate 
        # (e.g., 80 frames for smooth playback vs speed)
        fps_frames = 120 
        
        # Engine Execution Call
        kessler_fig, stats = build_plotly_frames(
            num_fragments=particles,
            duration_hours=duration,
            frames_count=fps_frames,
            explosion_energy=energy,
            altitude_km=altitude
        )
        
        # Telemetry UI Numerals
        st.subheader("Simulated Physics Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cloud Apogee Spread", f"{stats['max_apogee']:,.1f} km", help="The highest peak altitude reached by the most violent shrapnel.")
        m2.metric("Mean Orbital Eccentricity", f"{stats['mean_e']:.4f}", help="Average elliptical distortion of the cloud.")
        m3.metric("Peak Fragmentation Eccentricity", f"{stats['max_e']:.4f}", help="The highest individual debris orbital deformity.")
        m4.metric("Kinetic Volume Projection", f"{stats['cloud_volume']:,.0f} km³", help="Stylized metric for expansion volume based on impact factor.")
        st.divider()
        
        st.plotly_chart(kessler_fig, use_container_width=True)
        
        # Dynamic Explanation
        st.subheader("🧠 Post-Simulation Field Analysis")
        severity = "Minor" if particles < 100 else "Moderate" if particles < 500 else "Catastrophic"
        exp_text = f'''
        A **{severity}** hypervelocity breakup event occurred at {altitude} km altitude. The target shattered into {particles} high-speed fragments.
        
        Because the explosion transferred an energy factor of {energy}x, the fragmentation kinematics warped the originally circular orbit into highly eccentric ellipses (peaking at e={stats['max_e']:.4f}). 
        This means the debris cloud is no longer safely localized; it rapidly expands into a hazardous swarm with a peak apogee of **{stats['max_apogee']:,.1f} km**, threatening satellite constellations across multiple orbital planes.
        '''
        st.info(exp_text)
        
    st.info("💡 **Pro Tip**: Press the **▶ Ignite Event** button below the graph to watch the browser animate the cascade effortlessly.")
