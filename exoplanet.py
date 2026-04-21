import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ==========================================================
# CONFIGURATION SECTION
# ==========================================================

st.set_page_config(
    page_title="Exoplanet Explorer",
    layout="wide",
    page_icon="🌌"
)

DATA_PATH = 'DATASET/exoplanets_min.csv'


# ==========================================================
# DATA LOADING
# ==========================================================

@st.cache_data
def load_exoplanet_data():
    """
    Loads the dataset from CSV.
    Applies basic validation and cleaning.
    """
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # Drop rows without essential orbital data
    df = df.dropna(subset=['pl_name', 'pl_orbsmax', 'pl_orbeccen'])

    return df


# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

def sidebar_controls(df):
    """
    Handles all sidebar UI components.
    """
    st.sidebar.title("🔭 Controls")

    # Planet selection
    planet_list = sorted(df['pl_name'].unique().tolist())
    selected_planet = st.sidebar.selectbox("Select Exoplanet", planet_list)

    # Visualization controls
    st.sidebar.subheader("Visualization Settings")

    resolution = st.sidebar.slider("Orbit Resolution", 100, 1000, 300)
    scale_factor = st.sidebar.slider("Scale Orbit Size", 0.5, 5.0, 1.0)
    show_grid = st.sidebar.checkbox("Show Grid", True)
    show_labels = st.sidebar.checkbox("Show Labels", True)

    return selected_planet, resolution, scale_factor, show_grid, show_labels


# ==========================================================
# ORBIT CALCULATION
# ==========================================================

def compute_orbit(a, e, resolution):
    """
    Computes orbital coordinates using Kepler's equation.
    """
    theta = np.linspace(0, 2 * np.pi, resolution)

    if e < 1:
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
    else:
        r = np.full_like(theta, a)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(theta)

    return x, y, z


# ==========================================================
# METRICS DISPLAY
# ==========================================================

def display_metrics(planet_data):
    """
    Displays telemetry and discovery data.
    """
    st.subheader("System Telemetry")

    a = planet_data['pl_orbsmax']
    e = planet_data['pl_orbeccen']

    st.metric("Semi-Major Axis (AU)", f"{a:.4f}")
    st.metric("Eccentricity", f"{e:.4f}")

    mass = planet_data['pl_bmasse']
    radius = planet_data['pl_rade']

    st.metric("Mass (Earth)", "Unknown" if pd.isna(mass) else f"{mass:.2f}")
    st.metric("Radius (Earth)", "Unknown" if pd.isna(radius) else f"{radius:.2f}")

    st.divider()

    st.subheader("Discovery Log")
    st.write(f"Host Star: {planet_data['hostname']}")
    st.write(f"Year: {planet_data['disc_year']}")
    st.write(f"Method: {planet_data['discoverymethod']}")
    st.write(f"Facility: {planet_data['disc_facility']}")


# ==========================================================
# PLOT CREATION
# ==========================================================

def create_3d_plot(x, y, z, planet_data, selected_planet, show_grid, show_labels, scale):
    """
    Creates a 3D orbit visualization using Plotly.
    """

    # Apply scaling
    x = x * scale
    y = y * scale
    z = z * scale

    fig = go.Figure()

    # Host star
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text' if show_labels else 'markers',
        marker=dict(size=15, color='yellow'),
        text=[f"⭐ {planet_data['hostname']}"],
        name="Host Star"
    ))

    # Orbit path
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(width=3),
        name="Orbit"
    ))

    # Planet position
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers+text' if show_labels else 'markers',
        marker=dict(size=8, color='red'),
        text=[selected_planet],
        name="Planet"
    ))

    max_bound = max(abs(x).max(), abs(y).max()) * 1.2

    fig.update_layout(
        title=f"Orbit Visualization: {selected_planet}",
        scene=dict(
            xaxis=dict(
                title='X (AU)',
                range=[-max_bound, max_bound],
                showgrid=show_grid
            ),
            yaxis=dict(
                title='Y (AU)',
                range=[-max_bound, max_bound],
                showgrid=show_grid
            ),
            zaxis=dict(
                title='Z (AU)',
                range=[-max_bound, max_bound],
                showgrid=show_grid
            ),
            aspectmode='cube'
        ),
        template="plotly_dark",
        height=650
    )

    return fig


# ==========================================================
# MAIN APP
# ==========================================================

def render_exoplanet_explorer():
    """
    Main function to render the Streamlit app.
    """

    st.title("🌌 Exoplanet Orbit Explorer")
    st.markdown("Interactive visualization of real exoplanet systems.")

    df = load_exoplanet_data()

    if df.empty:
        return

    selected_planet, resolution, scale, show_grid, show_labels = sidebar_controls(df)

    planet_data = df[df['pl_name'] == selected_planet].iloc[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        display_metrics(planet_data)

    with col2:
        x, y, z = compute_orbit(
            planet_data['pl_orbsmax'],
            planet_data['pl_orbeccen'],
            resolution
        )

        fig = create_3d_plot(
            x, y, z,
            planet_data,
            selected_planet,
            show_grid,
            show_labels,
            scale
        )

        st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":
    render_exoplanet_explorer()
