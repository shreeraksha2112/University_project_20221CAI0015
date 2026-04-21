import numpy as np
from orbital_env import OrbitalEnvironment
from astropy import units as u
from hapsira.twobody import Orbit
from hapsira.bodies import Earth

# ==========================================================
# CONFIGURATION & CONSTANTS
# ==========================================================

DEFAULT_ALTITUDE = 500 * u.km
DEFAULT_NOISE_STD = 0.001

# Thresholds for scenario classification
COLLISION_DISTANCE_THRESHOLD = 0.5  # km (example)
RENDEZVOUS_DISTANCE_THRESHOLD = 0.05  # km

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def validate_roe(roe):
    """
    Validates the Relative Orbital Elements array.
    ROE format: [da, dlambda, dex, dey, dix, diy]
    """
    if len(roe) != 6:
        raise ValueError("ROE must contain exactly 6 elements")

    if not isinstance(roe, np.ndarray):
        raise TypeError("ROE must be a numpy array")

    return True


def log_scenario_details(name, roe, noise_std):
    """
    Logs scenario parameters for debugging and traceability.
    """
    print(f"\n===== {name.upper()} SCENARIO =====")
    print(f"ROE: {roe}")
    print(f"Noise Std Dev: {noise_std}")
    print("====================================\n")


def create_chief_orbit(altitude=DEFAULT_ALTITUDE):
    """
    Creates a circular orbit for the chief spacecraft.
    """
    return Orbit.circular(Earth, alt=altitude)


# ==========================================================
# SCENARIO GENERATOR CLASS
# ==========================================================

class ScenarioGenerator:
    """
    Generates orbital scenarios for:
    1. Collision Avoidance
    2. Rendezvous Operations
    3. Custom Scenarios (extendable)

    Designed for reinforcement learning and simulation testing.
    """

    def __init__(self, default_noise=DEFAULT_NOISE_STD):
        self.default_noise = default_noise

    # ------------------------------------------------------
    # COLLISION SCENARIO
    # ------------------------------------------------------

    def generate_collision_scenario(self, noise_std=None):
        """
        Generates a high-risk collision trajectory.

        Characteristics:
        - High relative velocity
        - Non-zero eccentricity (curved trajectory)
        - Moderate separation for fast approach
        """

        if noise_std is None:
            noise_std = self.default_noise

        # Chief orbit
        chief_orbit = create_chief_orbit()

        # Define aggressive collision ROE
        da = 0.15       # semi-major axis difference (km-scale)
        dlambda = 1.0   # along-track separation
        dex = 0.05      # eccentricity x-component
        dey = 0.02      # eccentricity y-component
        dix = 0.0       # inclination x
        diy = 0.0       # inclination y

        deputy_roe = np.array([da, dlambda, dex, dey, dix, diy])

        # Validate ROE
        validate_roe(deputy_roe)

        # Log details
        log_scenario_details("Collision", deputy_roe, noise_std)

        # Create environment
        env = OrbitalEnvironment(
            chief_orbit=chief_orbit,
            deputy_roe=deputy_roe,
            noise_std=noise_std
        )

        return env

    # ------------------------------------------------------
    # RENDEZVOUS SCENARIO
    # ------------------------------------------------------

    def generate_rendezvous_scenario(self, noise_std=None):
        """
        Generates a controlled rendezvous scenario.

        Characteristics:
        - Large initial separation
        - Complex orbital geometry
        - Requires precise maneuvering
        """

        if noise_std is None:
            noise_std = self.default_noise

        chief_orbit = create_chief_orbit()

        # Challenging rendezvous configuration
        da = -0.05
        dlambda = 3.5
        dex = 0.15
        dey = 0.1
        dix = 0.08
        diy = 0.05

        deputy_roe = np.array([da, dlambda, dex, dey, dix, diy])

        validate_roe(deputy_roe)
        log_scenario_details("Rendezvous", deputy_roe, noise_std)

        env = OrbitalEnvironment(
            chief_orbit=chief_orbit,
            deputy_roe=deputy_roe,
            noise_std=noise_std
        )

        return env

    # ------------------------------------------------------
    # CUSTOM SCENARIO
    # ------------------------------------------------------

    def generate_custom_scenario(self, roe, noise_std=None):
        """
        Allows user-defined ROE input for flexible simulations.
        """

        if noise_std is None:
            noise_std = self.default_noise

        validate_roe(roe)

        chief_orbit = create_chief_orbit()

        log_scenario_details("Custom", roe, noise_std)

        env = OrbitalEnvironment(
            chief_orbit=chief_orbit,
            deputy_roe=roe,
            noise_std=noise_std
        )

        return env

    # ------------------------------------------------------
    # SCENARIO SUMMARY
    # ------------------------------------------------------

    def describe_scenario(self, roe):
        """
        Provides a qualitative description of scenario type.
        """

        da, dlambda, dex, dey, dix, diy = roe

        if abs(da) > 0.1 and abs(dlambda) < 2:
            return "High-speed collision risk scenario"

        elif abs(dlambda) > 2:
            return "Long-range rendezvous scenario"

        elif abs(dex) > 0.1 or abs(dey) > 0.1:
            return "Elliptical transfer trajectory"

        else:
            return "Nominal orbital configuration"


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def run_demo():
    """
    Runs demo scenarios and prints initial states.
    """

    generator = ScenarioGenerator()

    # Generate collision scenario
    collision_env = generator.generate_collision_scenario()
    collision_state = collision_env.reset()

    print("Collision Scenario Initial State:")
    print(collision_state)

    # Generate rendezvous scenario
    rendezvous_env = generator.generate_rendezvous_scenario()
    rendezvous_state = rendezvous_env.reset()

    print("\nRendezvous Scenario Initial State:")
    print(rendezvous_state)

    # Custom example
    custom_roe = np.array([0.02, 1.0, 0.01, 0.01, 0.0, 0.0])
    custom_env = generator.generate_custom_scenario(custom_roe)

    print("\nCustom Scenario Description:")
    print(generator.describe_scenario(custom_roe))


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run_demo()
