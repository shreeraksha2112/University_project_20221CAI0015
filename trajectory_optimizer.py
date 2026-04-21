import numpy as np
from scipy.optimize import minimize


class SCPOptimizer:
    """
    Sequential Convex Programming (SCP) optimizer using SLSQP.
    Refines an initial action while enforcing safety constraints.
    """

    def __init__(
        self,
        action_dim=3,
        safe_distance=0.01,
        max_iter=50,
        tol=1e-4,
        verbose=False
    ):
        self.action_dim = action_dim
        self.safe_distance = safe_distance
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    # ============================
    # Public API
    # ============================

    def optimize(self, current_state, initial_action, predicted_trajectory=None):
        """
        Main optimization entry point.
        """

        self._validate_inputs(current_state, initial_action)

        if self.verbose:
            self._log("Starting optimization...")

        bounds = self._build_bounds()
        constraints = self._build_constraints(current_state)

        try:
            result = minimize(
                fun=self._objective,
                x0=initial_action,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options=self._solver_options()
            )

            return self._handle_result(result, initial_action)

        except Exception as e:
            self._log(f"Optimization failed with exception: {e}")
            return self._fallback(initial_action)

    # ============================
    # Objective Function
    # ============================

    def _objective(self, u):
        """
        Fuel minimization objective: minimize ||u||^2
        """
        energy = np.sum(np.square(u))

        if self.verbose:
            self._log(f"Objective evaluated: {energy:.6f}")

        return energy

    # ============================
    # Constraints
    # ============================

    def _build_constraints(self, current_state):
        """
        Builds constraint dictionary list for optimizer.
        """
        return [
            {
                'type': 'ineq',
                'fun': lambda u: self._collision_constraint(u, current_state)
            }
        ]

    def _collision_constraint(self, u, current_state):
        """
        Enforces minimum safe distance constraint.
        """

        predicted = self._predict_state(current_state, u)
        dist = self._compute_distance(predicted)

        constraint_value = dist - self.safe_distance

        if self.verbose:
            self._log(f"Constraint value: {constraint_value:.6f}")

        return constraint_value

    # ============================
    # State Prediction
    # ============================

    def _predict_state(self, state, action):
        """
        Simple dynamics model approximation.
        """

        da = state[0] + action[0] * 0.001
        dlambda = state[1]
        dix = state[4]

        return np.array([da, dlambda, dix])

    def _compute_distance(self, predicted_state):
        """
        Computes Euclidean distance.
        """
        return np.linalg.norm(predicted_state)

    # ============================
    # Bounds and Options
    # ============================

    def _build_bounds(self):
        """
        Build bounds for each action dimension.
        """
        return [(-10.0, 10.0) for _ in range(self.action_dim)]

    def _solver_options(self):
        """
        Solver configuration.
        """
        return {
            'disp': self.verbose,
            'ftol': self.tol,
            'maxiter': self.max_iter
        }

    # ============================
    # Result Handling
    # ============================

    def _handle_result(self, result, fallback_action):
        """
        Processes optimizer output.
        """

        if result.success:
            if self.verbose:
                self._log("Optimization succeeded.")
                self._log(f"Optimal action: {result.x}")
            return result.x

        else:
            self._log("Optimization failed. Using fallback action.")
            return self._fallback(fallback_action)

    def _fallback(self, action):
        """
        Fallback behavior if optimization fails.
        """
        return np.array(action)

    # ============================
    # Input Validation
    # ============================

    def _validate_inputs(self, state, action):
        """
        Ensures inputs are valid.
        """

        if not isinstance(state, (list, np.ndarray)):
            raise ValueError("State must be array-like.")

        if not isinstance(action, (list, np.ndarray)):
            raise ValueError("Action must be array-like.")

        if len(action) != self.action_dim:
            raise ValueError("Action dimension mismatch.")

        if len(state) < 5:
            raise ValueError("State must have at least 5 elements.")

    # ============================
    # Logging Utility
    # ============================

    def _log(self, message):
        """
        Controlled logging.
        """
        if self.verbose:
            print(f"[SCPOptimizer] {message}")

    # ============================
    # Optional Extensions
    # ============================

    def set_safe_distance(self, new_distance):
        """
        Dynamically update safety constraint.
        """
        self.safe_distance = new_distance

    def set_verbose(self, verbose=True):
        """
        Toggle verbosity.
        """
        self.verbose = verbose

    def set_max_iterations(self, max_iter):
        """
        Adjust solver iteration limit.
        """
        self.max_iter = max_iter

    def set_tolerance(self, tol):
        """
        Adjust solver tolerance.
        """
        self.tol = tol

    # ============================
    # Debug Utilities
    # ============================

    def debug_evaluate(self, state, action):
        """
        Evaluate objective and constraint manually.
        """
        obj = self._objective(action)
        constraint = self._collision_constraint(action, state)

        return {
            "objective": obj,
            "constraint": constraint
        }

    def print_summary(self):
        """
        Prints current configuration.
        """
        print("=== SCP Optimizer Summary ===")
        print(f"Action Dimension: {self.action_dim}")
        print(f"Safe Distance: {self.safe_distance}")
        print(f"Max Iterations: {self.max_iter}")
        print(f"Tolerance: {self.tol}")
        print(f"Verbose: {self.verbose}")
        print("=============================")


# ============================
# Example Usage (Test Block)
# ============================

if __name__ == "__main__":
    optimizer = SCPOptimizer(verbose=True)

    current_state = np.array([0.01, 0.0, 0.0, 0.0, 0.01])
    initial_action = np.array([1.0, 0.0, 0.0])

    optimized_action = optimizer.optimize(
        current_state=current_state,
        initial_action=initial_action
    )

    print("Initial Action:", initial_action)
    print("Optimized Action:", optimized_action)

    debug_info = optimizer.debug_evaluate(current_state, optimized_action)
    print("Debug Info:", debug_info)

    optimizer.print_summary()
