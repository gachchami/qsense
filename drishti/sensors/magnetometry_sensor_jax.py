from utils.sensor_utils_jax import (
    generate_squeezed_light_jax,
    apply_loss_jax,
    simulate_faraday_rotation_jax,
)
import jax.numpy as jnp


class QuantumMagnetometer:
    """
    Quantum Magnetometer using JAX for numerical operations.
    """

    def __init__(self, dim, squeezing_factor, magnetic_field, interaction_time, efficiency):
        self.dim = dim
        self.squeezing_factor = squeezing_factor
        self.magnetic_field = magnetic_field
        self.interaction_time = interaction_time
        self.efficiency = efficiency
        self.state = None
        self.sensitivity = None

    def prepare(self):
        """
        Generate squeezed light and apply photon loss using JAX.
        """
        print("DEBUG: Preparing squeezed light...")
        self.state = generate_squeezed_light_jax(self.dim, self.squeezing_factor)
        self.state = apply_loss_jax(self.state, self.efficiency)
        print(f"DEBUG: Prepared State Dimensions: {self.state.shape}")

    def sense(self):
        """
        Simulate Faraday rotation using JAX.
        """
        print("DEBUG: Simulating Faraday rotation...")
        self.state = simulate_faraday_rotation_jax(self.state, self.magnetic_field, self.interaction_time)
        print("DEBUG: Rotated State:", self.state)

    def measure(self):
        """
        Calculate the sensitivity of the magnetometer.
        """
        print("DEBUG: Measuring sensitivity...")
        self.sensitivity = jnp.linalg.norm(self.state)
        print(f"DEBUG: Sensitivity: {self.sensitivity}")
        return self.sensitivity
