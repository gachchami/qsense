from sensors.abstract_sensor import QuantumSensorBase
from utils.sensor_utils import (
    generate_squeezed_light,
    simulate_faraday_rotation,
    calculate_magnetometer_sensitivity,
    apply_photon_loss
)


class QuantumMagnetometer(QuantumSensorBase):
    """
    Quantum Magnetometer implementation using QuTiP.
    """

    def __init__(self, num_photons, squeezing_factor, magnetic_field, interaction_time, noise_level, efficiency):
        self.num_photons = num_photons
        self.squeezing_factor = squeezing_factor
        self.magnetic_field = magnetic_field
        self.interaction_time = interaction_time
        self.noise_level = noise_level
        self.efficiency = efficiency
        self.sensitivity = None

    def prepare(self):
        """
        Generate squeezed light and calculate baseline sensitivity.
        """
        print("DEBUG: Preparing squeezed light...")
        self.state = generate_squeezed_light(self.num_photons, self.squeezing_factor)
        self.state = apply_photon_loss(self.state, self.efficiency)  # Apply photon loss
        print(f"DEBUG: Squeezed State After Loss Dimensions: {self.state.dims}")

        self.sensitivity = calculate_magnetometer_sensitivity(
            self.noise_level, self.squeezing_factor, self.num_photons
        )
        print(f"DEBUG: Baseline Sensitivity: {self.sensitivity} pT/√Hz")

    def sense(self):
        """
        Simulate Faraday rotation due to magnetic field interaction.
        """
        print("DEBUG: Simulating Faraday rotation...")
        self.polarization_angle = simulate_faraday_rotation(
            0.0, self.magnetic_field, self.interaction_time
        )
        print(f"DEBUG: Polarization Angle After Rotation: {self.polarization_angle}")

    def measure(self):
        """
        Return the sensitivity and polarization angle.
        """
        print(f"DEBUG: Final Sensitivity: {self.sensitivity}")
        return self.polarization_angle, self.sensitivity
