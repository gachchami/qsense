import pytest
from sensors.magnetometry_sensor import QuantumMagnetometer
from jax import random


def test_magnetometry_sensor():
    num_photons = int(1e6)
    squeezing_factor = 0.5
    magnetic_field = 0.1
    interaction_time = 1.0
    noise_level = 0.01
    key = random.PRNGKey(42)

    sensor = QuantumMagnetometer(num_photons, squeezing_factor, magnetic_field, interaction_time, noise_level, key)
    polarization_angle, sensitivity = sensor.run()

    assert polarization_angle > 0, "Polarization angle should be positive."
    assert sensitivity > 0, "Sensitivity should be positive."
