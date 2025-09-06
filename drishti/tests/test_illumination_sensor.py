import pytest
from sensors.illumination_sensor import QuantumIlluminationSensor
from jax import random


def test_illumination_sensor():
    num_photons = 1000
    reflectivity = 0.1
    noise_level = 0.05
    key = random.PRNGKey(42)

    sensor = QuantumIlluminationSensor(num_photons, reflectivity, noise_level, key)
    correlation = sensor.run()

    assert correlation > 0, "Correlation should be positive."
