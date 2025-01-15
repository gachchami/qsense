from sensors.abstract_sensor import QuantumSensorBase
from utils.sensor_utils import (
    create_entangled_state,
    apply_reflectivity,
    add_noise,
    measure_correlation,
)


class QuantumIlluminationSensor(QuantumSensorBase):
    """
    Quantum Illumination Sensor implementation using QuTiP.
    """

    def __init__(self, reflectivity, noise_level):
        self.reflectivity = reflectivity
        self.noise_level = noise_level
        self.state = None
        self.correlation = None

    def prepare(self):
        """
        Create an entangled state of signal and idler photons.
        """
        print("DEBUG: Preparing the entangled state...")
        self.state = create_entangled_state()
        print(f"DEBUG: Entangled state: {self.state}")
        print(f"DEBUG: Prepared State Dimensions: {self.state.dims}")

    def sense(self):
        """
        Simulate interaction with the target and noise addition.
        """
        print("DEBUG: Simulating target interaction and adding noise...")
        self.state = apply_reflectivity(self.state, self.reflectivity)
        self.state = add_noise(self.state, self.noise_level)

    def measure(self):
        """
        Measure the correlation between signal and idler photons.
        """
        print("DEBUG: Measuring correlation...")
        self.correlation = measure_correlation(self.state)
        print(f"DEBUG: Measured Correlation: {self.correlation}")
        return self.correlation
