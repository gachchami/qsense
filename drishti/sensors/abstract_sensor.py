from abc import ABC, abstractmethod

class QuantumSensorBase(ABC):
    """
    Abstract Base Class for Quantum Sensors.
    """

    @abstractmethod
    def prepare(self):
        """Prepare the quantum system."""
        pass

    @abstractmethod
    def sense(self):
        """Simulate the interaction with the target."""
        pass

    @abstractmethod
    def measure(self):
        """Extract information from the quantum system."""
        pass

    def run(self):
        """Execute the sensor workflow."""
        self.prepare()
        self.sense()
        return self.measure()
