from sensors.magnetometry_sensor_jax import QuantumMagnetometer


def run_magnetometer():
    dim = 20  # Truncated Hilbert space dimension
    squeezing_factor = 0.37  # Squeezing factor for 3.2 dB improvement
    magnetic_field = 0.1  # Magnetic field in Tesla
    interaction_time = 1.0  # Interaction time
    efficiency = 0.7  # 70% detection efficiency

    print("DEBUG: Initializing Quantum Magnetometer with JAX...")
    sensor = QuantumMagnetometer(dim, squeezing_factor, magnetic_field, interaction_time, efficiency)

    print("DEBUG: Starting the sensor workflow...")
    sensor.prepare()
    sensor.sense()
    sensitivity = sensor.measure()

    print(f"Final Sensitivity: {sensitivity}")


if __name__ == "__main__":
    run_magnetometer()
