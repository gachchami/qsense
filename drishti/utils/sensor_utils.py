from qutip import squeeze, basis, tensor, destroy
import numpy as np


def generate_squeezed_light(num_photons, squeezing_factor):
    """
    Generate squeezed vacuum state for magnetometry.

    Args:
        num_photons (int): Number of Fock states to include in the Hilbert space.
        squeezing_factor (float): Strength of squeezing.

    Returns:
        Qobj: Squeezed vacuum state.
    """
    print("DEBUG: Generating squeezed vacuum state...")
    vacuum = basis(num_photons, 0)  # Vacuum state
    squeezed_state = squeeze(num_photons, squeezing_factor) * vacuum
    print(f"DEBUG: Squeezed State Dimensions: {squeezed_state.dims}")
    return squeezed_state


def apply_photon_loss(state, efficiency):
    """
    Apply photon loss to the quantum state using a beam splitter model.

    Args:
        state (Qobj): Quantum state of the system.
        efficiency (float): Detection efficiency (0 < efficiency <= 1).

    Returns:
        Qobj: Quantum state after applying photon loss.
    """
    print("DEBUG: Applying photon loss...")
    loss_op = np.sqrt(efficiency) * destroy(state.dims[0][0])  # Loss operator
    loss_state = loss_op * state * loss_op.dag()
    return loss_state.unit()


def simulate_faraday_rotation(polarization_angle, magnetic_field, interaction_time):
    """
    Simulate Faraday rotation.
    """
    print("DEBUG: Simulating Faraday rotation...")
    return polarization_angle + magnetic_field * interaction_time


def calculate_magnetometer_sensitivity(noise_level, squeezing_factor, photon_count):
    """
    Calculate the sensitivity of a magnetometer in pT/√Hz.
    """
    print("DEBUG: Calculating magnetometer sensitivity...")
    shot_noise = noise_level / np.sqrt(photon_count)
    squeezing_gain = np.exp(-squeezing_factor)  # Reduced variance due to squeezing
    return shot_noise * squeezing_gain
