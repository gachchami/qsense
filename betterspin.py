import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from math import factorial

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Define collective spin operators for N spin-1/2 particles
def J_x(N):
    terms = [
        np.kron(np.eye(2**i), np.kron(sigma_x, np.eye(2**(N-i-1))))
        for i in range(N)
    ]
    return 0.5 * np.sum(terms, axis=0)

def J_y(N):
    terms = [
        np.kron(np.eye(2**i), np.kron(sigma_y, np.eye(2**(N-i-1))))
        for i in range(N)
    ]
    return 0.5 * np.sum(terms, axis=0)

def J_z(N):
    terms = [
        np.kron(np.eye(2**i), np.kron(sigma_z, np.eye(2**(N-i-1))))
        for i in range(N)
    ]
    return 0.5 * np.sum(terms, axis=0)

# Define the one-axis twisting Hamiltonian for spin squeezing
def one_axis_twisting(N, chi, t):
    Jz = J_z(N)
    return expm(-1j * chi * t * Jz @ Jz)

# Define the initial coherent spin state (CSS)
def coherent_spin_state(N, theta, phi):
    state = np.zeros(2**N, dtype=complex)
    state[0] = 1  # All spins down
    R = expm(-1j * theta * (np.cos(phi) * J_x(N) + np.sin(phi) * J_y(N)))
    return R @ state

# Define the phase estimation sensitivity
def phase_sensitivity(state, J_operator, phase):
    """
    Calculate the phase sensitivity for a given state and operator.
    Handles cases where expectation values are very small.
    """
    U = expm(-1j * phase * J_operator)
    evolved_state = U @ state
    expectation = np.vdot(evolved_state, J_operator @ evolved_state)
    variance = np.vdot(evolved_state, J_operator @ J_operator @ evolved_state) - expectation**2

    # Debugging prints
    print(f"State norm: {np.linalg.norm(state):.4f}, Expectation: {expectation}, Variance: {variance}")

    # Handle small expectation values
    epsilon = 1e-6
    if np.abs(expectation) < epsilon:
        print("Warning: Expectation value too small, using variance only.")
        return np.sqrt(np.abs(variance))  # Use only the variance if expectation is too small

    # Calculate and return sensitivity
    return np.sqrt(np.abs(variance)) / np.abs(expectation)

# Parameters
chi = 0.1  # Twisting strength
t = 1.0  # Evolution time
phase = np.pi / 4  # Fixed phase for sensitivity calculation

# Number of particles to test (N < 4 for debugging)
N_values = np.arange(1, 16)

# Arrays to store phase sensitivities
spin_sensitivities = []
unsqueezed_sensitivities = []

# Loop over number of particles
for N in N_values:
    print(f"\n=== N = {N} ===")

    # Initial state
    css = coherent_spin_state(N, np.pi / 2, 0)
    css /= np.linalg.norm(css)  # Normalize the state
    print(f"Initial coherent spin state (CSS):\n{css}")

    # One-axis twisting for spin squeezing
    squeezed_spin_state = one_axis_twisting(N, chi, t) @ css
    squeezed_spin_state /= np.linalg.norm(squeezed_spin_state)
    print(f"Squeezed spin state:\n{squeezed_spin_state}")

    # Calculate phase sensitivities
    spin_sensitivity = phase_sensitivity(squeezed_spin_state, J_x(N), phase)
    unsqueezed_sensitivity = phase_sensitivity(css, J_x(N), phase)

    spin_sensitivities.append(spin_sensitivity)
    unsqueezed_sensitivities.append(unsqueezed_sensitivity)

    # Print final results for the current N
    print(f"Sensitivity - Spin Squeezing: {spin_sensitivity:.4f}, Unsqueezed: {unsqueezed_sensitivity:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(N_values, spin_sensitivities, label='Spin Squeezing', marker='o')
plt.plot(N_values, unsqueezed_sensitivities, label='Unsqueezed', linestyle='--', marker='^')
plt.xlabel('Number of Particles (N)')
plt.ylabel('Phase Sensitivity')
plt.title('Phase Sensitivity vs Number of Particles')
plt.legend()
plt.grid()
plt.show()
