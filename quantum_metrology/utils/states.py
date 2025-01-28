# utils/states.py

import qutip as qt
import numpy as np
from .operators import apply_squeezing_operator

def generate_coherent_state(j):
    """
    Generate a coherent spin state aligned along the x-axis.
    """
    theta, phi = np.pi / 2, 0
    return qt.spin_coherent(j, theta, phi)

def get_squeezed_state(N, Jy, Jz, state):
    """
    Find the squeezed state with minimum variance.
    """
    chi_values = np.linspace(0.001, 1.0, 5000)
    best_chi = None
    best_state = None

    for chi in chi_values:
        squeezed_state = apply_squeezing_operator(chi, Jy) * state
        variance = qt.variance(Jy, squeezed_state)

        if variance < N / 4.0:
            best_chi = chi
            best_state = squeezed_state
            break

    return best_chi, best_state
