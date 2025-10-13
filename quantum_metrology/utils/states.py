# utils/states.py

"""Quantum state generation module.

This module provides functions for generating and manipulating quantum states 
used in quantum metrology simulations, including coherent states and squeezed states.
"""

from typing import Tuple

import numpy as np
import qutip as qt
from qutip import Qobj


def generate_coherent_state(j: float) -> Qobj:
    """Generate a coherent spin state aligned along the x-axis.
    
    Creates a spin coherent state |j,θ,φ⟩ with θ=π/2 and φ=0, 
    which corresponds to alignment along the x-axis.
    
    Args:
        j: Total angular momentum quantum number
        
    Returns:
        Coherent spin state as a Qobj
    """
    theta, phi = np.pi / 2, 0
    return qt.spin_coherent(j, theta, phi)


def apply_squeezing(chi: float, Jy: Qobj, state: Qobj) -> Qobj:
    """Apply squeezing to a quantum state.
    
    Applies a squeezing operation with parameter chi to the given state.
    
    Args:
        chi: Squeezing parameter
        Jy: Angular momentum operator in y-direction
        state: Quantum state to be squeezed
        
    Returns:
        Squeezed quantum state
    """
    squeezing_operator = (-1j * chi * Jy**2).expm()
    return squeezing_operator * state


def get_squeezed_state(N: int, Jy: Qobj, Jz: Qobj, state: Qobj) -> Tuple[float, Qobj]:
    """Find the squeezed state with minimum variance.
    
    Iteratively searches for the optimal squeezing parameter that minimizes
    the variance in a specific direction.
    
    Args:
        N: Number of particles
        Jy: Angular momentum operator in y-direction
        Jz: Angular momentum operator in z-direction
        state: Initial quantum state to be squeezed
        
    Returns:
        Tuple containing:
            - optimal squeezing parameter (chi)
            - optimally squeezed quantum state
    """
    chi_values = np.linspace(0.001, 1.0, 5000)
    best_chi = None
    best_state = None

    for chi in chi_values:
        squeezed_state = apply_squeezing(chi, Jy, state)
        variance = qt.variance(Jy, squeezed_state)

        if variance < N / 4.0:
            best_chi = chi
            best_state = squeezed_state
            break

    return best_chi, best_state