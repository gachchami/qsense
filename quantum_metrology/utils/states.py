# utils/states.py
"""Quantum state generation module.

This module provides functions for generating and manipulating quantum states 
used in quantum metrology simulations, including coherent states and squeezed states.
"""

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
