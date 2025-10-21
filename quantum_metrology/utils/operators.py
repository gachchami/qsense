# utils/operators.py
"""Quantum operators module for metrology simulations.

This module provides functions for generating and manipulating quantum operators
used in metrology simulations
"""

import numpy as np
from qutip import Qobj


def _calculate_correction_angle(j: int, mu: float) -> float:
    A = 1 - (np.cos(mu)**(2 * j - 2))
    B = 4 * np.sin(mu / 2) * (np.cos(mu / 2)**(2 * j - 2))

    # Handle potential division by zero
    if abs(A) < 1e-10:
        delta = 0 if B == 0 else np.sign(B) * np.pi / 4
    else:
        delta = 0.5 * np.arctan(B / A)

    return -1 * delta


def get_squeezing_operator(squeezing_strength: float, J: Qobj) -> Qobj:
    """Apply the squeezing operator to the quantum state.
    
    Creates the squeezing operator with given strength and returns the operator as QObj.
    
    Args:
        squeezing_strength: Parameter controlling the amount of squeezing
        J: Angular momentum operator
        
    Returns:
        Squeezing operator as a Qobj
    """
    return (-1j * squeezing_strength * J**2).expm()


def optimal_theta(n: int) -> float:
    """Calculate the optimal squeezing angle for n particles.
    
    Computes the optimal squeezing angle for a given number of particles
    to achieve maximum phase sensitivity.
    
    Args:
        n: Number of particles
        
    Returns:
        Optimal squeezing angle in radians
    """
    return (24**(1 / 6)) * ((n)**(-2 / 3))
