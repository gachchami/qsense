# utils/operators.py

"""Quantum operators module for metrology simulations.

This module provides functions for generating and manipulating quantum operators
used in metrology simulations, such as angular momentum operators and
squeezing operators.
"""

from typing import Callable, Tuple

import numpy as np
import qutip as qt
from qutip import Qobj


def J_n(theta: float, Jy: Qobj, Jz: Qobj) -> Qobj:
    """Calculate the operator J_n(theta).
    
    Computes a rotated angular momentum operator along direction specified by theta.
    
    Args:
        theta: Rotation angle in radians
        Jy: Angular momentum operator in y-direction
        Jz: Angular momentum operator in z-direction
        
    Returns:
        Rotated angular momentum operator
    """
    return Jy * np.cos(theta) + Jz * np.sin(theta)


def apply_squeezing_operator(squeezing_strength: float, Jy: Qobj) -> Qobj:
    """Apply the squeezing operator to the quantum state.
    
    Creates the squeezing operator with given strength and returns the operator.
    
    Args:
        squeezing_strength: Parameter controlling the amount of squeezing
        Jy: Angular momentum operator in y-direction
        
    Returns:
        Squeezing operator as a Qobj
    """
    return (-1j * squeezing_strength * Jy**2).expm()


def optimal_theta(n: int) -> float:
    """Calculate the optimal squeezing angle for n particles.
    
    Computes the optimal squeezing angle for a given number of particles
    to achieve maximum phase sensitivity.
    
    Args:
        n: Number of particles
        
    Returns:
        Optimal squeezing angle in radians
    """
    return (3**(1/6)) * ((n)**(-2/3)) / 2