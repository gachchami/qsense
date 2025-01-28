# utils/operators.py

import numpy as np
import qutip as qt

def J_n(theta, Jy, Jz):
    """
    Calculate the operator J_n(theta).
    """
    return Jy * np.cos(theta) + Jz * np.sin(theta)

def apply_squeezing_operator(squeezing_strength, Jy):
    """
    Apply the squeezing operator to the quantum state.
    """
    return (-1j * squeezing_strength * Jy**2).expm()

def optimal_theta(n):
    return (3**(1/6))*((n)**(-2/3))/2
