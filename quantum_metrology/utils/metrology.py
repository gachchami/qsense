# utils/metrology.py
"""Quantum metrology calculation module.

This module provides functions for calculating quantum Fisher information,
phase variances, and other metrics used in quantum metrology simulations.
"""

import math
from typing import List, Tuple

import numpy as np
import qutip as qt
from qutip import Qobj

from .operators import apply_squeezing_operator, optimal_theta
from .states import generate_coherent_state


def calculate_fisher_information(N: int, sensing_time: float,
                                 is_entangled: bool) -> float:
    """Calculate Fisher Information for a given setup.
    
    Computes the quantum Fisher information, which quantifies the
    maximum amount of information about a parameter that can be extracted.
    
    Args:
        N: Number of particles
        sensing_time: Time spent in the sensing phase
        is_entangled: Whether the particles are entangled (True) or not (False)
        
    Returns:
        Fisher information value
    """
    return N**2 * sensing_time**2 if is_entangled else N * sensing_time**2


def calculate_phase_variances(
        N_max: int, omega: float,
        tau_sense: float) -> Tuple[np.ndarray, np.ndarray, List[Qobj]]:
    """Calculate phase variances for separable and entangled states.
    
    Computes the phase estimation variance for both SQL (Standard Quantum Limit)
    and HL (Heisenberg Limit) strategies for various particle numbers.
    
    Args:
        N_max: Maximum number of particles to consider
        omega: Frequency for sensing
        B: Magnetic field strength
        tau_sense: Sensing time duration
        
    Returns:
        Tuple containing:
            - phase_variance_sql: Array of phase variances for separable states
            - phase_variance_hl: Array of phase variances for entangled states
            - final_states: List of quantum states at different stages of the protocol
    """
    phase_variance_sql = []
    phase_variance_hl = []
    final_states = []

    for N in range(1, N_max + 1):
        j = N / 2

        Jx = qt.jmat(j, 'x')
        Jy = qt.jmat(j, 'y')
        Jz = qt.jmat(j, 'z')

        if N == N_max:
            print(Jz)

        optimal_theta_j = optimal_theta(N)
        mu = optimal_theta_j * 2
        A = 1 - (np.cos(mu)**(2 * j - 2))
        B_value = 4 * np.sin(mu / 2) * (np.cos(mu / 2)**(2 * j - 2))

        # Handle potential division by zero
        if abs(A) < 1e-10:
            delta = 0 if B_value == 0 else np.sign(B_value) * np.pi / 4
        else:
            delta = 0.5 * np.arctan(B_value / A)

        v = -1 * delta

        tau_prep = 1.0
        tau_meas = 1.0

        if N == N_max:
            print(
                f"tau_prep: {tau_prep}, tau_meas: {tau_meas}, tau_sense: {tau_sense}"
            )

        # Generate initial coherent state
        coherent_state = generate_coherent_state(j)
        if N == N_max:
            final_states.append(coherent_state)

        # Separable strategy
        H_sense = omega * Jz
        evolved_coherent_state = (-1j * H_sense *
                                  tau_sense).expm() * coherent_state

        phase_var_sql = qt.variance(Jy, evolved_coherent_state)
        exp_Jy_sql = qt.expect(Jy, evolved_coherent_state)

        phase_variance_sql.append(phase_var_sql)

        if N == N_max:
            final_states.append(evolved_coherent_state)

        # Entangled strategy - squeezing and rotation
        squeezing_operator = apply_squeezing_operator(optimal_theta_j / 2, Jy)
        squeezed_state = squeezing_operator * coherent_state

        if N == N_max:
            final_states.append(squeezed_state)

        rotate_operator = (-1j * v * Jx).expm()
        squeezed_state = rotate_operator * squeezed_state

        if N == N_max:
            final_states.append(squeezed_state)

        # Evolution under sensing Hamiltonian for HL
        evolved_squeezed_state = (-1j * H_sense *
                                  tau_sense).expm() * squeezed_state

        phase_var_hl = qt.variance(Jy, evolved_squeezed_state)
        exp_Jy_hl = qt.expect(Jy, evolved_squeezed_state)

        print(f"N: {N}, hl_exp: {exp_Jy_hl}, sql_exp: {exp_Jy_sql}")
        phase_variance_hl.append(phase_var_hl)

        if N == N_max:
            final_states.append(evolved_squeezed_state)

    return (np.array(phase_variance_sql), np.array(phase_variance_hl),
            final_states)
