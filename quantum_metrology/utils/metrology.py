# utils/metrology.py

import numpy as np
import qutip as qt
from .states import generate_coherent_state, get_squeezed_state
from .operators import apply_squeezing_operator, optimal_theta

def calculate_fisher_information(N, sensing_time, is_entangled):
    """
    Calculate Fisher Information
    """
    return N**2 * sensing_time**2 if is_entangled else N * sensing_time**2

def calculate_phase_variances(N_max, omega, B, tau_sense):
    """
    Calculate phase variances for separable and entangled states.
    """
    phase_variance_sql = []
    phase_variance_hl = []
    phase_measured_sql = []
    phase_measured_hl = []
    final_states = []
    for N in range(1, N_max + 1):
        j = N / 2

        Jx = qt.jmat(j, 'x')
        Jy = qt.jmat(j, 'y')
        Jz = qt.jmat(j, 'z')
    
        optimal_theta_j = optimal_theta(N)
        mu = optimal_theta_j * 2
        A = 1 - (np.cos(mu) ** (2 * j - 2))
        B = 4 * np.sin(mu / 2) * (np.cos( mu / 2) ** (2 * j - 2))
        delta = 0.5 * np.arctan(B / A)
        v = -1 * delta
      
      #   tau_prep = tau_prep_fraction * T_total
      #   tau_meas = tau_meas_fraction * T_total
      #   tau_sense = T_total - tau_prep - tau_meas
        tau_prep = 1.0
        tau_meas = 1.0
        if N == N_max:
           print(tau_prep, tau_meas,  tau_sense)
        coherent_state = generate_coherent_state(j)
        if N == N_max:
           final_states.append(coherent_state)
        # Separable strategy
        H_sense = omega * B * Jz
        evolved_coherent_state = (-1j * H_sense * tau_sense).expm() * coherent_state
        phase_var_sql = qt.variance(Jy, evolved_coherent_state)
        phase_meas_sql = qt.expect(Jy, evolved_coherent_state)
        phase_variance_sql.append(phase_var_sql)
        phase_measured_sql.append(phase_meas_sql)
        if N == N_max:
           final_states.append(evolved_coherent_state)

        # Entangled strategy
        squeezing_operator = apply_squeezing_operator(optimal_theta_j/2, Jy)
        squeezed_state = squeezing_operator * coherent_state
        if N == N_max:
           final_states.append(squeezed_state)
        rotate_operator = (-1j * v * Jx).expm()
        squeezed_state = rotate_operator * squeezed_state
        if N == N_max:
           final_states.append(squeezed_state)
        evolved_squeezed_state = (-1j * H_sense * tau_sense).expm() * squeezed_state
        phase_var_hl = qt.variance(Jy, evolved_squeezed_state)
        phase_meas_hl = qt.expect(Jy, evolved_squeezed_state)
        phase_variance_hl.append(phase_var_hl)
        phase_measured_hl.append(phase_meas_hl)

        if N == N_max:
           final_states.append(evolved_squeezed_state)
        
    return np.array(phase_variance_sql), np.array(phase_variance_hl), np.array(phase_measured_sql), np.array(phase_measured_hl), final_states
