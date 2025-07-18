# utils/plotting.py
"""Data visualization module for quantum metrology results.

This module provides functions for visualizing the results of quantum
metrology simulations, including phase variances and metrological gain.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    N_max: int,
    phase_variance_sql: np.ndarray,
    phase_variance_hl: np.ndarray,
) -> None:
    """Plot phase variances and metrological gain.
    
    Creates visualizations of the phase variance and magnetic field
    estimation results for both SQL and HL strategies.
    
    Args:
        N_max: Maximum number of particles considered
        phase_variance_sql: Array of phase variances for separable states
        phase_variance_hl: Array of phase variances for entangled states
    """
    x = range(1, N_max + 1)

    # Create figure for both plots
    plt.figure(figsize=(12, 6))

    # Phase variance plot
    plt.subplot(1, 2, 1)
    plt.plot(x, phase_variance_sql, label="SQL", linestyle="--", marker="o")
    plt.plot(x, phase_variance_hl, label="HL", linestyle="-", marker="x")
    plt.title("Phase Variance")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Phase Variance")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # # Estimated magnetic field plot
    # factor = omega * tau_sense
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(x, B_sql/factor, label="SQL", linestyle="--", marker="o")
    # plt.plot(x, B_hl/factor, label="HL", linestyle="-", marker="x")
    # plt.title("Estimated Magnetic Field")
    # plt.xlabel("Number of Particles (N)")
    # plt.ylabel("Magnetic Field")
    # plt.legend()
    # plt.grid()

    # plt.tight_layout()
    # plt.show()
