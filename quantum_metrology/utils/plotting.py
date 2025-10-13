# utils/plotting.py
"""Data visualization module for quantum metrology results.

This module provides functions for visualizing the results of quantum
metrology simulations.
"""

import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from .metrology import Experiment, QState


def mean_component(j: float, state, axis: str) -> float:
    """Compute <J_axis> expectation value."""
    return float(qt.expect(axis, state))


def mean(j: int, state: QState) -> float:
    """Compute |<J>| = sqrt(<Jx>^2 + <Jy>^2 + <Jz>^2)."""
    Jx = qt.jmat(j, 'x')
    Jy = qt.jmat(j, 'y')
    Jz = qt.jmat(j, 'z')

    mx = qt.expect(Jx, state)
    my = qt.expect(Jy, state)
    mz = qt.expect(Jz, state)
    return np.linalg.norm(np.array([mx, my, mz], dtype=float))


def plot_noise(N_max: int, experiments: List[Experiment]) -> None:
    """Plot phase noise.
    
    Creates visualizations of the phase similar to what Kitagawa and Ueda demonstrated.
    x axis = [2,5,10,50,100]
    
    Args:
        N_max: Maximum number of particles considered
        experiments: The list of experiments
    """
    n = np.arange(1, N_max + 1)
    j = n / 2
    phase_noise_sql = []
    phase_noise_hl = []

    # Create figure for both plots
    plt.figure(figsize=(12, 6))
    for experiment in experiments:
        phase_noise_sql.append(experiment.evolved_coherent_state.variance)
        phase_noise_hl.append(experiment.evolved_squeezed_state.variance)

    # Phase noise plot
    plt.subplot(1, 2, 1)

    plt.plot(j, j / 2, label="J/2", linestyle="-", linewidth=2)
    plt.plot(j, phase_noise_sql, label="Coherent state", color="red",
             marker="x", markersize="4", linewidth=0)
    plt.plot(j, phase_noise_hl, label="OAT", linestyle="--", marker="x",
             markersize="4")
    plt.plot(j, (1 / 2) * ((j / 3)**(1 / 3)), label="1/2(J/3)^1/3",
             linestyle="-", linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Noise reduction with OAT spin squeezing")
    plt.xticks([1, 2, 5, 10, 20, 50, 100], labels=[1, 2, 5, 10, 20, 50, 100])
    plt.yticks([0.5, 0.55, 2, 5, 10, 20, 50],
               labels=[0.5, 0.55, 2, 5, 10, 20, 50])
    plt.xlabel("J")
    plt.ylabel("Noise")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_sensitivity(N_max: int, experiments: List[Experiment]) -> None:
    """Plot phase sensitivity.
    
    Creates visualizations of sensitivity.
    
    Args:
        N_max: Maximum number of particles considered
        List[Experiment]: List of all the experiement objects
    """
    n = np.arange(1, N_max + 1)
    j = n / 2
    phase_sensitivity_sql = []
    phase_sensitivity_hl = []

    plt.figure(figsize=(12, 6))

    for experiment in experiments:

        evolved_css_sentivity = math.sqrt(
            experiment.evolved_coherent_state.variance) / mean(
                experiment.N / 2, experiment.evolved_coherent_state)

        phase_sensitivity_sql.append(evolved_css_sentivity)

        evolved_sss_sensitivity = math.sqrt(
            experiment.evolved_squeezed_state.variance) / mean(
                experiment.N / 2, experiment.evolved_squeezed_state)
        phase_sensitivity_hl.append(evolved_sss_sensitivity)
    plt.subplot(1, 2, 1)
    plt.plot(j / 2, phase_sensitivity_sql, label="CSS", linestyle="--",
             marker="x", markersize="1")
    plt.plot(j / 2, phase_sensitivity_hl, label="SSS", linestyle="-",
             marker="x")
    plt.yscale('log')
    plt.title("Sensitivity ")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Phase Sensitivity")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
