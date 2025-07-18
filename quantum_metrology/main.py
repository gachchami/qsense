# main.py

"""Main script for quantum metrology simulation.

This script runs the quantum metrology simulation, calculating phase variances
and visualizing the results including Husimi Q-functions.
"""

import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
from qutip import Qobj

from config import N_MAX, TAU_SENSE, OMEGA
from utils.metrology import calculate_phase_variances
from utils.plotting import plot_results
from utils.husimi import compute_husimi_q, plot_husimi_q


def process_final_states(
    final_states: List[Qobj],
    j: float,
    max_states: Optional[int] = None
) -> None:
    """Process and visualize the final quantum states.

    Computes and plots the Husimi Q-function for the given quantum states.

    Args:
        final_states: List of quantum states to visualize
        j: Total angular momentum quantum number
        max_states: Maximum number of states to visualize (None for all)
    """
    if max_states is not None:
        states_to_process = final_states[:max_states]
    else:
        states_to_process = final_states

    state_descriptions = [
        "Initial Coherent State",
        "Evolved Coherent State (SQL)",
        "Squeezed State",
        "Rotated Squeezed State",
        "Evolved Squeezed State (HL)"
    ]

    for i, state in enumerate(states_to_process):
        print(f"Processing state {i+1}/{len(states_to_process)}...")

        # Calculate Husimi Q-function (with reduced resolution for faster computation)
        Q_vals, theta_grid, phi_grid = compute_husimi_q(
            j,
            state,
            n_theta=150,  # Reduced resolution for faster computation
            n_phi=300
        )

        # Get state name if available, otherwise use generic name
        state_name = state_descriptions[i] if i < len(
            state_descriptions) else f"State {i+1}"

        # Plot the Husimi Q-function
        plot_husimi_q(Q_vals, theta_grid, phi_grid, state_name=state_name)


def main() -> None:
    """Main function to run the quantum metrology simulation."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Quantum Metrology Simulation')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--max-states', type=int, default=5,
                        help='Maximum number of states to visualize')
    args = parser.parse_args()

    # Display simulation parameters
    print("Running quantum metrology simulation with parameters:")
    print(f"N_MAX = {N_MAX}")
    print(f"TAU_SENSE = {TAU_SENSE}")
    print("-" * 50)

    # Start timing
    start_time = time.time()

    # Calculate phase variances and metrological gain
    print("Calculating phase variances...")
    phase_var_sql, phase_var_hl, final_states = calculate_phase_variances(
        N_MAX, OMEGA, TAU_SENSE
    )

    calc_time = time.time() - start_time
    print(f"Calculation completed in {calc_time:.2f} seconds")

    # Plot results if not disabled
    if not args.no_plots:
        print("Generating phase variance and magnetic field plots...")
        plot_results(N_MAX, phase_var_sql, phase_var_hl, TAU_SENSE)

        # Process final states for visualization
        print("Generating Husimi Q-function visualizations...")
        j = N_MAX / 2
        process_final_states(final_states, j, args.max_states)

    total_time = time.time() - start_time
    print(f"Total simulation time: {total_time:.2f} seconds")
    print("Simulation completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
