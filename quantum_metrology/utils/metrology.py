# utils/metrology.py
"""Quantum metrology calculation module.

This module provides functions for calculating_phase noise
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import qutip as qt

from .operators import (
    _calculate_correction_angle,
    get_squeezing_operator,
    optimal_theta,
)
from .states import generate_coherent_state


class QState(qt.Qobj):
    """
    Custom QState class that implements the qt.Qobj but attaches the variance value to the object
    """

    def __init__(self, qobj: qt.Qobj, variance: float = 0.0):
        # initialize the parent Qobj with data + dims
        super().__init__(qobj.full(), dims=qobj.dims)
        self.variance = variance


@dataclass
class Experiment:
    """
    Data class to record all the experiment specific data
    """
    N: int = 0
    theta: float = 0.0
    mu: float = 0.0
    v: float = 0.0
    initial_axis: Optional[Union[qt.Qobj, Tuple[qt.Qobj, ...]]] = None
    measurement_axis: Optional[Union[qt.Qobj, Tuple[qt.Qobj, ...]]] = None
    coherent_state: Optional[QState] = None
    evolved_coherent_state: Optional[QState] = None
    squeezed_state: Optional[QState] = None
    evolved_squeezed_state: Optional[QState] = None


def calculate_phase_noise(N_max: int) -> List[Experiment]:
    """Calculate phase variances for separable and entangled states.
    
    Computes the phase variance/noise for 1..N_Max number of spin particles.
    
    Args:
        N_max: Maximum number of particles
        
    Returns:
        List of Experiment: Array of Experiment containing parameters of the experiment and its 
        results.
    """
    experiments = []

    for N in range(1, N_max + 1):

        experiment = Experiment()
        experiment.N = N

        j = experiment.N / 2
        experiment.mu = optimal_theta(j)

        Jx = qt.jmat(j, 'x')
        Jy = qt.jmat(j, 'y')
        Jz = qt.jmat(j, 'z')

        experiment.initial_axis = Jx
        experiment.measurement_axis = Jz

        experiment.v = _calculate_correction_angle(j, experiment.mu)

        # Generate initial coherent state
        experiment.coherent_state = QState(generate_coherent_state(j))
        experiment.coherent_state.variance = qt.variance(
            experiment.measurement_axis, experiment.coherent_state)

        H_sense = -Jy
        experiment.evolved_coherent_state = (
            -1j * H_sense).expm() * experiment.coherent_state

        experiment.evolved_coherent_state.variance = qt.variance(
            experiment.measurement_axis, experiment.evolved_coherent_state)

        squeezing_operator = get_squeezing_operator(experiment.mu / 2,
                                                    experiment.measurement_axis)
        squeezed_state = QState(squeezing_operator * experiment.coherent_state)

        rotate_operator = (-1j * experiment.v * experiment.initial_axis).expm()

        experiment.squeezed_state = QState(rotate_operator * squeezed_state)
        experiment.squeezed_state.variance = qt.variance(
            experiment.measurement_axis, experiment.squeezed_state)

        experiment.evolved_squeezed_state = QState(
            (-1j * H_sense).expm() * experiment.squeezed_state)

        experiment.evolved_squeezed_state.variance = qt.variance(
            experiment.measurement_axis, experiment.evolved_squeezed_state)

        experiments.append(experiment)
    return experiments
