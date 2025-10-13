# config.py
"""
Configuration parameters.

This module defines the key parameters used throughout the quantum metrology
simulation, including time scales, particle numbers and physical constants.

"""
from typing import Final

# Actual sensing time
TAU_SENSE: Final[float] = 0.5

# System parameters
N_MAX: Final[int] = 200  # Maximum number of particles
OMEGA: Final[float] = 1.0  # Frequency for sensing
