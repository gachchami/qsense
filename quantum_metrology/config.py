# config.py
"""
Configuration parameters.

This module defines the key parameters used throughout the quantum metrology
simulation, including time scales, particle numbers and physical constants.

"""
from typing import Final

# Total fixed time for the experiment
T_TOTAL: Final[float] = 1.0

# Time allocation fractions
TAU_PREP_FRACTION: Final[float] = 0.1  # Preparation time fraction
TAU_MEAS_FRACTION: Final[float] = 0.05  # Measurement time fraction
TAU_SENSE_FRACTION: Final[float] = 1 - TAU_PREP_FRACTION - TAU_MEAS_FRACTION

# Actual sensing time
TAU_SENSE: Final[float] = 0.5

# System parameters
N_MAX: Final[int] = 400  # Maximum number of particles
OMEGA: Final[float] = 1.0  # Frequency for sensing
