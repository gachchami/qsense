# config.py
T_TOTAL = 1.0  # Total fixed time for the experiment
TAU_PREP_FRACTION = 0.1  # Preparation time fraction
TAU_MEAS_FRACTION = 0.05  # Measurement time fraction
TAU_SENSE_FRACTION = 1 - TAU_PREP_FRACTION - TAU_MEAS_FRACTION
N_MAX = 20  # Maximum number of particles
OMEGA = 3.0  # Frequency for sensing
