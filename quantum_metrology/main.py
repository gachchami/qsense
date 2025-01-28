# main.py

from config import T_TOTAL, TAU_PREP_FRACTION, TAU_MEAS_FRACTION, N_MAX, OMEGA
from utils.metrology import calculate_phase_variances
from utils.plotting import plot_results
from utils.husimi import compute_husimi_q, plot_husimi_q

def main():
    # Calculate phase variances and metrological gain
    phase_var_sql, phase_var_hl, final_states = calculate_phase_variances(
        N_MAX, OMEGA, TAU_PREP_FRACTION, TAU_MEAS_FRACTION, T_TOTAL
    )
    plot_results(N_MAX, phase_var_sql, phase_var_hl)
    j = N_MAX/2
    for state in final_states:
      Q_vals, theta_grid, phi_grid = compute_husimi_q(j, state)
      plot_husimi_q(Q_vals, theta_grid, phi_grid, state_name="State")

if __name__ == "__main__":
    main()
