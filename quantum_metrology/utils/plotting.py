# utils/plotting.py

import matplotlib.pyplot as plt

def plot_results(N_max, phase_variance_sql, phase_variance_hl, B_sql, B_hl, omega, tau_sense):
    """
    Plot phase variances and metrological gain.
    """
    x = range(1, N_max + 1)

    # Phase variance plot
    plt.figure(figsize=(12, 6))
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
    factor = omega * tau_sense
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, B_sql/factor, label="SQL", linestyle="--", marker="o")
    plt.plot(x, B_hl/factor, label="HL", linestyle="-", marker="x")
    plt.title("B")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Magnetic Field")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
