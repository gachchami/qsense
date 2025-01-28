# utils/husimi.py

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

def compute_husimi_q(J, state, n_theta=300, n_phi=600):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    Q_vals = np.zeros_like(theta_grid, dtype=np.float64)

    # Loop through the grid to compute the Husimi Q function
    for i in range(n_phi):  # Loop over n_phi for the correct axis (axis=1 in grid)
        for j in range(n_theta):  # Loop over n_theta for the correct axis (axis=0 in grid)
            test_state = qt.spin_coherent(J, theta_grid[i, j], phi_grid[i, j])
            Q_vals[i, j] = np.abs(test_state.dag() * state)**2

    # Normalize the Q function
    Q_vals /= np.max(Q_vals)

    return Q_vals, theta_grid, phi_grid

def plot_husimi_q(Q_vals, theta_grid, phi_grid, state_name="Quantum State"):
    """
    Plot the Husimi Q function on a Bloch sphere.
    """
                
    max_x, max_y, max_z, max_theta, max_phi = find_max_overlap(Q_vals, theta_grid, phi_grid)

    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the Bloch sphere
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    sphere_x = np.sin(v) * np.cos(u)
    sphere_y = np.sin(v) * np.sin(u)
    sphere_z = np.cos(v)
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color="gray", alpha=0.2, edgecolor="none")

    # Normalize Q values and map to colors
    norm = plt.Normalize(Q_vals.min(), Q_vals.max())
    colors = plt.cm.viridis(norm(Q_vals))

    # Plot the Husimi Q function on the sphere, mapping the colors
    sphere = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, alpha=0.8, shade=False)

    # Plot the 3D axes centered on the Bloch sphere
    ax.plot([-1, 1], [0, 0], [0, 0], color='black', lw=2)  # X-axis
    ax.plot([0, 0], [-1, 1], [0, 0], color='black', lw=2)  # Y-axis
    ax.plot([0, 0], [0, 0], [-1, 1], color='black', lw=2)  # Z-axis

    # Plot the coherent state's position
    ax.scatter(max_x, max_y, max_z, color='black', s=100, zorder=9)

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    mappable.set_array(Q_vals)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label(f"Husimi Q({state_name})")

    # X-axis: Line from (-1,0,0) to (1,0,0)
    ax.plot([0, 1.5], [0, 0], [0, 0], color='black', lw=1, zorder=10, linestyle='--')
    ax.text(1.5, 0, 0, 'X', color='black', fontsize=15, ha='center', va='center', zorder=10)

    # Y-axis: Line from (0,-1,0) to (0,1,0)
    ax.plot([0, 0], [0, 1.5], [0, 0], color='black', lw=1, zorder=10, linestyle='--')
    ax.text(0, 1.5, 0, 'Y', color='black', fontsize=15, ha='center', zorder=10)

    # Z-axis: Line from (0,0,-1) to (0,0,1)
    ax.plot([0, 0], [0, 0], [0, 1.1], color='black', lw=1, zorder=10, linestyle='--')
    ax.text(0, 0, 1.2, 'Z', color='black', fontsize=15, ha='center', zorder=10)

    # Customize the plot
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.set_zlabel('Z-axis', fontsize=12)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # Set the view angle to ensure the "hottest" region is visible
    ax.view_init(elev=15, azim=45)  # Adjust the view if needed
    ax.set_axis_off()
    plt.tight_layout()

    # Show the plot
    plt.show()

def find_max_overlap(Q_vals, theta_grid, phi_grid):
    """
    Find the maximum overlap coordinates in the Husimi Q function.
    """
    max_idx = np.unravel_index(np.argmax(Q_vals), Q_vals.shape)
    max_theta = theta_grid[max_idx]
    max_phi = phi_grid[max_idx]

    max_x = np.sin(max_theta) * np.cos(max_phi)
    max_y = np.sin(max_theta) * np.sin(max_phi)
    max_z = np.cos(max_theta)

    return max_x, max_y, max_z, max_theta, max_phi
