from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from qutip import Qobj
from scipy.special import binom


@dataclass
class SurfaceData:
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    facecolors: np.ndarray


def get_husimi_bloch_surface_data(state: Union[np.ndarray, Qobj], J: int,
                                  n_theta: int = 500, n_phi: int = 500,
                                  q_alpha: float = 1.0,
                                  log_min: float = -3.0) -> SurfaceData:
    """
   Prepare the Husimi-Q surface data for a spin-J state to be plotted on a Bloch sphere.

   Parameters
   ----------
   state : ndarray or Qobj
      Quantum state vector in J_z basis, with m = -J to +J.
   J : int
      Total spin quantum number.
   n_theta : int
      Grid resolution in polar angle θ.
   n_phi : int
      Grid resolution in azimuthal angle φ.
   q_alpha : float
      Transparency of the Husimi surface (0 = fully transparent, 1 = opaque).
   log_min : float
      Minimum log10(Q) value to display (used for color scaling).

   Returns
   -------
   SurfaceData
      A dataclass containing X, Y, Z coordinates and corresponding RGBA colors.
   """
    # Convert Qobj to numpy vector
    if isinstance(state, Qobj):
        if not state.isket:
            raise ValueError("Only kets (pure states) are allowed.")
        state = state.full().ravel()[::-1]  # QuTiP stores +J…−J order

    state = np.asarray(state, dtype=np.complex128).ravel()
    if state.size != 2 * J + 1:
        raise ValueError(f"state length {state.size} ≠ 2J+1 = {2 * J + 1}")
    if not np.isclose(np.linalg.norm(state), 1.0):
        state /= np.linalg.norm(state)

    # Binomial √C(2J, J+m)
    m_vals = np.arange(-J, J + 1)
    binom_sqrt = np.sqrt(binom(2 * J, J + m_vals))

    # Grid for θ and φ
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    # Coherent state coefficients ⟨θ,φ|m⟩
    cos_th2 = np.cos(TH / 2)[..., None]
    sin_th2 = np.sin(TH / 2)[..., None]
    phase = np.exp(-1j * m_vals * PH[..., None])

    coeffs = (binom_sqrt * cos_th2**(J + m_vals) * sin_th2**(J - m_vals) *
              phase)

    # Overlap ⟨θ,φ|ψ⟩ and Q-function
    overlap = np.tensordot(coeffs, state.conj(), axes=(2, 0))
    Q = np.abs(overlap)**2
    Q /= Q.max()

    # Color scale (logarithmic)
    Q_log = np.log10(Q + np.finfo(float).eps)
    Q_log_clipped = np.clip(Q_log, log_min, 0.0)
    norm = (Q_log_clipped - log_min) / (-log_min)

    # Custom colormap
    navy = (0.07, 0.25, 0.55)
    green = (0.00, 0.63, 0.29)
    yellow = (0.96, 0.86, 0.01)
    red = (0.95, 0.18, 0.05)
    cmap = LinearSegmentedColormap.from_list("blue_hot",
                                             [navy, green, yellow, red])

    face_rgba = cmap(norm)
    face_rgba[..., -1] = q_alpha  # apply opacity

    # Convert (θ, φ) → Cartesian
    X = np.sin(TH) * np.sin(PH)
    Y = np.sin(TH) * np.cos(PH)
    Z = np.cos(TH)

    return SurfaceData(X=X, Y=Y, Z=Z, facecolors=face_rgba)


def plot_multiple_surfaces(surface_list: list[SurfaceData]) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw Bloch sphere shell
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=(0.07, 0.25, 0.55), edgecolor='k',
                    linewidth=0.2, alpha=0.2)

    # Add each surface
    for surf in surface_list:
        ax.plot_surface(surf.X, surf.Y, surf.Z, facecolors=surf.facecolors,
                        rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False)

    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=30, azim=30)
    plt.tight_layout()
    plt.show()
