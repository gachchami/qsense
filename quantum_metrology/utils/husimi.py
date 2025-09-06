# utils/husimi.py
"""Husimi Q-function calculation and visualization module.

This module provides functions for computing and visualizing the Husimi Q-function
representation of quantum states on the Bloch sphere.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D backen
from scipy.special import binom


def plot_husimi_bloch(
        state,
        J: int,
        ax,
        n_theta: int = 500,
        n_phi: int = 500,
        sphere_alpha: float = 0.25,
        q_alpha: float = 1.0,  # fully opaque hotspot
        title: str | None = None) -> Tuple[Figure, Axes3D]:
    """
   Plot the Husimi-Q distribution of a pure spin-J state on a Bloch sphere.

   Parameters
   ----------
   state
      NumPy array (length 2 J+1, m = −J…+J).
   J
      Total spin quantum number.
   n_theta , n_phi
      Resolution of the (θ, φ) evaluation grid.
   sphere_alpha
      Transparency of the navy shell (0 = invisible, 1 = opaque).
   q_alpha
      Transparency of the coloured Husimi layer.
   title
      Custom title; default is ``Husimi-Q on Bloch sphere (J = …)``.

   Returns
   -------
   fig , ax
      The Matplotlib figure and its 3-D axes object.
   """
    # ---- sanity checks -------------------------------------------------
    state = np.asarray(state, dtype=np.complex128).ravel()
    if state.size != 2 * J + 1:
        raise ValueError(f"state length {state.size} ≠ 2J+1 = {2 * J + 1}")
    if not np.isclose(state.conj() @ state, 1.0):
        state = state / np.linalg.norm(state)

    # ---- combinatorial prefactors √C(2J, J+m) --------------------------
    m_vals = np.arange(-J, J + 1)
    binom_sqrt = np.sqrt(binom(2 * J, J + m_vals))  # shape (2J+1,)

    # ---- θ-φ grids -----------------------------------------------------
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")  # (n_theta, n_phi)
    # fig = plt.figure(figsize=(12, 8), facecolor="w")
    # ---- coefficients c_m(θ,φ) ----------------------------------------
    cos_th2 = np.cos(TH / 2)[..., None]  # add m-axis
    sin_th2 = np.sin(TH / 2)[..., None]

    exponent_cos = J + m_vals
    exponent_sin = J - m_vals
    phase = np.exp(-1j * m_vals * PH[..., None])  # e^{−i m φ}

    coeffs = (binom_sqrt * cos_th2**exponent_cos * sin_th2**exponent_sin * phase
             )  # (..., 2J+1)

    # ---- overlap ⟨θ,φ|ψ⟩ and Q-function -------------------------------
    print(coeffs.shape, state.shape)
    overlap = np.tensordot(coeffs, state.conj(), axes=(2, 0))
    Q = np.abs(overlap)**2
    Q /= Q.max()

    # ---- logarithmic colour scale (top two decades) --------------------
    Qlog = np.log10(Q + np.finfo(float).eps)
    cmax, cmin = 0, -3

    navy = (0.07, 0.25, 0.55)  # sphere colour  (R,G,B)
    green = (0.00, 0.63, 0.29)
    yellow = (0.96, 0.86, 0.01)
    red = (0.95, 0.18, 0.05)

    cmap = LinearSegmentedColormap.from_list("blue_hot",
                                             [navy, green, yellow, red], N=256)

    # ---- θ,φ → Cartesian with x↔y swap ---------------------------------
    X = np.sin(TH) * np.sin(PH)
    Y = np.sin(TH) * np.cos(PH)
    Z = np.cos(TH)

    # ---- figure --------------------------------------------------------
    ax.patch.set_facecolor("white")

    # transparent Bloch sphere shell
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, facecolor=navy, edgecolor="k", cstride=16,
                    linewidth=0.05, alpha=sphere_alpha)

    # Husimi-Q surface
    normed = (Qlog - cmin) / (cmax - cmin)
    fc = cmap(normed)
    ax.plot_surface(X, Y, Z, facecolors=fc, rcount=n_theta, ccount=n_phi,
                    antialiased=True, linewidth=0, zorder=2, alpha=q_alpha,
                    shade=False)

    # reference axes (quiver arrows)
    b = "black"
    axis_len = [1.75, 1.75, 1.75]
    linestyles = ["dashed", "dashed", "dashed"]
    quiv_opts = [
        dict(
            length=axis_len[i],
            arrow_length_ratio=0.05,
            linewidth=0.5,
            linestyle=linestyles[i],
            normalize=False,
        ) for i in range(len(axis_len))
    ]
    ax.quiver(0, 0, 0, 0, 1, 0, color=b, **quiv_opts[0])  # X
    ax.quiver(0, 0, 0, 0, -1, -1, color=b, **quiv_opts[1], label="Y")  # Y
    ax.quiver(0, 0, 0, 0, 0, 1, color=b, **quiv_opts[2])  # Z
    ax.text(0, axis_len[0] + 0.3, 0, "  X", color=b, fontsize=12, weight="bold")
    ax.text(0, -axis_len[1] - 0.3, -axis_len[1], "  Y", color=b, fontsize=12,
            weight="bold")
    ax.text(0, 0, axis_len[2], "  Z", color=b, fontsize=12, weight="bold")

    # colour bar
    # mappable = plt.cm.ScalarMappable(cmap=cmap)
    # mappable.set_clim(cmin, cmax)
    # cb = fig.colorbar(mappable, shrink=0.8, pad=0.05, ax=ax)
    # cb.set_ticks([cmin, cmin + 1, cmax])
    # cb.set_ticklabels(
    #     [r"$10^{-2}$ (low overlap)", r"$10^{-1}$", r"$10^{0}$ (high overlap)"])
    # cb.set_label(r"$\log_{10} Q(\theta,\varphi)$", fontsize=12)

    # viewpoint and limits
    ax.view_init(elev=30, azim=30)
    rng = 1.35
    ax.set_xlim([-rng, rng])
    ax.set_ylim([-rng, rng])
    ax.set_zlim([-rng, rng])
    try:  # Matplotlib ≥3.3
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    ax.set_axis_off()
    return ax


def qobj_to_coeffs(qobj: qt.Qobj) -> np.ndarray:
    if not qobj.isket:  # the routine only handles pure kets
        raise TypeError("state must be a ket Qobj")
    vec = qobj.full().ravel()  # dense NumPy array (order +J … −J)
    return vec[::-1]  # reverse → −J … +J


def plot_all(J: int, states):
    fig, axis = plt.subplots(5, 1, figsize=(6, 30),
                             subplot_kw={'projection': '3d'})
    for i in range(len(states)):
        plot_husimi_bloch(J=J, state=qobj_to_coeffs(states[i]), ax=axis[i])

    #axis.set_title("Husimi-Q on Bloch sphere", fontsize=14)
    plt.title("Husimi Q-function for", fontsize=14)
    plt.tight_layout()
    plt.show()
