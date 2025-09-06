# tests/test_husimi.py

import pytest
from utils.husimi import compute_husimi_q, find_max_overlap
import qutip as qt
import numpy as np

def test_compute_husimi_q():
    j = 2
    state = qt.spin_coherent(j, np.pi / 2, 0)
    Q_vals, theta_grid, phi_grid = compute_husimi_q(j, state)
    assert Q_vals.shape == theta_grid.shape
    assert np.isclose(np.max(Q_vals), 1.0)

def test_find_max_overlap():
    j = 2
    state = qt.spin_coherent(j, np.pi / 2, 0)
    Q_vals, theta_grid, phi_grid = compute_husimi_q(j, state)
    max_x, max_y, max_z, max_theta, max_phi = find_max_overlap(Q_vals, theta_grid, phi_grid)
    assert -1 <= max_x <= 1
    assert -1 <= max_y <= 1
    assert -1 <= max_z <= 1
