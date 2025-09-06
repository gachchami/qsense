# tests/test_operators.py

import pytest
from utils.operators import J_n
import qutip as qt
import numpy as np

def test_J_n():
    Jy = qt.jmat(2, 'y')
    Jz = qt.jmat(2, 'z')
    theta = np.pi / 4
    result = J_n(theta, Jy, Jz)
    assert result is not None
