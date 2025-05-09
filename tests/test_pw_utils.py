import numpy as np
import math
from Trefftz_EPW_2D.pw_utils import build_pw_params


def test_build_pw_params_propagating():
    P_elem = 8
    thetas, zetas = build_pw_params(cx=0.0, cy=0.0, P_elem=P_elem, eps_val=1.0, zmode=1, k=2.0)
    assert isinstance(thetas, list) and isinstance(zetas, list)
    assert len(thetas) == P_elem
    # propagating only → exactly one zeta = 0
    assert len(zetas) == 1 and zetas[0] == 0.0


def test_build_pw_params_evanescent_branch():
    P_elem = 5
    thetas, zetas = build_pw_params(cx=0.5, cy=0.5, P_elem=P_elem, eps_val=2.0, zmode=0, k=3.0)
    # For Parolin distribution (zmode=0), M = ceil(sqrt(P_elem)) directions and M zetas
    M = math.ceil(math.sqrt(P_elem))
    assert isinstance(thetas, list) and isinstance(zetas, list)
    assert len(thetas) == M
    assert len(zetas) == M
