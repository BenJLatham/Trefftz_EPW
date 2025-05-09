import numpy as np
from Trefftz_EPW_2D.analytic import coeffs, sum_u


def test_coeffs_basic():
    A0, B0 = coeffs(0, epsm=1.0, k=2.0)
    # for a trivial case you might know A0≈…, B0≈…
    assert np.isfinite(A0) and np.isfinite(B0)


def test_sum_u_r_zero():
    # sum_u should be finite at the origin for any order
    val = sum_u(N=3, r=np.hypot(0.0, 0.0), k=1.0, theta=np.arctan2(0.0, 0.0), epsm=2.0)
    # If val is an array, ensure every entry is finite
    assert np.all(np.isfinite(val))
