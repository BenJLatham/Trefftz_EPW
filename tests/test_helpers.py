import numpy as np
from Trefftz_EPW_2D.helpers import complex_sqrt, psi


def test_complex_sqrt_real_and_negative():
    assert complex_sqrt(4) == 2.0
    val = complex_sqrt(-1)
    assert isinstance(val, complex)
    assert np.isclose(val.real, 0)
    assert np.isclose(val.imag, 1)


def test_psi_zero_and_small_z():
    # psi(0) should return 1 exactly
    assert psi(0) == 1.0
    # for small z, psi(z) ≈ 1 + z/2
    z = 1e-12 + 1e-12j
    approx = psi(z)
    assert np.allclose(approx, 1.0 + z / 2, rtol=1e-6, atol=0)
