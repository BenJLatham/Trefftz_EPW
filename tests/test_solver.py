import numpy as np
import pytest
from Trefftz_EPW_2D.solver import solve_system
from scipy.sparse import identity


def test_solve_system_identity():
    # A x = b with A = I should return x = b
    n = 4
    A = identity(n, format="csr", dtype=complex)
    b = np.arange(1, n + 1, dtype=complex)
    x = solve_system(A=A, b=b, use_pardiso=False, alpha=None)
    assert np.allclose(x, b)


def test_solve_system_regularized():
    # With alpha, solves (A^H A + αI)x = A^H b
    n = 3
    A = identity(n, format="csr", dtype=complex)
    b = np.array([1, 2, 3], dtype=complex)
    alpha = 0.1
    x = solve_system(A=A, b=b, use_pardiso=False, alpha=alpha)
    # For identity, (I + αI)x = b → x = b/(1+α)
    assert np.allclose(x, b / (1 + alpha))
