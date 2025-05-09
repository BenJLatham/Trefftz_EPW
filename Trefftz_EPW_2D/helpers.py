"""
helpers.py

Miscellaneous helper functions: complex-valued sqrt with
correct casting and a stable psi(z) = (exp(z)-1)/z implementation.
"""

import numpy as np

__all__ = ["complex_sqrt", "psi"]


def complex_sqrt(val):
    """
    Compute the square root of a complex number input, ensuring the result is complex.

    Parameters:
      val : scalar or array-like
        Input value(s) to take the square root of.

    Returns:
      complex or ndarray
        Complex square root of val.
    """
    # Ensure input is converted to complex before taking sqrt
    return np.sqrt(np.array(val, dtype=np.complex128))


def psi(z):
    """
    Compute psi(z) = (exp(z) - 1)/z with stability at z ~ 0.

    For |z| below a small threshold, returns 1 to avoid division by near-zero.

    Parameters:
      z : array-like
        Input array of complex or real numbers.

    Returns:
      ndarray of complex
        The values of (exp(z) - 1)/z elementwise, with limit 1 at z=0.
    """
    z = np.asarray(z)
    # Threshold under which to approximate (exp(z)-1)/z by its limit at zero
    threshold = 1e-10
    result = np.empty_like(z, dtype=np.complex128)
    close = np.abs(z) < threshold
    # For small z, psi(z) ~ 1
    result[close] = 1.0
    # For the rest, compute directly
    not_close = ~close
    result[not_close] = np.expm1(z[not_close]) / z[not_close]
    return result
