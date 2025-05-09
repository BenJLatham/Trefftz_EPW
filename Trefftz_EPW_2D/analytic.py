"""
analytic.py

Analytic disk-scattering solution: compute Bessel/Hankel coefficients and field sums.
"""

import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp, iv, ivp

__all__ = [
    "coeffs",
    "compute_coeffs_array",
    "GenerateWeight_array",
    "u_ana",
    "sum_u",
]


def coeffs(mode, epsm, k, theta0=np.pi / 2):
    """
    Compute the scattering coefficients A_mode and B_mode for a given angular mode.

    Parameters:
      mode   : integer mode number.
      epsm   : relative permittivity of the inclusion (can be negative).
      k      : wavenumber in the exterior medium.
      theta0 : incident angle in radians (default pi/2).

    Returns:
      A_mode, B_mode : complex coefficients for the Hankel and Bessel expansions.
    """
    # Exterior field at r=1
    jl = (1j) ** mode * jv(mode, k) * np.exp(-1j * mode * theta0)
    djl = (1j) ** mode * jvp(mode, k) * np.exp(-1j * mode * theta0)
    h = hankel1(mode, k)
    dh = h1vp(mode, k, 1)

    # Build system depending on sign of epsm
    if epsm <= 0:
        kmm = k * np.sqrt(-epsm)
        il_ = iv(mode, kmm)
        dil_ = ivp(mode, kmm)
        M_l = np.array([[h, -il_], [dh, -(epsm ** (-1)) * np.sqrt(-epsm) * dil_]])
    else:
        km = k * np.sqrt(epsm)
        jlm = jv(mode, km)
        djlm = jvp(mode, km)
        M_l = np.array([[h, -jlm], [dh, -(epsm ** (-1)) * np.sqrt(epsm) * djlm]])

    # Right-hand side from incident field
    J_l = np.array([[-jl], [-djl]])

    # Solve 2×2 system for coefficients
    A_mode, B_mode = np.linalg.inv(M_l) @ J_l
    return A_mode[0], B_mode[0]


def compute_coeffs_array(ns, epsm, k, theta0=np.pi / 2):
    """
    Compute arrays of scattering coefficients A_n and B_n for modes in ns.

    Parameters:
      ns     : sequence of integer mode indices.
      epsm   : relative permittivity (can be negative).
      k      : wavenumber.
      theta0 : incident angle in radians.

    Returns:
      A, B : arrays of complex coefficients, each of length len(ns).
    """
    A = np.empty(len(ns), dtype=complex)
    B = np.empty(len(ns), dtype=complex)
    for i, n in enumerate(ns):
        A[i], B[i] = coeffs(n, epsm, k, theta0)
    return A, B


def GenerateWeight_array(ns, r, k, epsm, theta0=np.pi / 2):
    """
    Vectorized computation of the radial weight for each mode and radius.

    Parameters:
      ns     : sequence of integer mode indices.
      r      : array-like of radii (scalar or array).
      k      : wavenumber.
      epsm   : relative permittivity (can be negative).
      theta0 : incident angle in radians.

    Returns:
      result : complex array of shape (len(ns), *r.shape) giving radial weights.
    """
    A, B = compute_coeffs_array(ns, epsm, k, theta0)
    r = np.array(r)
    scalar_input = np.isscalar(r)
    r = np.array(r, ndmin=1)
    result = np.empty((len(ns),) + r.shape, dtype=complex)

    # Region outside the inclusion (r >= 1)
    mask = r >= 1.0
    if np.any(mask):
        for i, n in enumerate(ns):
            result[i][mask] = A[i] * hankel1(n, k * r[mask]) + (1j) ** n * np.exp(
                -1j * n * theta0
            ) * jv(n, k * r[mask])
    # Region inside the inclusion (r < 1)
    if np.any(~mask):
        if epsm >= 0:
            km = k * np.sqrt(epsm)
            for i, n in enumerate(ns):
                result[i][~mask] = B[i] * jv(n, km * r[~mask])
        else:
            kmm = k * np.sqrt(-epsm)
            for i, n in enumerate(ns):
                result[i][~mask] = B[i] * iv(n, kmm * r[~mask])
    if scalar_input:
        return result.squeeze()
    return result


def u_ana(ns, r, k, theta, epsm, theta0=np.pi / 2):
    """
    Compute the modal contributions u_n(r, theta) for modes in ns.

    Parameters:
      ns     : sequence of integer mode indices.
      r      : array-like of radii.
      theta  : array-like of angles (radians).
      k      : wavenumber.
      epsm   : relative permittivity (can be negative).
      theta0 : incident angle in radians.

    Returns:
      complex array of shape (len(ns), *r.shape) of modal fields.
    """
    radial = GenerateWeight_array(ns, r, k, epsm, theta0)
    # Align ns and theta shapes for phase factor
    ns_arr = np.array(ns)
    new_shape = (len(ns),) + (1,) * np.ndim(theta)
    ns_reshaped = ns_arr.reshape(new_shape)
    theta_expanded = theta[None, ...]
    phase = np.exp(1j * ns_reshaped * theta_expanded)
    return radial * phase


def sum_u(N, r, k, theta, epsm, theta0=np.pi / 2):
    """
    Sum the series of modal contributions from n=-N to n=+N.

    Parameters:
      N      : non-negative integer, highest mode index.
      r      : array-like of radii.
      theta  : array-like of angles (radians).
      k      : wavenumber.
      epsm   : relative permittivity (can be negative).
      theta0 : incident angle in radians.

    Returns:
      complex array of shape r.shape of the total scattered field.
    """
    ns = np.arange(-N, N + 1)
    u_modes = u_ana(ns, r, k, theta, epsm, theta0)
    return np.sum(u_modes, axis=0)
