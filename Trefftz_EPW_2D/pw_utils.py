"""
pw_utils.py

Plane-wave generation and caching utilities: generate_plane_waves_from_dist and build_pw_params.
"""

from scipy.optimize import bisect
import numpy as np
from scipy.special import gamma, gammaincc, gammainc
from joblib import Memory

# Set up a cache for plane-wave parameters
memory = Memory("./pw_cache", verbose=0)

__all__ = ["generate_plane_waves_from_dist", "build_pw_params"]


def generate_plane_waves_from_dist(R, k, N, xc=0, yc=0, troubleshoot_zeta=0, offset=0):
    """
    Generate plane waves centered at (xc, yc)
    and corresponding phi and zeta values, including their gradients.

    Parameters:
      R                : float
                         Radius (half the mesh width).
      k                : float
                         Wave number.
      N                : int
                         Number of plane waves.
      xc, yc           : float
                         Coordinates of the center.
      troubleshoot_zeta: int
                         Troubleshooting level:
                         0 (default) - Normal behavior.
                         1 - Set zeta=[1].
                         2 - Set zeta=[-2,-1, 1, 2]
                         3 - Set zeta=[-2,-1, 0, 1, 2].
                         4 - Additional mode (e.g. [0.0002,0.0001,0,-0.0001,-0.0002])
      offset           : float
                         Offset angle in radians for the first direction of phi.

    Returns:
      plane_wave_functions : list of tuples
                            Each tuple (plane_wave, plane_wave_grad)
                              where both are callables.
      phi_values          : ndarray
                            Array of direction angles.
      zeta_values         : ndarray
                            Array of evanescence parameters.
    """
    # Ensure troubleshoot_zeta is valid
    assert troubleshoot_zeta in [
        0,
        1,
        2,
        3,
        4,
    ], "troubleshoot_zeta must be 0, 1, 2, 3, or 4"

    # Determine count of directions M and zeta modes P
    if troubleshoot_zeta == 0:
        M = int(np.ceil(np.sqrt(N)))
        P = int(np.floor(N / 8))
    elif troubleshoot_zeta == 1:
        M = int(np.ceil(N))
        P = int(np.floor(N / 8))
    elif troubleshoot_zeta == 2:
        M = int(np.ceil(N))
        P = int(np.floor(N / 8))
    elif troubleshoot_zeta == 3:
        M = int(np.ceil(0.5 * N))
        P = int(np.floor(N / 8))
    elif troubleshoot_zeta == 4:
        M = int(np.ceil(0.2 * N))
        P = int(np.floor(N / 8))

    # Direction angles
    phi_values = offset + 2 * np.pi * np.arange(M) / M
    phi_values = np.mod(phi_values, 2 * np.pi)

    # Precompute constant
    abs_kR = np.abs(k) * R

    def Gamma(s, x):
        if s > 0:
            return gamma(s) * gammaincc(s, x)
        else:
            # Use recurrence to extend to negative s
            n = int(-s) + 1
            s_new = s + n
            G = gamma(s_new) * gammaincc(s_new, x)
            for j in range(n):
                s_current = s_new - j
                G = (G - x**s_current * np.exp(-x)) / s_current
            return G

    def mu_approx(p, zeta):
        # Approximate weight function for large |p|
        if np.abs(p) >= 25:
            num = gammainc(0.5 + 2 * p, abs_kR * np.exp(np.abs(zeta)))
            den = gammainc(0.5 + 2 * p, abs_kR)
            return num / den
        else:
            # Full expression
            term1 = (abs_kR) ** (2 * p) * Gamma(
                0.5 - 2 * p, abs_kR * np.exp(np.abs(zeta))
            )
            term2 = (abs_kR) ** (-2 * p) * Gamma(
                0.5 + 2 * p, abs_kR * np.exp(np.abs(zeta))
            )
            denom1 = (abs_kR) ** (2 * p) * Gamma(0.5 - 2 * p, abs_kR)
            denom2 = (abs_kR) ** (-2 * p) * Gamma(0.5 + 2 * p, abs_kR)
            return (term1 + term2) / (denom1 + denom2)

    def sum_mu(P, zeta):
        return sum(mu_approx(p, zeta) for p in range(1, int(P) + 1))

    def Upsilon(P, zeta):
        term1 = Gamma(0.5, abs_kR * np.exp(np.abs(zeta))) / (2 * Gamma(0.5, abs_kR))
        return 0.5 + np.sign(zeta) * (0.5 - (term1 + sum_mu(P, zeta)) / (2 * P + 1))

    def find_root_Upsilon(P, m, M, tol=1e-6):
        target = (m - 0.5) / M

        def f(zeta):
            return Upsilon(P, zeta) - target

        a, b = -10, 10
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            raise ValueError(f"No root for m={m} in [{a},{b}]")
        return bisect(f, a, b, xtol=tol)

    # Determine zeta values
    if troubleshoot_zeta == 1:
        zeta_values = np.array([0.0])
    elif troubleshoot_zeta == 2:
        zeta_values = np.array([-2, -1, 1, 2], dtype=float)
    elif troubleshoot_zeta == 3:
        zeta_values = np.array([-2, -1, 0, 1, 2], dtype=float)
    elif troubleshoot_zeta == 4:
        zeta_values = np.array([0.0002, 0.0001, 0.0, -0.0001, -0.0002], dtype=float)
    else:
        # Solve for each root
        zeta_values = np.zeros(M, dtype=float)
        for m in range(1, M // 2 + 1):
            zeta_values[m - 1] = find_root_Upsilon(P, m, M)
        for m in range(M // 2 + 1, M + 1):
            zeta_values[m - 1] = -zeta_values[M - m]

    # Build plane-wave functions and gradients
    plane_wave_functions = []
    for phi in phi_values:
        for zeta in zeta_values:

            def plane_wave(x, y, zeta=zeta, phi=phi):
                return np.exp(
                    (1j * k * np.cosh(zeta))
                    * (np.cos(phi) * (x - xc) + np.sin(phi) * (y - yc))
                    + (-k * np.sinh(zeta))
                    * (-np.sin(phi) * (x - xc) + np.cos(phi) * (y - yc))
                )

            def plane_wave_grad(x, y, zeta=zeta, phi=phi):
                val = plane_wave(x, y, zeta, phi)
                grad = np.array(
                    [
                        (1j * k * np.cosh(zeta)) * np.cos(phi)
                        + (-k * np.sinh(zeta)) * -np.sin(phi),
                        (1j * k * np.cosh(zeta)) * np.sin(phi)
                        + (-k * np.sinh(zeta)) * np.cos(phi),
                    ]
                )
                return val[..., np.newaxis] * grad

            # Attach metadata
            plane_wave.zeta = zeta
            plane_wave.phi = phi
            plane_wave_grad.zeta = zeta
            plane_wave_grad.phi = phi
            plane_wave_functions.append((plane_wave, plane_wave_grad))

    return plane_wave_functions, phi_values, zeta_values


@memory.cache
def build_pw_params(
    cx: float, cy: float, P_elem: int, eps_val: float, zmode: int, k: float
):
    """
    Cached generation of plane-wave direction (theta)
    and decay (zeta) lists.

    Parameters:
      cx, cy : float  – element center coordinates
      P_elem : int    – number of plane waves per element
      eps_val: float  – permittivity value
      zmode  : int    – troubleshoot_zeta mode
      k      : float  – base wave number

    Returns:
      thetas: list of floats
      zetas : list of floats
    """
    if zmode == 1:
        thetas = np.linspace(0, 2 * np.pi, P_elem, endpoint=False)
        return list(thetas), [0.0]

    # Otherwise compute via full generator
    pw, thetas, zetas = generate_plane_waves_from_dist(
        R=100.0,
        k=(eps_val**0.5) * k if eps_val != 1 else k,
        N=P_elem,
        xc=cx,
        yc=cy,
        troubleshoot_zeta=zmode,
    )
    return list(thetas), list(zetas)
