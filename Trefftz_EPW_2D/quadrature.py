"""
quadrature.py

Quadrature rules for PWDG: Gaussian quadrature on reference triangle.
"""

import numpy as np

__all__ = ["gaussian_quadrature_points_and_weights"]


def gaussian_quadrature_points_and_weights(n):
    """
    Return quadrature points and weights for the reference triangle based on a chosen rule.

    The reference triangle is the set
       {(ξ,η) : ξ >= 0, η >= 0, ξ + η <= 1}.
    Weights are scaled to sum to 0.5 (the triangle's area).

    Supported rules:
      - n = 1, 3, 4, 6, 13

    Parameters:
      n : int
        Number of quadrature points.

    Returns:
      points  : ndarray of shape (n, 2)
        The (ξ,η) coordinates of quadrature points.
      weights : ndarray of shape (n,)
        The corresponding quadrature weights.

    Raises:
      ValueError if n is not one of the supported values.
    """
    if n == 1:
        # 1-point rule: use the barycenter.
        points = np.array([[1 / 3, 1 / 3]])
        weights = np.array([0.5])

    elif n == 3:
        # 3-point rule (degree 2)
        points = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        # weights sum to 0.5
        weights = np.array([1 / 6, 1 / 6, 1 / 6])

    elif n == 4:
        # 4-point rule (degree 3, one negative weight)
        points = np.array([[1 / 3, 1 / 3], [0.2, 0.2], [0.6, 0.2], [0.2, 0.6]])
        weights = np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])
        # scale to sum to 0.5
        weights = weights * (0.5 / np.sum(weights))

    elif n == 6:
        # 6-point Dunavant rule (degree 4)
        points = np.array(
            [
                [0.4459484909, 0.4459484909],
                [0.4459484909, 0.1081030182],
                [0.1081030182, 0.4459484909],
                [0.0915762135, 0.0915762135],
                [0.0915762135, 0.8168475729],
                [0.8168475729, 0.0915762135],
            ]
        )
        weights = np.array(
            [
                0.2233815897,
                0.2233815897,
                0.2233815897,
                0.1099517437,
                0.1099517437,
                0.1099517437,
            ]
        )
        # scale to sum to 0.5
        weights = weights * (0.5 / np.sum(weights))

    elif n == 13:
        # 13-point rule
        points = np.array(
            [
                [0.333333333333333, 0.333333333333333],
                [0.479308067841920, 0.260345966079040],
                [0.260345966079040, 0.479308067841920],
                [0.260345966079040, 0.260345966079040],
                [0.869739794195568, 0.065130102902216],
                [0.065130102902216, 0.869739794195568],
                [0.065130102902216, 0.065130102902216],
                [0.500000000000000, 0.000000000000000],
                [0.500000000000000, 0.500000000000000],
                [0.000000000000000, 0.500000000000000],
                [0.333333333333333, 0.000000000000000],
                [0.666666666666667, 0.333333333333333],
                [0.000000000000000, 0.333333333333333],
            ]
        )
        weights = np.array(
            [
                -0.149570044467670,
                0.175615257433204,
                0.175615257433204,
                0.175615257433204,
                0.053347235608838,
                0.053347235608838,
                0.053347235608838,
                0.077113760890257,
                0.077113760890257,
                0.077113760890257,
                0.097135796801733,
                0.097135796801733,
                0.097135796801733,
            ]
        )
        weights = weights * (0.5 / np.sum(weights))

    else:
        raise ValueError(f"Quadrature rule for n = {n} is not implemented.")

    return points, weights
