import numpy as np
import pytest
from Trefftz_EPW_2D.quadrature import gaussian_quadrature_points_and_weights


def test_supported_rules_sum_weights():
    # For each supported n,
    # weights should sum to 0.5 (area of ref triangle)
    for n in [1, 3, 4, 6, 13]:
        pts, w = gaussian_quadrature_points_and_weights(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert pts.shape[0] == n
        assert w.shape == (n,)
        assert np.isclose(w.sum(), 0.5)


def test_unsupported_rule_raises():
    with pytest.raises(ValueError):
        gaussian_quadrature_points_and_weights(2)
