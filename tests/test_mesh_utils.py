import numpy as np
from Trefftz_EPW_2D.mesh_utils import is_point_inside_triangle


def test_point_in_triangle():
    tri = np.array([[0, 0], [1, 0], [0, 1]])
    assert is_point_inside_triangle([0.1, 0.1], tri)
    assert not is_point_inside_triangle([1.1, 1.1], tri)
