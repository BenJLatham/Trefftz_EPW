import pytest
from Trefftz_EPW_2D.core import Trefftz2d


def test_core_import_and_defaults():
    solver = Trefftz2d(k=1.0, epsilon=2.0, order=3)
    # Check that attributes exist and are initialized to None or correct type
    assert solver.k == 1.0
    assert solver.epsilon == 2.0
    assert solver.order == 3
    assert solver.mesh is None
    assert solver.solution is None


@pytest.mark.parametrize("mesh_type", ["disk", "square"])
def test_core_quadrature_rule(mesh_type):
    solver = Trefftz2d(k=1.0, epsilon=1.0, order=1)
    # quadrature_rule should return pts, w of matching length
    pts, w = solver.quadrature_rule(1)
    assert pts.shape == (1, 2)
    assert w.shape == (1,)
